"""Test LAMMPS by running a large number of MD simulations with different runtimes"""
from pathlib import Path
from platform import node
import argparse
import json
import gzip

from tqdm import tqdm
from ase import Atoms
import pandas as pd
import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor, MPIExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher

from mofa.model import MOFRecord
from mofa.simulation.cp2k import compute_partial_charges
from mofa.simulation.mace import MACERunner
from mofa.utils.conversions import write_to_string


def run_cp2k(strc: MOFRecord, cp2k_invocation: str, steps: int) -> tuple[float, tuple[Atoms, Path]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        strc: MOF to use
        cp2k_invocation: Command to invoke CP2K
        steps: Number of optimization steps
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from mofa.simulation.cp2k import CP2KRunner
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'cp2k-run-{steps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # Run
    runner = CP2KRunner(cp2k_invocation, run_dir=run_dir, close_cp2k=True)
    start_time = perf_counter()
    output = runner.run_optimization(strc, steps=steps)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranks-per-node', help='Number of CP2K ranks to deploy per node', type=int, default=4)
    parser.add_argument('--num-nodes', help='Number of nodes to use per computation', type=int, default=1)
    parser.add_argument('--steps', help='Number of optimization steps to run', default=4, type=int)
    parser.add_argument('--num-to-run', help='Number of MOFs to evaluate', default=4, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    # Select the correct configuration
    if args.config == "local":
        assert args.num_nodes == 1, 'Only support 1 node for local config'
        cp2k_cmd = (f'env OMP_NUM_THREADS={12 // args.ranks_per_node} /usr/bin/mpiexec -np {args.ranks_per_node}'
                    f' /home/lward/Software/cp2k-2024.1/exe/local/cp2k_shell.psmp')
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1)])
    elif args.config == "polaris":
        cp2k_cmd = (f'mpiexec -n $PARSL_NUM_NODES --ppn {args.ranks_per_node} --hosts $PARSL_MPI_NODELIST'
                    f' --cpu-bind depth --depth {32 // args.ranks_per_node} -env OMP_NUM_THREADS={32 // args.ranks_per_node} '
                    '/lus/eagle/projects/ExaMol/cp2k-2024.1/set_affinity_gpu_polaris.sh '
                    '/lus/eagle/projects/ExaMol/cp2k-2024.1/exe/local_cuda/cp2k_shell.psmp')
        config = Config(retries=1, executors=[
            MPIExecutor(
                label='mpi',
                max_workers_per_block=32 // args.num_nodes,
                provider=PBSProProvider(
                    launcher=SimpleLauncher(),
                    account='ExaMol',
                    queue='prod',
                    select_options="ngpus=4",
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""
module reset
module use /soft/modulefiles 
module swap PrgEnv-nvhpc PrgEnv-gnu
module load cray-fftw
module load cudatoolkit-standalone/12.2
module load cray-libsci
module list
source activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris

# Launch MPS daemon
NNODES=`wc -l < $PBS_NODEFILE`
mpiexec -n ${NNODES} --ppn 1 /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/bin/enable_mps_polaris.sh &

cd $PBS_O_WORKDIR
pwd
which python
hostname
                    """,
                    nodes_per_block=32,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    cpus_per_node=32,
                    walltime="6:00:00",
                )
            )
        ])
    elif args.config == "aurora":
        assert args.ranks_per_node == 12, 'We only support 1 rank per tile on Aurora'
        cp2k_cmd = (f'mpiexec -n {args.num_nodes * args.ranks_per_node} --ppn {args.ranks_per_node}'
                    f' --cpu-bind depth --depth={104 // args.ranks_per_node} -env OMP_NUM_THREADS={104 // args.ranks_per_node} '
                    '--env OMP_PLACES=cores '
                    '/lus/flare/projects/MOFA/lward/mof-generation-at-scale/bin/cp2k_shell')
        config = Config(
            retries=2,
            executors=[
                HighThroughputExecutor(
                    label="sunspot_test",
                    prefetch_capacity=0,
                    max_workers_per_node=1,
                    provider=PBSProProvider(
                        account="MOFA",
                        queue="debug",
                        worker_init="""
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
cd $PBS_O_WORKDIR

pwd
which python
hostname
                                """,
                        walltime="1:00:00",
                        launcher=SimpleLauncher(),
                        scheduler_options="#PBS -l filesystems=home:flare",
                        nodes_per_block=args.num_nodes,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel batch jobs
                        cpus_per_node=208,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Make the MACE runner
    runner = MACERunner(
        model_path=Path('../../input-files/mace/mace-mp0_medium'),
        run_dir=Path('mace-relax'),
    )

    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(run_cp2k)
        relax_app = PythonApp(runner.run_optimization)

        # Gather MOFs from an example set
        example_set = pd.read_csv('raw-data/ZnNCO_hMOF_cat0_valid_ads_angle_clean.csv').sample(frac=1., random_state=1).head(args.num_to_run)
        example_set['name'] = example_set['cifname'].apply(lambda x: x[:-4])
        with gzip.open('data/hmof.json.gz', 'rb') as fp:
            structures = dict((x['name'], x) for x in map(json.loads, fp) if x['identifiers']['name'] in set(example_set['name'].tolist()))
        print(f'Found {len(structures)} to evaluate')

        # Load previous results
        if Path('charges.jsonl').exists():
            prev_runs = [(x, y) for x, y in pd.read_json('charges.jsonl', lines=True)[['mof', 'steps']].values]
            print(f'Found {len(prev_runs)} previous runs')
        else:
            prev_runs = []

        # Submit each MOF
        for mof in tqdm(structures.values(), total=len(structures)):
            mof = MOFRecord(**mof)
            if (mof.name, args.steps) not in prev_runs:

                # First relax
                atoms, _ = relax_app(mof, 'default', steps=1024, fmax=0.5).result()
                mof.structure = write_to_string(atoms, 'vasp')

                # Then submit CP2K
                future = test_app(mof, cp2k_cmd, args.steps)
                runtime, (atoms, run_path) = future.result()

                # Get the strain
                try:
                    charges = compute_partial_charges(run_path).arrays['q'].tolist()
                except:
                    charges = None
                # Store the result
                with open('charges.jsonl', 'a') as fp:
                    print(json.dumps({
                        'host': node(),
                        'nodes': args.num_nodes,
                        'ranks-per-node': args.ranks_per_node,
                        'cp2k_cmd': cp2k_cmd,
                        'steps': args.steps,
                        'mof': mof.name,
                        'runtime': runtime,
                        'charges': charges,
                        'strc': atoms
                    }), file=fp)
