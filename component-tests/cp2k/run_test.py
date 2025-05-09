"""Test LAMMPS by running a large number of MD simulations with different runtimes"""
from concurrent.futures import as_completed
from pathlib import Path
from platform import node
import argparse
import json

from tqdm import tqdm
from ase import Atoms
import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher

from mofa.model import MOFRecord
from mofa.utils.conversions import write_to_string


def test_function(strc: MOFRecord, cp2k_invocation: str, steps: int) -> tuple[float, tuple[Atoms, Path]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        strc: MOF to use
        cp2k_invocation: Command to invoke CP2K
        steps: Number of optimization steps
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from mofa.simulation.dft.cp2k import CP2KRunner
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'run-{steps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # Run
    runner = CP2KRunner(cp2k_invocation, run_dir=run_dir)
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
                    f' /home/lward/Software/cp2k-lward-fork/exe/local/cp2k_shell.psmp')
        config = Config(executors=[HighThroughputExecutor(max_workers=1)])
    elif args.config == "polaris":
        cp2k_cmd = (f'mpiexec -n {args.num_nodes * args.ranks_per_node} --ppn {args.ranks_per_node}'
                    f' --cpu-bind depth --depth {32 // args.ranks_per_node} -env OMP_NUM_THREADS={32 // args.ranks_per_node} '
                    '/lus/eagle/projects/MOFA/lward/cp2k-2025.1/set_affinity_gpu_polaris.sh '
                    '/lus/eagle/projects/MOFA/lward/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp')
        config = Config(retries=1, executors=[
            HighThroughputExecutor(
                max_workers_per_node=1,
                provider=PBSProProvider(
                    launcher=SimpleLauncher(),
                    account='MOFA',
                    queue='debug',
                    select_options="ngpus=4",
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""
module list
source activate /lus/eagle/projects/MOFA/lward/mof-generation-at-scale/env

# Launch MPS daemon
NNODES=`wc -l < $PBS_NODEFILE`
#mpiexec -n ${NNODES} --ppn 1 /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/bin/enable_mps_polaris.sh &

cd $PBS_O_WORKDIR
pwd
which python
hostname
                    """,
                    nodes_per_block=args.num_nodes,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    cpus_per_node=32,
                    walltime="1:00:00",
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
                        nodes_per_block=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel batch jobs
                        cpus_per_node=208,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(test_function)

        # Submit each MOF
        futures = []
        with open('../lammps-md/example-mofs.json') as fp:
            for line, _ in zip(fp, range(args.num_to_run)):
                mof = MOFRecord(**json.loads(line))
                future = test_app(mof, cp2k_cmd, args.steps)
                future.mof = mof
                futures.append(future)

        # Store results
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                print(f'{future.mof.name} failed: {future.exception()}')
                continue
            runtime, (atoms, run_path) = future.result()

            # Get the strain
    #        charges = compute_partial_charges(run_path).arrays['q']
            # Store the result
            with open('runtimes.json', 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'nodes': args.num_nodes,
                    'ranks-per-node': args.ranks_per_node,
                    'dft_cmd': cp2k_cmd,
                    'steps': args.steps,
                    'mof': future.mof.name,
                    'runtime': runtime,
    #                'charges': charges.tolist(),
                    'strc': write_to_string(atoms, 'vasp')
                }), file=fp)
