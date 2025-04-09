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
from mofa.scoring.geometry import LatticeParameterChange
from mofa.simulation.cp2k import compute_partial_charges
from mofa.utils.conversions import write_to_string


def test_function(strc: MOFRecord, invocation: str, steps: int) -> tuple[float, tuple[Atoms, Path]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        strc: MOF to use
        cp2k_invocation: Command to invoke PWDFT
        steps: Number of optimization steps
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from mofa.simulation.pwdft import PWDFTRunner
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'run-{steps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # Run
    runner = PWDFTRunner(pwdft_cmd=invocation, run_dir=run_dir)
    start_time = perf_counter()
    output = runner.run_optimization(strc, steps=steps)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-nodes', help='Number of nodes to use per computation', type=int, default=1)
    parser.add_argument('--steps', help='Number of optimization steps to run', default=4, type=int)
    parser.add_argument('--num-to-run', help='Number of MOFs to evaluate', default=4, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    # Select the correct configuration
    if args.config == "local":
        assert args.num_nodes == 1, 'Only support 1 node for local config'
        pwdft_cmd = 'mpirun -np 12 /home/lward/Software/PWDFT/build/pwdft'
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1)])
    elif args.config == "aurora":
        assert args.ranks_per_node == 12, 'We only support 1 rank per tile on Aurora'
        pwdft_cmd = (f'mpiexec -n {args.num_nodes * args.ranks_per_node} --ppn {args.ranks_per_node}'
                    f' --cpu-bind depth --depth={104 // args.ranks_per_node} -env OMP_NUM_THREADS={104 // args.ranks_per_node} '
                    '--env OMP_PLACES=cores '
                    '/lus/flare/projects/MOFA/lward/PWDFT/build_sycl/pwdft')
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
                future = test_app(mof, pwdft_cmd, args.steps)
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
            with open('runtimes.jsonl', 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'nodes': args.num_nodes,
                    'ranks-per-node': args.ranks_per_node,
                    'cp2k_cmd': cp2k_cmd,
                    'steps': args.steps,
                    'mof': future.mof.name,
                    'runtime': runtime,
    #                'charges': charges.tolist(),
                    'strc': write_to_string(atoms, 'vasp')
                }), file=fp)
