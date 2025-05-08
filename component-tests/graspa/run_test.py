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
from parsl.launchers import MpiExecLauncher


def test_function(path: Path, invocation: list[str], timesteps: int) -> tuple[float, list[Atoms]]:
    """Run a gRASPA simulation, report runtime and resultant capacities

    Args:
        path: Path to a directory containing DDEC output
        invocation: Command to invoke gRASPA
        timesteps: Number of GCMC time steps
    Returns:
        - Runtime (s)
        - Gas capacities
    """
    from mofa.simulation.raspa.graspa_sycl import GRASPASyclRunner
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'run-{timesteps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # Run
    name = path.name[:12]
    runner = GRASPASyclRunner(invocation, run_dir)
    start_time = perf_counter()
    output = runner.run_gcmc(name, path, 'CO2', 300, 1e4, n_cycle=timesteps)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help='Number of timesteps to run', default=100000, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    # Select the correct configuraion
    if args.config == "local":
        raspa_cmd = ['/home/lward/Software/lammps-2Aug2023/build/lmp', '-sf', 'omp']
        config = Config(executors=[HighThroughputExecutor(max_workers=1, cpu_affinity='block')])
    elif args.config == "polaris":
        raspa_cmd = (
            '/lus/eagle/projects/MOFA/lward/lammps-29Aug2024/build-gpu-nompi-mixed/lmp '
            '-sf gpu -pk gpu 1'
        ).split()
        config = Config(retries=4, executors=[
            HighThroughputExecutor(
                max_workers_per_node=4,
                cpu_affinity='block-reverse',
                available_accelerators=4,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    account='MOFA',
                    queue='debug',
                    select_options="ngpus=4",
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""
module list
source activate /lus/eagle/projects/MOFA/lward/mof-generation-at-scale/env

cd $PBS_O_WORKDIR
pwd
which python
hostname
                    """,
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    cpus_per_node=32,
                    walltime="1:00:00",
                )
            )
        ])
    elif args.config == "aurora":
        raspa_cmd = ('/lus/flare/projects/MOFA/lward/gRASPA/graspa-sycl/bin/sycl.out',)
        config = Config(
            retries=2,
            executors=[
                HighThroughputExecutor(
                    label="sunspot_test",
                    available_accelerators=12,  # Ensures one worker per accelerator
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers_per_node=12,
                    cores_per_worker=16,
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
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),  # EnsureDs 1 manger per node and allows it to divide work among all 208 threads
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

        # Submit each MOF in cp2k-runs
        futures = []
        for path in Path('cp2k-runs').rglob('DDEC6_even_tempered_net_atomic_charges.xyz'):
            future = test_app(path.parent, raspa_cmd, args.timesteps)
            future.path = path
            futures.append(future)

        # Store results
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                print(f'{future.mof.name} failed: {future.exception}')
                continue
            runtime, output = future.result()

            # Store the result
            with open('runtimes.json', 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'raspa_cmd': raspa_cmd,
                    'path': str(path.parent),
                    'timesteps': args.timesteps,
                    'runtime': runtime,
                    'output': output
                }), file=fp)
