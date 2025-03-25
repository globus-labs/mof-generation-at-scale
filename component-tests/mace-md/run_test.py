from concurrent.futures import as_completed
import argparse
import json
from pathlib import Path

import pandas as pd
import parsl
from parsl import Config, HighThroughputExecutor
from parsl.app.python import PythonApp
from platform import node
from ase import Atoms
from parsl.launchers import MpiExecLauncher
from parsl.providers import PBSProProvider
from tqdm import tqdm

from mofa.scoring.geometry import LatticeParameterChange
from mofa.model import MOFRecord
from mofa.utils.conversions import write_to_string


def test_function(mof: MOFRecord, timesteps: int, device: str) -> tuple[float, list[Atoms]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        mof: MOF to use
        timesteps: Number of MD time steps
        device: Which device to use for execution
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from mofa.simulation.mace import MACERunner, load_model
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'run-{timesteps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # Run
    load_model(device)  # Cache the model _before_ we run MD
    runner = MACERunner(run_dir=run_dir, device=device)
    start_time = perf_counter()
    output = runner.run_molecular_dynamics(mof, timesteps, timesteps // 5)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help='Number of timesteps to run', default=1000, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    parser.add_argument('--device', help='Which device to use for executing LAMMPS')
    parser.add_argument('--max-repeats', help='How many repeats to perform at most', default=None, type=int)
    args = parser.parse_args()

    # Select the correct configuration
    if args.config == "local":
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1, cpu_affinity='block')])
    elif args.config.startswith("aurora"):
        # Map processes to specific tiles or whole devices
        if args.config == "aurora":
            accel_count = 12
        elif args.config == "aurora-device":
            accel_count = 6
        else:
            raise ValueError(f'Not supported: {args.config}')

        # Ensures processes are mapped to physical cores
        workers_per_socket = accel_count // 2
        cores_per_socket = 52
        cores_per_worker = cores_per_socket // workers_per_socket
        assigned_cores = []
        for socket in range(2):
            start = cores_per_socket * socket
            assigned_cores.extend(f"{start + w * cores_per_worker}-{start + (w + 1) * cores_per_worker - 1}" for w in range(workers_per_socket))

        config = Config(
            retries=2,
            executors=[
                HighThroughputExecutor(
                    label="aurora_test",
                    available_accelerators=accel_count,
                    cpu_affinity='list:' + ":".join(assigned_cores),  # Assigns cpus in sequent
                    prefetch_capacity=0,
                    max_workers_per_node=accel_count,
                    cores_per_worker=208 // accel_count,
                    provider=PBSProProvider(
                        account="MOFA",
                        queue="debug",
                        worker_init=f"""
        module load frameworks
        source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
        export ZE_FLAT_DEVICE_HIERARCHY={'FLAT' if accel_count == 12 else 'COMPOSITE'}
        cd $PBS_O_WORKDIR
        pwd
        which python
        hostname
                                """,
                        walltime="1:00:00",
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        scheduler_options="#PBS -l filesystems=home:flare",
                        nodes_per_block=1,
                        min_blocks=0,
                        max_blocks=1,
                        cpus_per_node=208,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f'No configuration defined for {args.config}')

    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(test_function)

        # Determine which MOFs to skip
        runtimes_path = Path('runtimes.json')
        to_skip = set()
        if args.max_repeats is not None and runtimes_path.is_file():
            count_run = pd.read_json(runtimes_path, lines=True).query(f'timesteps == {args.timesteps}')['mof'].value_counts()
            for name, count in count_run:
                if count >= args.max_repeats:
                    to_skip.add(name)

        # Submit each MOF
        futures = []
        with open('../lammps-md/example-mofs.json') as fp:
            for line in fp:
                mof = MOFRecord(**json.loads(line))
                if mof.name not in to_skip:
                    future = test_app(mof, args.timesteps, args.device)
                    future.mof = mof
                    futures.append(future)

        # Store results
        scorer = LatticeParameterChange()
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                print(f'{future.mof.name} failed: {future.exception()}')
                continue
            runtime, traj = future.result()

            # Get the strain
            # TODO (wardlt): Simplify how we compute strain
            traj_vasp = [(i, write_to_string(t, 'vasp')) for i, t in traj]
            mof = future.mof
            mof.md_trajectory['uff'] = traj_vasp
            strain = scorer.score_mof(mof)

            # Store the result
            with open(runtimes_path, 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'timesteps': args.timesteps,
                    'mof': mof.name,
                    'runtime': runtime,
                    'strain': strain,
                    'device': args.device,
                    'config': args.config,
                }), file=fp)
