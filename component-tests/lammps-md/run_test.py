"""Test LAMMPS by running a large number of MD simulations with different runtimes"""
from concurrent.futures import as_completed
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

from mofa.model import MOFRecord
from mofa.scoring.geometry import LatticeParameterChange
from mofa.utils.conversions import write_to_string


def test_function(mof: MOFRecord, lammps_invocation: list[str], timesteps: int, environ: dict | None = None) -> tuple[float, list[Atoms]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        strc: MOF to use
        lammps_invocation: Command to invoke LAMMPS
        timesteps: Number of MD time steps
        environ: Additional environment variables
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from mofa.simulation.lammps import LAMMPSRunner
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'run-{timesteps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # Run
    lmp_runner = LAMMPSRunner(lammps_invocation, lmp_sims_root_path=str(run_dir), lammps_environ=environ)
    start_time = perf_counter()
    output = lmp_runner.run_molecular_dynamics(mof, timesteps, timesteps // 5)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help='Number of timesteps to run', default=1000, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    # Select the correct configuraion
    if args.config == "local":
        lammps_cmd = ['/home/lward/Software/lammps-2Aug2023/build/lmp', '-sf', 'omp']
        lammps_env = None
        config = Config(executors=[HighThroughputExecutor(max_workers=1, cpu_affinity='block')])
    elif args.config == "polaris":
        lammps_cmd = (
            '/lus/eagle/projects/MOFA/lward/lammps-29Aug2024/build-gpu-nompi-mixed/lmp '
            '-sf gpu -pk gpu 1'
        ).split()
        lammps_env = {'OMP_NUM_THREADS': '1'}
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
    elif args.config == "sunspot":
        lammps_cmd = ('/home/knight/lammps-git/src/lmp_aurora_gpu-lward '
                      '-pk gpu 1 -sf gpu').split()
        lammps_env = {'OMP_NUM_THREADS': '1'}
        accel_ids = [
            f"{gid}.{tid}"
            for gid in range(6)
            for tid in range(2)
        ]
        config = Config(
            retries=2,
            executors=[
                HighThroughputExecutor(
                    label="sunspot_test",
                    available_accelerators=accel_ids,  # Ensures one worker per accelerator
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers=12,
                    cores_per_worker=16,
                    provider=PBSProProvider(
                        account="CSC249ADCD08_CNDA",
                        queue="workq",
                        worker_init="""
source activate /lus/gila/projects/CSC249ADCD08_CNDA/mof-generation-at-scale/env
module reset
module use /soft/modulefiles/
module use /home/ftartagl/graphics-compute-runtime/modulefiles
module load oneapi/release/2023.12.15.001
module load intel_compute_runtime/release/775.20
module load gcc/12.2.0
module list

source activate /lus/gila/projects/CSC249ADCD08_CNDA/mof-generation-at-scale/env

cd $PBS_O_WORKDIR
pwd
which python
hostname
                        """,
                        walltime="1:10:00",
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        select_options="system=sunspot,place=scatter",
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
        with open('example-mofs.json') as fp:
            for line in fp:
                mof = MOFRecord(**json.loads(line))
                future = test_app(mof, lammps_cmd, args.timesteps, lammps_env)
                future.mof = mof
                futures.append(future)

        # Store results
        scorer = LatticeParameterChange(md_length=args.timesteps)
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                print(f'{future.mof.name} failed: {future.exception}')
                continue
            runtime, traj = future.result()

            # Get the strain
            # TODO (wardlt): Simplify how we compute strain
            traj_vasp = [write_to_string(t, 'vasp') for t in traj]
            mof = future.mof
            mof.md_trajectory['uff'] = {str(args.timesteps): traj_vasp}
            strain = scorer.score_mof(mof)

            # Store the result
            with open('runtimes.json', 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'lammps_cmd': lammps_cmd,
                    'timesteps': args.timesteps,
                    'mof': mof.name,
                    'runtime': runtime,
                    'strain': strain
                }), file=fp)
