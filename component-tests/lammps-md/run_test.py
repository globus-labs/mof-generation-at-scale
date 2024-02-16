"""Test LAMMPS by running a large number of MD simulations with different runtimes"""
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


def test_function(mof: MOFRecord, lammps_invocation: list[str], timesteps: int) -> tuple[float, list[Atoms]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        strc: MOF to use
        lammps_invocation: Command to invoke LAMMPS
        timesteps: Number of MD time steps
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
    lmp_runner = LAMMPSRunner(lammps_invocation, lmp_sims_root_path=str(run_dir))
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
        config = Config(executors=[HighThroughputExecutor(max_workers=1, cpu_affinity='block')])
    elif args.config == "polaris":
        lammps_cmd = ('/lus/eagle/projects/ExaMol/mofa/lammps-2Aug2023/src/lmp_polaris_nvhpc_kokkos '
                      '-k on g 1 -sf kk -pk kokkos neigh half neigh/qeq full newton on ').split()
        config = Config(executors=[
            HighThroughputExecutor(
                max_workers=4,
                cpu_affinity='block-reverse',
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    account='ExaMol',
                    queue='debug',
                    select_options="ngpus=4",
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""""",
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    cpus_per_node=32,
                    walltime="1:00:00",
                )
            )
        ])
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Prepare parsl
    parsl.load(config)
    test_app = PythonApp(test_function)

    # Submit each MOF
    futures = []
    with open('example-mofs.json') as fp:
        for line in fp:
            mof = MOFRecord(**json.loads(line))
            future = test_app(mof, lammps_cmd, args.timesteps)
            future.mof = mof
            futures.append(future)

    # Store results
    scorer = LatticeParameterChange()
    for future in tqdm(futures):
        runtime, traj = future.result()

        # Get the strain
        # TODO (wardlt): Simplify how we compute strain
        traj_vasp = [write_to_string(t, 'vasp') for t in traj]
        mof = future.mof
        mof.md_trajectory['uff'] = traj_vasp
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
