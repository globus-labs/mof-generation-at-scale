"""Run MD simulations with different FFs or numbers of timesteps"""
from concurrent.futures import as_completed
from pathlib import Path
from platform import node
import argparse
import json
import gzip

import pandas as pd
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
from mofa.simulation.interfaces import MDInterface
from mofa.simulation.lammps import LAMMPSRunner
from mofa.simulation.mace import MACERunner
from mofa.utils.conversions import write_to_string


def test_function(
        mof: MOFRecord,
        timesteps: int,
        runner: MDInterface,
) -> tuple[float, list[tuple[int, Atoms]]]:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        strc: MOF to use
        ff: Which forcefield to use
        timesteps: Number of MD time steps
        lammps_invocation: Command to invoke LAMMPS
        environ: Additional environment variables
        device: Device to use for MACE execution
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from time import perf_counter

    # Run
    start_time = perf_counter()
    output = runner.run_molecular_dynamics(mof, timesteps, timesteps // 5)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help='Number of timesteps to run', default=1000, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    parser.add_argument('--num-to-run', help='How many from the subset to run', default=16, type=int)
    parser.add_argument('--ff', help='Which forcefield to use', default='uff')
    parser.add_argument('--continue-runs', help='Continue previously-run trajectories', action='store_true')
    args = parser.parse_args()

    # Select the correct configuraion
    if args.config == "local":
        lammps_cmd = "/home/lward/Software/lammps-mace/build-mace/lmp -k on g 1 -sf kk".split()
        lammps_env = {}
        device = 'cuda'
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1)])
    elif args.config == "polaris":
        lammps_cmd = ('/lus/eagle/projects/ExaMol/mofa/lammps-2Aug2023/build-kokkos-nompi/lmp '
                      '-k on g 1 -sf kk').split()
        lammps_env = {'OMP_NUM_THREADS': '1'}
        config = Config(retries=4, executors=[
            HighThroughputExecutor(
                max_workers_per_node=4,
                cpu_affinity='block-reverse',
                available_accelerators=4,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    account='ExaMol',
                    queue='debug',
                    select_options="ngpus=4",
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""
module load kokkos
module load nvhpc/23.3
module list
source activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris

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
        lammps_cmd = (
            "/lus/flare/projects/MOFA/lward/lammps-kokkos/src/lmp_macesunspotkokkos "
            "-k on g 1 -sf kk"
        ).split()
        lammps_env = {}
        device = 'xpu'
        config = Config(
            retries=1,
            executors=[
                HighThroughputExecutor(
                    label="sunspot_test",
                    available_accelerators=12,  # Ensures one worker per accelerator
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers_per_node=12,
                    provider=PBSProProvider(
                        account="MOFA",
                        queue="prod",
                        worker_init="""
# General environment variables
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Needed for LAMMPS
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $PBS_O_WORKDIR
pwd
which python
hostname
                        """,
                        walltime="6:00:00",
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),
                        scheduler_options="#PBS -l filesystems=home:flare",
                        nodes_per_block=32,
                        min_blocks=0,
                        max_blocks=1,
                        cpus_per_node=208,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Prepare the runner
    run_dir = Path(f'{args.ff}-run-{args.timesteps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    if args.ff == 'uff':
        runner = LAMMPSRunner(lammps_command=lammps_cmd, lmp_sims_root_path=str(run_dir), lammps_environ=lammps_env)
    elif args.ff == 'mace':
        model_path = Path('../../input-files/mace/mace-mp0_medium-lammps.pt').absolute()
        runner = MACERunner(lammps_cmd=lammps_cmd,
                            model_path=model_path,
                            run_dir=run_dir,
                            device=device)
    else:
        raise ValueError(f'No such forcefield: {args.ff}')

    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(test_function)


        # Gather MOFs from an example set
        example_set = pd.read_csv('raw-data/ZnNCO_hMOF_cat0_valid_ads_angle_clean.csv').sample(args.num_to_run, random_state=1)
        example_set['name'] = example_set['cifname'].apply(lambda x: x[:-4])
        with gzip.open('data/hmof.json.gz', 'rb') as fp:
            structures = dict((x['name'], x) for x in map(json.loads, fp) if x['identifiers']['name'] in set(example_set['name'].tolist()))
        print(f'Found {len(structures)} to evaluate')

        # Load previous results
        out_strains = Path(f'{args.ff}-strains.jsonl')
        if out_strains.is_file():
            prev_run = pd.read_json(out_strains, lines=True)[['mof', 'timesteps', 'structure']]
            print(f'Found {len(prev_run)} previous runs with forcfield {args.ff}')

            latest_run = prev_run.sort_values('timesteps', ascending=True).drop_duplicates('mof', keep='last')
            latest_run = dict((n, (t, s)) for n, t, s in latest_run.values)

            all_runs = prev_run.groupby('mof')['timesteps'].apply(set).to_dict()
        else:
            latest_run = {}
            all_runs = {}

        # Submit each MOF
        futures = []
        for name, info in structures.items():
            mof = MOFRecord(**info)
            # Add the latest timestep to the MOF record
            num_ran = args.timesteps
            if mof.name in latest_run:
                timesteps, strc = latest_run[mof.name]
                if args.timesteps in all_runs[mof.name]:
                    continue  # We're done

                if args.continue_runs:
                    mof.md_trajectory[runner.traj_name] = [(timesteps, strc)]
                    num_ran -= timesteps
            elif args.continue_runs:
                continue  # Skip if not already run
            future = test_app(mof, args.timesteps, runner)
            future.mof = mof
            future.num_ran = num_ran
            futures.append(future)

        # Store results
        scorer = LatticeParameterChange(md_level=runner.traj_name)
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                print(f'{future.mof.name} failed: {future.exception()}')
                continue
            runtime, traj = future.result()

            # Get the strain
            # TODO (wardlt): Simplify how we compute strain
            traj_vasp = [(s, write_to_string(t, 'vasp')) for (s, t) in traj]
            mof = future.mof
            mof.md_trajectory[runner.traj_name] = traj_vasp
            strain = scorer.score_mof(mof)

            # Store the result
            with open(out_strains, 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'lammps_cmd': lammps_cmd,
                    'timesteps': args.timesteps,
                    'mof': mof.name,
                    'runtime': runtime,
                    'steps_ran': future.num_ran,
                    'strain': strain,
                    'structure': traj_vasp[-1][-1]
                }), file=fp)

