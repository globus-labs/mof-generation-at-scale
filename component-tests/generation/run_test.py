from concurrent.futures import as_completed
from itertools import product
from platform import node
from pathlib import Path
import argparse
import json

from tqdm import tqdm
import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import MpiExecLauncher

# Hard-coded defaults
_model_path = "../../tests/files/difflinker/geom_difflinker_given_anchors.ckpt"
_templates = list(Path("../../input-files/zn-paddle-pillar/").glob("template*yml"))


def test_function(model_path: Path, n_atoms: int, template: Path, n_samples: int, device: str) -> float:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        model_path: Path to the model
        n_atoms: Size of the ligand to generate
        template: Starting template
        n_samples: Number of samples per batch
        device: Device on which to run generation
    Returns:
        - Runtime (s)
    """
    from mofa.generator import run_generator
    from mofa.model import LigandTemplate
    from time import perf_counter

    # Run
    template = LigandTemplate.from_yaml(template)
    start_time = perf_counter()
    result = list(run_generator(
        model=model_path,
        templates=[template],
        n_atoms=n_atoms,
        n_samples=n_samples,
        device=device
    ))
    run_time = perf_counter() - start_time

    return run_time


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Version of DiffLinker to run', default=_model_path)
    parser.add_argument('--template-paths', nargs='+', help='Templates to use for test seeds', default=_templates)
    parser.add_argument('--num-samples', nargs='+', type=int, help='Number of samples per batch', default=[64, 32])
    parser.add_argument('--num-atoms', nargs='+', type=int, help='Number of atoms per molecule', default=[9, 12, 15])
    parser.add_argument('--device', help='Device on which to run DiffLinker', default='cuda')
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    # Select the correct configuraion
    if args.config == "local":
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1, cpu_affinity='block')])
    elif args.config == "polaris":
        config = Config(retries=1, executors=[
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
    elif args.config.startswith("sunspot"):
        # Map processes to specific tiles or whole devices
        if args.config == "sunspot":
            accel_ids = [
               f"{gid}.{tid}"
               for gid in range(6)
               for tid in range(2)
            ]
        elif args.config == "sunspot-device":
            accel_ids = [
               f"{gid}.0,{gid}.1"
               for gid in range(6)
            ]
        else:
            raise ValueError(f'Not supported: {args.config}')

        # Ensures processes are mapped to physical cores
        workers_per_socket = len(accel_ids) // 2
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
                    label="sunspot_test",
                    available_accelerators=accel_ids,  # Ensures one worker per accelerator
                    cpu_affinity='list:' + ":".join(assigned_cores),  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers_per_node=len(accel_ids),
                    cores_per_worker=208 // len(accel_ids),
                    provider=PBSProProvider(
                        account="CSC249ADCD08_CNDA",
                        queue="workq",
                        worker_init=f"""
source activate /lus/gila/projects/CSC249ADCD08_CNDA/mof-generation-at-scale/env
module reset
module use /soft/modulefiles/
module use /home/ftartagl/graphics-compute-runtime/modulefiles
module load oneapi/release/2023.12.15.001
module load intel_compute_runtime/release/775.20
module load gcc/12.2.0
module list

{"" if len(accel_ids) == 12 else "export IPEX_TILE_AS_DEVICE=0"}
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
                        max_blocks=1, # Can increase more to have more parallel batch jobs
                        cpus_per_node=208,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Prepare parsl
    parsl.load(config)
    test_app = PythonApp(test_function)

    # Submit all combinations
    futures = []
    for template, n_atoms, n_samples in product(args.template_paths, args.num_atoms, args.num_samples):
        kwargs = {'template': str(template), 'n_atoms': n_atoms, 'n_samples': n_samples}
        future = test_app(args.model_path, n_atoms, template, n_samples, device=args.device)
        future.info = kwargs
        futures.append(future)

    # Store results
    for future in tqdm(as_completed(futures), total=len(futures)):
        runtime = future.result()

        # Store the result
        with open('runtimes.json', 'a') as fp:
            print(json.dumps({
                'host': node(),
                'model_path': str(args.model_path),
                'device': args.device,
                'runtime': runtime,
                'config': args.config,
                **future.info
            }), file=fp)
