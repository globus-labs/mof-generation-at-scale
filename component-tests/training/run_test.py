import json
import argparse
from itertools import cycle
from pathlib import Path

import gzip
from platform import node

import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher

from mofa.model import MOFRecord


# Hard-coded defaults
_model_path = "../../models/geom-300k/geom_difflinker_epoch=997_new.ckpt"
_config_path = "../../models/geom-300k/config-tf32-a100.yaml"
_training_set = Path("mofs.json.gz")


def test_function(model_path: Path, config_path: Path, training_set: list, num_epochs: int, device: str) -> float:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        model_path: Path to the model
        config_path: Path to the configuration file
        training_set: List of MOFs to use for training
        num_epochs: Number of training epochs to run
        device: Device on which to run generation
    Returns:
        - Runtime (s)
    """
    from tempfile import TemporaryDirectory
    from mofa.generator import train_generator
    from pathlib import Path
    from time import perf_counter

    # Run
    with TemporaryDirectory() as tmp:
        start_time = perf_counter()
        train_generator(
            starting_model=model_path,
            run_directory=Path(tmp),
            config_path=config_path,
            examples=training_set,
            num_epochs=num_epochs,
            device=device,
        )
        run_time = perf_counter() - start_time

    return run_time


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Version of DiffLinker to run', default=_model_path)
    parser.add_argument('--training-size', help='Number of entries to use from training set', type=int, default=128)
    parser.add_argument('--num-epochs', help='Number of training epochs', type=int, default=16)
    parser.add_argument('--device', help='Device on which to run DiffLinker', default='cuda')
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    # Load in examples from the training set
    training_set = []
    with gzip.open('mofs.json.gz') as fp:
        for line, _ in zip(fp, range(args.training_size)):
            record = json.loads(line)
            record.pop('_id')
            training_set.append(MOFRecord(**record))
    if len(training_set) < args.training_size:
        training_set = [l for l, _ in zip(cycle(training_set), range(args.training_size))]

    # Select the correct configuraion
    if args.config == "local":
        config = Config(executors=[HighThroughputExecutor(max_workers=1, cpu_affinity='block')])
    elif args.config == "polaris":
        config = Config(retries=1, executors=[
            HighThroughputExecutor(
                max_workers=4,
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
        config = Config(
            executors=[
                HighThroughputExecutor(
                    label="sunspot_test",
                    prefetch_capacity=0,
                    max_workers=1,
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

python -c "import intel_extension_for_pytorch as ipex; print(ipex.xpu.device_count())"

cd $PBS_O_WORKDIR
pwd
which python
hostname
                        """,
                        walltime="1:10:00",
                        launcher=SimpleLauncher(),
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
    parsl.load(config)
    test_app = PythonApp(test_function)

    # Call the training function
    runtime = test_app(_model_path, _config_path, training_set, num_epochs=args.num_epochs, device=args.device).result()

    # Save the result
    with open('runtimes.json', 'a') as fp:
        print(json.dumps({
            **args.__dict__,
            'runtime': runtime,
            'host': node()
        }), file=fp)
