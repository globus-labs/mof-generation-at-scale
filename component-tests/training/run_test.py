import json
import argparse
from itertools import cycle
from pathlib import Path
from concurrent.futures import as_completed

import gzip
from platform import node

import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import SimpleLauncher, MpiExecLauncher
from parsl.executors.high_throughput.mpi_resource_management import get_nodes_in_batchjob, Scheduler

from mofa.model import MOFRecord


# Hard-coded defaults
_model_path = "../../models/geom-300k/geom_difflinker_epoch=997_new.ckpt"
_config_path = "../../models/geom-300k/config-tf32-a100.yaml"
_training_set = Path("mofs.json.gz")


def test_function(model_path: Path, config_path: Path, training_set: list, num_epochs: int, device: str, node_list: list[str] | None) -> float:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        model_path: Path to the model
        config_path: Path to the configuration file
        training_set: List of MOFs to use for training
        num_epochs: Number of training epochs to run
        device: Device on which to run generation
        node_list: List of nodes over which to run training
    Returns:
        - Runtime (s)
    """
    from tempfile import TemporaryDirectory
    from mofa.generator import train_generator
    from pathlib import Path
    from time import perf_counter
    import os

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
            node_list=node_list,
            strategy='ddp' if node_list is not None else None
        )
        run_time = perf_counter() - start_time

    return run_time


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Version of DiffLinker to run', default=_model_path)
    parser.add_argument('--training-size', help='Number of entries to use from training set', type=int, default=128)
    parser.add_argument('--num-epochs', help='Number of training epochs', type=int, default=16)
    parser.add_argument('--num-nodes', help='Number of nodes', type=int, default=1)
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
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1, cpu_affinity='block')])
        parallel = False
        ranks_per_node = 1
        hosts = None
    elif args.config == "polaris":
        ranks_per_node = 4
        # Connect to nodes using their 
        hosts = list(get_nodes_in_batchjob(Scheduler.PBS))
        config = Config(executors=[
            HighThroughputExecutor(
                max_workers_per_node=ranks_per_node,
                cpu_affinity='block-reverse'
            )
        ])
    elif args.config.startswith("aurora"):
        ranks_per_node = 12
        hosts = list(get_nodes_in_batchjob(Scheduler.PBS))
        config = Config(
            executors=[
                HighThroughputExecutor(
                    label="aurora_test",
                    prefetch_capacity=0,
                    max_workers_per_node=12,
                    available_accelerators=12,
                    cpu_affinity='block'
                ),
            ]
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(test_function)

        # Call the training function
        futures = []
        for rank in range(ranks_per_node * args.num_nodes):
            futures.append(test_app(_model_path, _config_path, training_set, 
                                    num_epochs=args.num_epochs, 
                                    device=args.device,
                                    node_list=hosts))

        # Collect
        for future in as_completed(futures):
            runtime = future.result()

        # Save the result
        with open('runtimes.json', 'a') as fp:
            print(json.dumps({
                **args.__dict__,
                'runtime': runtime,
                'host': node(),
                'hosts': hosts
            }), file=fp)
