"""Functions pertaining to training and running the generative model"""
import gzip
import json
import os
import shutil
from platform import node
from dataclasses import asdict
from tempfile import TemporaryDirectory
from typing import Iterator
from pathlib import Path

import yaml

# TODO (wardlt): Torch is imported in the subsequent MOFA modules
#  I plan to refactor those modules eventually,
#  so am putting the Intel GPU imports here until I
#  shuffle everything into a good structure
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    import oneccl_bindings_for_pytorch as torch_ccl  # noqa: F401
except ImportError:
    pass

from mofa.model import LigandDescription, LigandTemplate, MOFRecord
from mofa.utils.difflinker_sample_and_analyze import main_run
from mofa.difflinker_train import get_args, main


def train_generator(
        starting_model: str | Path | None,
        run_directory: Path,
        config_path: str | Path,
        examples: list[MOFRecord],
        num_epochs: int = 10,
        device: str = 'cpu',
        strategy: str | None = None,
        node_list: list[str] = ()
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        starting_model: Path to the starting weights of the model. ``None`` to start from scratch
        run_directory: Directory in which to run training
        config_path: Path to the model configuration file
        examples: Path to examples used to train the generator. Should be a directory which contains SDF,
        num_epochs: Number of training epochs
        device: Device to use for training
        strategy: Strategy used for distributed training
        node_list: List of nodes participating in training
    Returns:
        Path to the new model weights
    """

    # Get rank if doing distributed training
    if len(node_list) > 0:
        rank_info = get_rank(node_list)
        # TODO (wardlt): I'm not sure what's required here
        #  Is it different than https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/
        os.environ['MASTER_ADDR'] = node_list[0]
        os.environ['MASTER_PORT'] = '65143'
        strategy = 'ddp'
    else:
        rank_info = None

    # Load configuration from YML file
    #  TODO (wardlt): Move away from argparse? Could simplify making modular configuration files
    args = get_args(['--config', str(config_path)])

    # Overwrite any arguments from argparse with any from the configuration file
    #  TODO (wardlt): Move this to `get_args`?
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list) and key != 'normalize_factors':
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    args.config = args.config.name
    args.strategy = strategy

    # Save the current model into the checkpoint directory, which is where difflinker will look for starting weights
    run_directory = Path(run_directory)
    run_directory.mkdir(exist_ok=True)
    chkpt_dir = run_directory / 'chkpt'
    if starting_model is not None:
        chkpt_dir.mkdir(exist_ok=True)
        shutil.copyfile(starting_model, chkpt_dir / 'difflinker_epoch=00.ckpt')
        args.resume = 'yes'

    # Write the training data to run directory, formatted as needed by difflinker
    #  TODO (wardlt): Include example molecules from GEOM?
    with gzip.open(run_directory / 'mofa_train.json.gz', 'wt') as fp:
        for example in examples:
            print(json.dumps(asdict(example)), file=fp)
    args.data = run_directory
    args.val_data_prefix = 'mofa_train'  # TODO (wardlt): Use separate data for validation?
    args.train_data_prefix = 'mofa_train'
    args.dataset_override = 'MOFA'

    # Overwrite the options provided by the Python function
    args.n_epochs = num_epochs
    args.test_epochs = num_epochs * 2  # Turn off testing during workflow training
    args.device = device

    return main(args=args, run_directory=run_directory, rank_info=rank_info)


def run_generator(
        model: str | Path,
        templates: list[LigandTemplate],
        n_atoms: int | str = 8,
        n_samples: int = 1,
        n_steps: int = None,
        device: str = 'cpu'
) -> Iterator[LigandDescription]:
    """Produce a set of new linkers given a model

    Args:
        n_atoms: Number of heavy atoms in the linker molecules to generate
        templates: Templates of ligands to be generated
        model: Path to the starting weights
        n_samples: Number of samples of molecules to generate
        n_steps: Number of denoising steps; if None, this value is 1,000 by default
        device: Device on which to run model

    Returns:
        New ligands
    """

    with TemporaryDirectory(prefix='mofagen-') as tmpdir:
        # Produce a sample directory full of XYZ files
        yield from main_run(
            templates=templates,
            output_dir=tmpdir,
            model=model,
            linker_size=str(n_atoms),
            n_samples=n_samples,
            n_steps=n_steps,
            device=device
        )


def get_rank(node_list: list[str]) -> tuple[int, int, int]:
    """Determine rank of this process based on its node ID and local rank

    Args:
        node_list: List of nodes participating in the training
    Returns:
        - Rank of this process
        - Total ranks in training pool
        - Ranks per node
    """
    node_name = node()
    for i, name in enumerate(node_list):
        if name.startswith(node_name):
            node_rank = i
            break
    else:
        raise ValueError(f'Node ({node_name}) not found in list: {", ".join(node_list)}')

    # Get the number of ranks per node from Parsl's environment variables
    #  See: https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html
    local_rank = int(os.environ['PARSL_WORKER_RANK'])
    local_size = int(os.environ['PARSL_WORKER_COUNT'])

    return node_rank * local_size + local_rank, local_size * len(node_list), local_size
