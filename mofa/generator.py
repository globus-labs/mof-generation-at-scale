"""Functions pertaining to training and running the generative model"""
import gzip
import json
from dataclasses import asdict
from tempfile import TemporaryDirectory
from typing import Iterator
from pathlib import Path

from mofa.model import LigandDescription, LigandTemplate, MOFRecord
from mofa.utils.difflinker_sample_and_analyze import main_run
from mofa.difflinker_train import get_args, main
import yaml


def train_generator(
        starting_model: str | Path | None,
        run_directory: Path,
        config_path: str | Path,
        examples: list[MOFRecord],
        num_epochs: int = 10,
        device: str = 'cpu',
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        starting_model: Path to the starting weights of the model
        run_directory: Directory in which to run training
        config_path: Path to the model configuration file
        examples: Path to examples used to train the generator. Should be a directory which contains SDF,
        num_epochs: Number of training epochs
        device: Device to use for training
    Returns:
        Path to the new model weights
    """

    # Load configuration from YML file
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
    args.device = device

    return main(args=args, run_directory=run_directory)


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
