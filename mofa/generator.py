"""Functions pertaining to training and running the generative model"""
from tempfile import TemporaryDirectory
from pathlib import Path

from mofa.utils.difflinker_sample_and_analyze import main_run
from mofa.difflinker_train import get_args, main
from ase.io import read
import yaml
import ase

from mofa.model import MOFRecord


def train_generator(
        starting_model: str | Path,
        config_path: str | Path,
        examples: Path | list[MOFRecord] = "../argonne_gnn_gitlab/DiffLinker/data/geom/datasets",
        num_epochs: int = 10,
        device: str = 'cpu'
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        starting_model: Path to the starting weights of the model
        config_path: Path to the model configuration file
        examples: Seed examples of linkers (data model TBD)
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

    # Overwrite the options provided by the Python function
    args.n_epochs = num_epochs
    args.device = device
    args.data = str(examples)
    main(args=args)


def run_generator(
        model: str | Path,
        input_path: str | Path,
        n_atoms: int | str = 8,
        n_samples: int = 1,
        n_steps: int = None,
        device: str = 'cpu'
) -> list[ase.Atoms]:
    """Produce a set of new linkers given a model

    Args:
        node: Which node to use: ['CuCu','ZnZn','ZnOZnZnZn'] to reproduce paper results
        n_atoms: Number of heavy atoms in the linker molecules to generate
        input_path: Path to MOF linker fragment containing SDF file
        model: Path to the starting weights
        n_samples: Number of samples of molecules to generate
        n_steps: Number of denoising steps; if None, this value is 1,000 by default
        device: Device on which to run model

    Returns:
        3D geometries of the generated linkers
    """

    with TemporaryDirectory(prefix='mofagen-') as tmpdir:
        # Produce a sample directory full of XYZ files
        main_run(
            input_path=input_path,
            output_dir=tmpdir,
            model=model,
            linker_size=str(n_atoms),
            n_samples=n_samples,
            n_steps=n_steps,
            anchors=None,
            device=device
        )

        # Load them from disk
        return [
            read(path)
            for path in Path(tmpdir).glob('*xyz')
        ]
