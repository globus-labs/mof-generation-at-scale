"""Functions pertaining to training and running the generative model"""
from pathlib import Path

from ase.io import write
import ase

from mofa.model import MOFRecord, LigandDescription


def train_generator(
        starting_model: str | Path,
        examples: list[MOFRecord],
        num_epochs: int
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        starting_model: Path to the starting weights of the model
        examples: Seed examples of linkers (data model TBD)
        num_epochs: Number of training epochs
    Returns:
        Path to the new model weights
    """
    raise NotImplementedError()


def run_generator(
        model: str | Path,
        fragment_template: LigandDescription,
        molecule_sizes: list[int],
        num_samples: int,
        fragment_spacing: float | None = None,
) -> list[ase.Atoms]:
    """
    Args:
        model: Path to the starting weights
        fragment_template: Template to be filled with linker atoms
        molecule_sizes: Number of heavy atoms in the linker to generate
        num_samples: Number of samples of molecules to generate
        fragment_spacing: Starting distance between the fragments
    Returns:
        3D geometries of the generated linkers
    """

    # Create the template input
    blank_template = fragment_template.generate_template(spacing_distance=fragment_spacing)
    write('test.sdf', blank_template)

    # Run the generator
    raise NotImplementedError()
