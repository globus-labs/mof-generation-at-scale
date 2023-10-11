"""Functions pertaining to training and running the generative model"""
from pathlib import Path

import ase

from mofa.model import MOFRecord
from mofa import 


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
        molecule_sizes: list[int],
        num_samples: int
) -> list[ase.Atoms]:
    """
    Args:
        model: Path to the starting weights
        molecule_sizes: Number of heavy atoms in the linker molecules to generate
        num_samples: Number of samples of molecules to generate
    Returns:
        3D geometries of the generated linkers
    """
    raise NotImplementedError()
