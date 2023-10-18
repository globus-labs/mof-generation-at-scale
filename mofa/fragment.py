"""Functions pertaining to fragmenting MOF linkers to generate data for generative model training and sampling"""
from pathlib import Path

import ase

from model import MOFRecord

def fragment_mof_linkers(
        starting_model: str | Path,
        examples: list[MOFRecord],
        num_epochs: int
) -> Path:
    """Fragment linkers of MOFs

    Args:
        starting_model: Path to the starting weights of the model
        examples: Seed examples of linkers (data model TBD)
        num_epochs: Number of training epochs
    Returns:
        Path to the new model weights
    """
    raise NotImplementedError()

