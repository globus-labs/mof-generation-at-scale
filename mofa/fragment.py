"""Functions pertaining to fragmenting MOF linkers to generate data for generative model training and sampling"""
from pathlib import Path

import ase

from mofa.model import MOFRecord

from mofa.difflinker_fragmentation import fragmentation
from mofa.difflinker_process_fragmentation import process_fragments
from typing import *

def fragment_mof_linkers(
        nodes: List[str] = ["CuCu"]
        # starting_model: str | Path,
        # examples: list[MOFRecord],
        # num_epochs: int
) -> Path:
    """Fragment linkers of MOFs

    Args:
        starting_model: Path to the starting weights of the model
        examples: Seed examples of linkers (data model TBD)
        num_epochs: Number of training epochs
    Returns:
        Path to the new model weights
    """
    # raise NotImplementedError()
    fragmentation(nodes)
    process_fragments(nodes)
        
