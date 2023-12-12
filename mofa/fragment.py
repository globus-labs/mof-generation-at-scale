"""Functions pertaining to fragmenting MOF linkers to generate data for generative model training and sampling"""
from pathlib import Path

from mofa.difflinker_fragmentation import fragmentation
from mofa.difflinker_process_fragmentation import process_fragments
from typing import Sequence


def fragment_mof_linkers(
        data_path: Path,
        output_dir: Path,
        nodes: Sequence[str] = ("ZnZn",)
):
    """Fragment linkers of MOFs and prepare them in a directory in the format needed by Diff linker

    Args:
        data_path: Path to the MOF data to be fragmented as a CSV file
        output_dir: Directory in which to write the training data
        nodes: Only fragment MOFs based on this node
    """
    fragmentation(data_path, output_dir, nodes)
    process_fragments(output_dir, nodes)
