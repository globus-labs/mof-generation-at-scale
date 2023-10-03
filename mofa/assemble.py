"""Functions for assembling a MOF structure"""
from typing import Sequence

from .model import NodeDescription, LigandDescription, MOFRecord


def assemble_mof(nodes: Sequence[NodeDescription], ligands: Sequence[LigandDescription], topology: str) -> MOFRecord:
    """Generate a new MOF from the description of the nodes, ligands and toplogy

    Args:
        nodes: Descriptions of each node
        ligands: Description of the ligands
        topology: Name of the topology

    Returns:
        A new MOF record
    """
    raise NotImplementedError()
