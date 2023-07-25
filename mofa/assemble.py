"""Functions for assembling a MOF structure"""
import ase


def assemble_mof(
        node: object,
        linker: str,
        topology: object
) -> ase.Atoms:
    """Generate a MOF structure from a recipe

    Args:
        node: Atomic structure of the nodes
        linker: SMILES string defining the linker object
        topology: Description of the network structure
    Returns:
        Description of the 3D structure of the MOF
    """
    raise NotImplementedError()
