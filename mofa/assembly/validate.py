"""Validate and standardize a generated molecule"""

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def validate_xyz(xyz: str) -> str:
    """Generate the SMILES string from an XYZ file
    and determine whether it has the proper bonds

    Args:
        xyz: XYZ of a molecule produced by a generator
    Returns:
        SMILES string of a validated molecule
    """

    # Parse the XYZ and detect bonds
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    rdDetermineBonds.DetermineBonds(mol)

    # Remove the Hs for parsimony
    mol_no_h = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol_no_h)
