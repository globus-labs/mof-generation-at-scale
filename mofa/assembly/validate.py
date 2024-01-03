"""Validate and standardize a generated molecule"""
from io import StringIO

from ase import Atoms

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def validate_xyz(atoms: Atoms) -> str:
    """Generate the SMILES string from n molecule structure if it passes some quality checks

    Quality checks:
        1. Molecule is only one fragment

    Args:
        atoms: Atoms object produced by the generator
    Returns:
        SMILES string of a validated molecule
    Raises:
        ValueError if molecule fails validation checks
    """

    # Write the atoms to an XYZ
    fp = StringIO()
    atoms.write(fp, format='xyz')
    xyz = fp.getvalue()

    # Parse the XYZ and detect bonds
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    rdDetermineBonds.DetermineBonds(mol)

    # Make sure it parses as one molecule
    if len(Chem.GetMolFrags(mol)) > 1:
        raise ValueError('Disconnected molecule')

    # Remove the Hs for parsimony
    mol_no_h = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol_no_h)
