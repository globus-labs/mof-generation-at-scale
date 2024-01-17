"""Utilities to go between XYZ files and SMILES strings"""
from threading import Lock

from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit import Chem


_generate_lock = Lock()


def xyz_to_mol(xyz: str) -> Chem.Mol:
    """Generate a RDKit Mol object with bonds from an XYZ string

    Args:
        xyz: XYZ string to parse
    Returns:
        RDKit mol object
    """

    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    rdDetermineBonds.DetermineBonds(mol)
    return mol


def xyz_to_smiles(xyz: str) -> str:
    """Generate a SMILES string from an XYZ string

    Args:
        xyz: XYZ string to parse
    Returns:
        SMILES string
    """

    mol = xyz_to_mol(xyz)
    return Chem.MolToSmiles(mol)


def smiles_to_xyz(smiles: str) -> str:
    """Generate an XYZ-format structure from a SMILES string

    Uses RDKit's 3D coordinate generation

    Args:
        smiles: SMILES string from which to generate molecule
    Returns:
        XYZ-format geometry
    """

    # From: https://github.com/exalearn/ExaMol/blob/main/examol/simulate/initialize.py
    with _generate_lock:
        # Generate 3D coordinates for the molecule
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)

        # Save the conformer to an XYZ file
        xyz = Chem.MolToXYZBlock(mol)
        return xyz
