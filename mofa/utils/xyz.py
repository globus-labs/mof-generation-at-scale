"""Utilities to go between XYZ files and SMILES strings"""
from threading import Lock
from typing import Collection

from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit import Chem
import numpy as np

from mofa.utils.src import const

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
    rdDetermineBonds.DetermineBondOrders(mol)
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


def unsaturated_xyz_to_mol(xyz: str) -> Chem.Mol:
    """Infer a molecule with reasonable bond orders from the positions of the backbone atoms

    Uses the distances between atoms to determine which atoms are bonded
    and the bond order.

    Args:
        xyz: 3D coordinates of atomic environments
        exclude_atoms: Indices of atoms on which no additional Hydrogens will be added, such as anchor groups
    Returns:
        Best guess of a saturated molecule
    """

    # First determine connectivity given 3D coordinates
    mol: Chem.Mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    conformer: Chem.Conformer = mol.GetConformer(0)
    positions = conformer.GetPositions()

    # Based on that connectivity, infer the bond order
    for bond in mol.GetBonds():
        bond: Chem.Bond = bond

        # Get the distance between atoms
        atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
        atom_1: Chem.Atom = atom_1
        distance = np.linalg.norm(
            positions[atom_1.GetIdx(), :] - positions[atom_2.GetIdx(), :]
        ) * 100  # Distance in pm, to match with the database

        # Infer if the bond order is larger than single
        # Adapted from "utils/src/molecule_builder.py
        type_1, type_2 = atom_1.GetSymbol(), atom_2.GetSymbol()
        margins = const.MARGINS_EDM
        bond_type = Chem.BondType.SINGLE

        if type_1 in const.BONDS_2 and type_2 in const.BONDS_2[type_1]:
            thr_bond2 = const.BONDS_2[type_1][type_2] + margins[1]
            if distance < thr_bond2:
                bond_type = Chem.BondType.DOUBLE
                if type_1 in const.BONDS_3 and type_2 in const.BONDS_3[type_1]:
                    thr_bond3 = const.BONDS_3[type_1][type_2] + margins[2]
                    if distance < thr_bond3:
                        bond_type = Chem.BondType.TRIPLE

        # TODO (wardlt): Only increase the bond order if it will not violate the valency rules of bond molecules
        bond.SetBondType(bond_type)

    # Add hydrogens to the molecule
    Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        # Force RDKit to place as many hydrogens on atom as possible
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(False)
    mol.UpdatePropertyCache()  # Detects the valency
    return mol


def unsaturated_xyz_to_xyz(xyz: str, exclude_atoms: Collection[int] = ()) -> str:
    """Add hydrogens to a molecule given only the backbone atoms

    Args:
        xyz: Positions of the backbone atoms
        exclude_atoms: Indices of atoms on which no additional Hydrogens will be added, such as anchor groups
    Returns:
        Best guess coordinates
    """

    # Infer the bond orders and place implicit hydrogens
    mol: Chem.Mol = unsaturated_xyz_to_mol(xyz)
    allowed_atoms = list(set(range(mol.GetNumAtoms())).difference(exclude_atoms))
    mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=allowed_atoms)
    return Chem.MolToXYZBlock(mol)
