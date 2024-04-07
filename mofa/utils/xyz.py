"""Utilities to go between XYZ files and SMILES strings"""
from threading import Lock
from typing import Collection

from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit import Chem
from openbabel import pybel
from openbabel import openbabel as OB

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


def unsaturated_xyz_to_xyz(xyz: str, exclude_atoms: Collection[int] = ()) -> str:
    """Add hydrogens to a molecule given only the backbone atoms

    Args:
        xyz: Positions of the backbone atoms
        exclude_atoms: Indices of atoms on which no additional Hydrogens will be added, such as anchor groups
    Returns:
        Best guess coordinates
    """

    pbmol = pybel.readstring("xyz", xyz)
    # some OBB C++ API black magic
    obmol = pbmol.OBMol
    obmol.SetTotalCharge(0)
    obmol.SetHydrogensAdded(False)
    for x in range(0, obmol.NumAtoms()):
        if x not in exclude_atoms:  # excluding the archor atoms such that no H is added to the -COO, -C#N, etc.
            obatom = obmol.GetAtom(x+1)
            obatom.SetFormalCharge(0)
            obatomicnum = obatom.GetAtomicNum()
            currBO = obatom.GetTotalValence()
            nH = OB.GetTypicalValence(obatomicnum, currBO, 0) - currBO
            ndeg = obatom.GetExplicitDegree()
            # to get some extra unsaturation for rigidity
            if obatomicnum == 6 and ndeg + nH == 4 and nH >= 1:
                nH = nH - 1
            elif obatomicnum == 7 and ndeg + nH == 3 and nH >= 1:
                nH = nH - 1
            obatom.SetImplicitHCount(nH)
    obmol.ConvertDativeBonds()
    obmol.AddHydrogens()
    # go back to OBB python API
    pbmol = pybel.Molecule(obmol)
    # convert to RDKitMol by converting SDF first
    xyz_str = pbmol.write(format='xyz', filename=None)
    return xyz_str
