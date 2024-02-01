"""Utilities to go between XYZ files and SMILES strings"""
from threading import Lock
from typing import Collection

from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit import Chem
from openbabel import pybel
from openbabel import openbabel as OB

from sklearn.metrics import pairwise_distances as pdist
import pandas as pd
import numpy as np

from itertools import chain
from io import StringIO
from pathlib import Path
from pymatgen.core import periodic_table

_generate_lock = Lock()
_bond_length_path = Path(__file__).parent.parent / "assembly" / "OChemDB_bond_threshold.csv"


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
            obatom.SetImplicitHCount(nH)
    obmol.ConvertDativeBonds()
    obmol.AddHydrogens()
    # go back to OBB python API
    pbmol = pybel.Molecule(obmol)
    # convert to RDKitMol by converting SDF first
    xyz_str = pbmol.write(format='xyz', filename=None)
    return xyz_str


def rubber_banding_COO_ligands_xyz(xyz: str, exclude_atoms: list[list[int]] = []) -> str:
    xyz_df = pd.read_csv(StringIO(xyz), sep=r"\s+", skiprows=2, header=None, index_col=None, names=["el", "x", "y", "z"])
    aa = xyz_df.loc[list(chain(*exclude_atoms)), :]
    nonaa = xyz_df[xyz_df.index.isin(list(set(xyz_df.index) - set(aa.index)))]
    aaC = aa[aa["el"]=="C"]#.reset_index(drop=True)
    COO_X = nonaa[nonaa["el"].isin(["C", "N"])]#.reset_index(drop=True)
    pdmat = pdist(aaC.loc[:, ["x", "y", "z"]].values, COO_X.loc[:, ["x", "y", "z"]].values)
    min2_indices = pdmat.argsort(axis=1)[:, 0]
    i1 = 0
    j1 = min2_indices[i1]
    i2 = 1
    j2 = min2_indices[i2]
    dist1 = pdmat[i1, j1]
    dist2 = pdmat[i2, j2]
    
    C1i = COO_X.index[j1]
    C2i = COO_X.index[j2]
    aaC1i = aaC.index[i1]
    aaC2i = aaC.index[i2]
    
    C1 = xyz_df.loc[C1i, ["x", "y", "z"]].values
    C2 = xyz_df.loc[C2i, ["x", "y", "z"]].values
    aaC1 = xyz_df.loc[aaC1i, ["x", "y", "z"]].values
    aaC2 = xyz_df.loc[aaC2i, ["x", "y", "z"]].values
    CCdirction = C1 - C2
    CCdirction = CCdirction / np.linalg.norm(CCdirction)
    CC_single_bond_length = 1.54
    anc1 = C1 + (CCdirction * CC_single_bond_length)
    anc2 = C2 - (CCdirction * CC_single_bond_length)
    for COO in exclude_atoms:
        anc = -9999
        aaC = np.array([-9999, -9999, -9999])
        if aaC1i in COO:
            anc = anc1
            aaC = aaC1
        elif aaC2i in COO:
            anc = anc2
            aaC = aaC2
        disp = anc - aaC
        xyz_df.loc[COO, ["x", "y", "z"]] = xyz_df.loc[COO, ["x", "y", "z"]].values + disp
    return str(len(xyz_df)) + "\n\n" + xyz_df.to_string(header=None, index=None)


def check_interatomic_distance(xyz):
    xyz_df = pd.read_csv(StringIO(xyz), sep=r"\s+", skiprows=2, header=None, index_col=None, names=["el", "x", "y", "z"])

    df = pd.read_csv(_bond_length_path, index_col=0)
    element2bondLengthMap = dict(zip(df["element"], df["min"] - (df["min"] * 0.3)))
    unique_bond_el = list(set(list(chain(*[["-".join(sorted([x, y])) for x in xyz_df["el"].to_list()] for y in xyz_df["el"].to_list()]))))
    unique_bond_el = unique_bond_el + ["Fr-Se"]
    for x in unique_bond_el:
        if x not in element2bondLengthMap:
            element2bondLengthMap[x] = 0.
            for y in x.split("-"):
                if not isinstance(periodic_table.Element(y).atomic_radius_calculated, type(None)):
                    element2bondLengthMap[x] = element2bondLengthMap[x] + periodic_table.Element(y).atomic_radius_calculated
    distMat = pdist(xyz_df.loc[:, ["x", "y", "z"]].values, xyz_df.loc[:, ["x", "y", "z"]].values)
    distMat[distMat == 0] = np.inf
    bondLengthThresMat = np.array([[element2bondLengthMap["-".join(sorted([x, y]))] for x in xyz_df["el"].to_list()] for y in xyz_df["el"].to_list()])
    return np.all(distMat > bondLengthThresMat)
