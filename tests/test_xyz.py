"""Test XYZ<->SMILES conversions"""
import pytest
from rdkit import Chem
from pytest import mark

from mofa.utils.xyz import smiles_to_xyz, xyz_to_smiles, unsaturated_xyz_to_mol, unsaturated_xyz_to_xyz


@mark.parametrize('smiles', ['C', 'C=C', 'c1cnccc1'])
def test_reversibility(smiles):
    """Make sure we can go from SMILES to XYZ and back"""

    xyz = smiles_to_xyz(smiles)
    new_smiles = xyz_to_smiles(xyz)

    # Convert both to inchi, which is unique
    start_inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    end_inchi = Chem.MolToInchi(Chem.MolFromSmiles(new_smiles))
    assert start_inchi == end_inchi


@mark.parametrize('smiles', ['C', 'C=C', pytest.param('c1cnccc1', marks=mark.xfail(reason='aromatic'))])
def test_from_unsaturated(smiles):
    """Test whether we can infer smiles from unsaturated"""

    # Create an XYZ w/o hydrogens
    xyz = smiles_to_xyz(smiles).rstrip().split("\n")
    non_hs = [i for i in xyz[2:] if not i.startswith("H")]
    non_h_xyz = f"{len(non_hs)}\n\n" + "\n".join(non_hs)

    # Try to re-determine the XYZ structure
    mol = unsaturated_xyz_to_mol(non_h_xyz)
    end_smiles = Chem.MolToSmiles(mol)

    # See if the InChI matches
    start_inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    end_inchi = Chem.MolToInchi(mol)
    assert start_inchi == end_inchi, f'In: {smiles} - Out: {end_smiles}'

    # Run forward through generating a new XYZ
    new_xyz = unsaturated_xyz_to_xyz(non_h_xyz)
    assert new_xyz.split()[0] == xyz[0].strip()  # Should start with the same number of atoms


# TODO (wardlt): Test the unsatured with a larger linker that includes anchor groups
