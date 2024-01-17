"""Test XYZ<->SMILES conversions"""

from rdkit import Chem
from pytest import mark

from mofa.utils.xyz import smiles_to_xyz, xyz_to_smiles


@mark.parametrize('smiles', ['C', 'C=C', 'c1cnccc1'])
def test_reversibility(smiles):
    """Make sure we can go from SMILES to XYZ and back"""

    xyz = smiles_to_xyz(smiles)
    new_smiles = xyz_to_smiles(xyz)

    # Convert both to inchi, which is unique
    start_inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    end_inchi = Chem.MolToInchi(Chem.MolFromSmiles(new_smiles))
    assert start_inchi == end_inchi
