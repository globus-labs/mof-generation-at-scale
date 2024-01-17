"""Test XYZ<->SMILES conversions"""

from rdkit import Chem
from pytest import mark

from mofa.utils.xyz import smiles_to_xyz, xyz_to_smiles, unsaturated_xyz_to_mol


@mark.parametrize('smiles', ['C', 'C=C', 'c1cnccc1'])
def test_reversibility(smiles):
    """Make sure we can go from SMILES to XYZ and back"""

    xyz = smiles_to_xyz(smiles)
    new_smiles = xyz_to_smiles(xyz)

    # Convert both to inchi, which is unique
    start_inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    end_inchi = Chem.MolToInchi(Chem.MolFromSmiles(new_smiles))
    assert start_inchi == end_inchi


@mark.parametrize('smiles', ['C', 'C=C', 'c1cnccc1'])
def test_from_unsaturated(smiles):
    """Test whether we can infer smiles from unsaturated"""

    # Create an XYZ w/o hydrogens
    xyz = smiles_to_xyz(smiles).rstrip().split("\n")
    non_hs = [i for i in xyz[2:] if not i.startswith("H")]
    non_h_xyz = f"{len(non_hs)}\n\n" + "\n".join(non_hs)

    # Try to re-determine the XYZ structure
    mol = unsaturated_xyz_to_mol(non_h_xyz)

    # See if the InChI matches
    start_inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    end_inchi = Chem.MolToInchi(mol)
    assert start_inchi == end_inchi, f'In: {smiles} - Out: {Chem.MolToSmiles(mol)}'

    # Get the implicit valence
    new_xyz = Chem.MolToXYZBlock(mol)
    assert len(xyz) == len(new_xyz.split("\n")), 'Size of the XYZ files are different'
