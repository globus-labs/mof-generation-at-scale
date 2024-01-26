from math import isclose

from pytest import mark
import numpy as np

from mofa.model import MOFRecord, LigandTemplate, LigandDescription
from mofa.utils.conversions import read_from_string


def test_create(example_cif):
    mof = MOFRecord.from_file(example_cif, identifiers={'local': 'test'})
    assert mof.identifiers['local'] == 'test'
    assert isclose(mof.atoms.cell.lengths()[0], 39.87968858)


def test_name(example_cif):
    # Same CIF, same name
    mof_1 = MOFRecord.from_file(example_cif)
    mof_2 = MOFRecord.from_file(example_cif)
    assert mof_2.name == mof_1.name

    # No structure, random name
    mof_3 = MOFRecord()
    mof_4 = MOFRecord()
    assert mof_3.name != mof_4.name


def test_ligand_model(file_path):
    template = LigandTemplate.from_yaml(file_path / 'difflinker' / 'templates' / 'template_COO.yml')
    assert template.anchor_type == 'COO'
    for xyz in template.xyzs:
        read_from_string(xyz, 'xyz')

    # Test making a new ligand
    ligand = template.create_description(
        atom_types=['O', 'C', 'O', 'C', 'O', 'O', 'C', 'C'],
        coordinates=np.arange(8 * 3).reshape(-1, 3)
    )
    assert ligand.anchor_type == template.anchor_type
    assert ligand.anchor_atoms == [[0, 1, 2], [3, 4, 5]]
    assert ligand.dummy_element == template.dummy_element

@mark.parametrize('anchor_type', ['COO', 'cyano'])
def test_ligand_description_H_inference(file_path, anchor_type):
    desc = LigandDescription.from_yaml(file_path / 'difflinker' / 'templates' / f'description_{anchor_type}.yml')
    orig_xyz_str = desc.xyz
    desc.infer_H_and_bond_safe()
    new_xyz_str = desc.xyz

    # Test if the Hs are added and no heavy atom information has been modified
    with_dummies = desc.replace_with_dummy_atoms()
    assert with_dummies.symbols.count(desc.dummy_element) == 2

@mark.parametrize('anchor_type', ['COO', 'cyano'])
def test_ligand_description(file_path, anchor_type):
    desc = LigandDescription.from_yaml(file_path / 'difflinker' / 'templates' / f'description_{anchor_type}.yml')

    # Test the ability to replace anchors with dummy atoms
    with_dummies = desc.replace_with_dummy_atoms()
    assert with_dummies.symbols.count(desc.dummy_element) == 2
