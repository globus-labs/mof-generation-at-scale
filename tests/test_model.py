from math import isclose

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
    template = LigandTemplate.from_yaml(file_path / 'difflinker' / 'templates' / 'template_cyano.yml')
    assert template.role == 'pillar'
    for xyz in template.xyzs:
        read_from_string(xyz, 'xyz')

    # Test making a new ligand
    ligand = template.create_description(
        atom_types=['O', 'C', 'O', 'C', 'O', 'O', 'C', 'C'],
        coordinates=np.arange(8 * 3).reshape(-1, 3)
    )
    assert ligand.role == template.role
    assert ligand.anchor_atoms == [[0, 1, 2], [3, 4, 5]]
    assert ligand.dummy_element == template.dummy_element


def test_ligand_description(file_path):
    desc = LigandDescription.from_yaml(file_path / 'difflinker' / 'templates' / 'description_COO.yml')
    assert desc.role == 'pillar'

    # Test that anchor groups match up
    assert np.equal(desc.atoms.symbols[desc.anchor_atoms[0]], ['O', 'C', 'O']).all()
    assert np.equal(desc.atoms.symbols[desc.anchor_atoms[1]], ['C', 'O', 'O']).all()

    # Test the ability to replace anchors with dummy atoms
    pass
