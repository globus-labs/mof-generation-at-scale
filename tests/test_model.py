from math import isclose

from ase.io import read
from rdkit import Chem
from pytest import mark
import numpy as np
import pandas as pd
import io
import itertools

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

    # Test initializing the positions
    symbols, positions, connect_ids = template.prepare_inputs()
    assert symbols == ['C', 'O', 'O'] * 2
    assert positions.shape == (6, 3)
    assert np.equal(connect_ids, [0, 3]).all()

    # Test making a new ligand
    ligand = template.create_description(
        atom_types=['C', 'O', 'O', 'C', 'O', 'O', 'C', 'C'],
        coordinates=np.arange(8 * 3).reshape(-1, 3)
    )
    assert ligand.anchor_type == template.anchor_type
    assert ligand.anchor_atoms == [[0, 1, 2], [3, 4, 5]]
    assert ligand.dummy_element == template.dummy_element


@mark.parametrize('anchor_type', ['COO'])
def test_ligand_description_H_inference(file_path, anchor_type):
    # Load the template and the new coordinates
    template = LigandTemplate.from_yaml(file_path / 'difflinker' / 'templates' / 'template_COO.yml')
    example_xyz = read(file_path / 'difflinker' / 'templates' / 'difflinker-coo-example.xyz')

    # Instantiate the template
    desc = template.create_description(example_xyz.get_chemical_symbols(), example_xyz.positions)
    needed_orig_xyz_str = "\n".join(desc.xyz.split("\n")[2:])
    needed_new_xyz_str = "\n".join(desc.xyz.split("\n")[2:])
    xyz_diff = needed_new_xyz_str.replace(needed_orig_xyz_str, "")
    df = pd.read_csv(io.StringIO(xyz_diff), sep=r"\s+", header=None, index_col=None, names=["element", "x", "y", "z"])

    # test if all added atoms are Hs
    all_added_atoms_are_Hs = np.all(df["element"].to_numpy() == "H")

    # test if none of the Hs are added to the anchor atoms
    # rdmol = Chem.rdmolfiles.MolFromMolBlock(desc.sdf)
    rdmol = Chem.rdmolfiles.MolFromXYZBlock(desc.xyz)
    H_is_detected_on_an_anchor = False
    for x in list(itertools.chain(*desc.anchor_atoms)):
        rdatom = rdmol.GetAtomWithIdx(x)
        nbrs = [x.GetSymbol() for x in list(rdatom.GetNeighbors())]
        if "H" in nbrs:
            H_is_detected_on_an_anchor = True
            break
    assert (not H_is_detected_on_an_anchor) and all_added_atoms_are_Hs


@mark.parametrize('anchor_type', ['COO', 'cyano'])
def test_ligand_description(file_path, anchor_type):
    desc = LigandDescription.from_yaml(file_path / 'difflinker' / 'templates' / f'description_{anchor_type}.yml')

    # Test the ability to replace anchors with dummy atoms
    with_dummies = desc.replace_with_dummy_atoms()
    assert with_dummies.symbols.count(desc.dummy_element) == 2
    size_change = 0
    if anchor_type == 'COO':
        size_change = -4  # Remove 4 oxygen
    elif anchor_type == 'cyano':
        size_change = 2  # Add two dummy atoms
    assert len(with_dummies) == len(desc.atoms) + size_change


@mark.parametrize('anchor_type', ['cyano'])
def test_ligand_description_swap(file_path, anchor_type):
    desc = LigandDescription.from_yaml(file_path / 'difflinker' / 'templates' / f'description_{anchor_type}.yml')
    new_desc = desc.swap_cyano_with_COO()
    assert new_desc.anchor_type == "COO" and new_desc.dummy_element == "At"
