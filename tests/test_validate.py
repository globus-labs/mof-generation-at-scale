"""Test functions which validate a molecule given an XYZ file"""
from ase.build import molecule
from pytest import fixture

from mofa.assembly.validate import process_ligands
from mofa.model import LigandDescription
from mofa.utils.conversions import write_to_string


@fixture()
def methane() -> LigandDescription:
    return LigandDescription(xyz=write_to_string(molecule('CH4'), 'xyz'), prompt_atoms=[[0]])


@fixture()
def bad_ethane() -> LigandDescription:  # don't use broken H to test disconnectivity because H are removed by default in RDkit
    atoms = molecule('C2H6')
    atoms.positions[0, 1] += 10
    return LigandDescription(xyz=write_to_string(atoms, 'xyz'), prompt_atoms=[[0]])


def test_valid(methane):
    valid_ligands, records = process_ligands([methane])
    assert len(valid_ligands) == 1
    assert records[0]['smiles'] == 'C'


def test_disconnected(bad_ethane):
    valid_ligands, records = process_ligands([bad_methane])
    assert valid_ligands == []
    assert not records[0]['valid']
