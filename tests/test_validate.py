"""Test functions which validate a molecule given an XYZ file"""
from ase import Atoms
from ase.build import molecule
from pytest import fixture, raises

from mofa.assembly.validate import validate_xyz


@fixture()
def methane() -> Atoms:
    return molecule('CH4')


@fixture()
def bad_methane() -> Atoms:
    atoms = molecule('CH4')
    atoms.positions[3, 1] += 10
    return atoms


def test_valid(methane):
    smi = validate_xyz(methane)
    assert smi == 'C'


def test_disconnected(bad_methane):
    with raises(ValueError) as exc:
        validate_xyz(bad_methane)
    assert 'Disconnected' in str(exc.value)
