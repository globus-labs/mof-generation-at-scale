"""Test functions which validate a molecule given an XYZ file"""
from io import StringIO

from ase.build import molecule
from pytest import fixture, raises

from mofa.assembly.validate import validate_xyz


def _to_xyz(atoms):
    fp = StringIO()
    atoms.write(fp, format='xyz')
    return fp.getvalue()


@fixture()
def methane() -> str:
    atoms = molecule('CH4')
    return _to_xyz(atoms)


@fixture()
def bad_methane() -> str:
    atoms = molecule('CH4')
    atoms.positions[3, 1] += 10
    return _to_xyz(atoms)


def test_valid(methane):
    smi = validate_xyz(methane)
    assert smi == 'C'


def test_disconnected(bad_methane):
    with raises(ValueError) as exc:
        validate_xyz(bad_methane)
    assert 'Disconnected' in str(exc.value)
