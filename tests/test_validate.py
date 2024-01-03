"""Test functions which validate a molecule given an XYZ file"""
from io import StringIO

from ase.build import molecule
from pytest import fixture

from mofa.assembly.validate import validate_xyz


@fixture()
def methane() -> str:
    atoms = molecule('CH4')
    fp = StringIO()
    atoms.write(fp, format='xyz')
    return fp.getvalue()


def test_validate(methane):
    smi = validate_xyz(methane)
    assert smi == 'C'
