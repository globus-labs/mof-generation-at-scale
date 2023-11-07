from pathlib import Path

from pytest import fixture


@fixture
def cif_files():
    return Path(__file__).parent / 'cif_files'
