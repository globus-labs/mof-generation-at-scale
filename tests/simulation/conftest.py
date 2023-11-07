from pathlib import Path

from pytest import fixture


@fixture
def cif_dir():
    return Path(__file__).parent / 'cif_files'
