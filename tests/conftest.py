
from pytest import fixture
from pathlib import Path
from ase.io.cif import read_cif
import ase

_files_path = Path(__file__).parent / 'files'


@fixture()
def example_cif() -> Path:
    return _files_path / 'check.cif'


@fixture()
def example_mof(example_cif) -> ase.Atoms:
    with open(example_cif) as fp:
        return next(read_cif(fp, index=slice(None)))
