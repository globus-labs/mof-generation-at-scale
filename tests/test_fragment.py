from tempfile import TemporaryDirectory
from pathlib import Path

from pytest import mark
from mofa.fragment import fragment_mof_linkers

_data_path = Path(__file__).parent / 'files' / 'difflinker' / 'hMOF_CO2_info.csv.gz'


@mark.parametrize('nodes', [['CuCu']])
def test_fragmentation(nodes):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fragment_mof_linkers(_data_path, tmpdir, nodes)
        assert (tmpdir / 'fragments_all' / "_".join(nodes) / 'hMOF_frag_frag.sdf').is_file()
