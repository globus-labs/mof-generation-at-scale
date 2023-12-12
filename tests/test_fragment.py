from pathlib import Path

from pytest import mark

from mofa.fragment import fragment_mof_linkers

_data_path = Path(__file__).parent / 'files' / 'difflinker' / 'hMOF_CO2_info.csv.gz'


@mark.parametrize('nodes', [['CuCu'], ['ZnZn']])
def test_fragmentation(nodes):
    out_dir = _data_path.parent / 'datasets'
    fragment_mof_linkers(_data_path, out_dir, nodes)
    assert (out_dir / 'fragments_all' / "_".join(nodes) / 'hMOF_frag_frag.sdf').is_file()
