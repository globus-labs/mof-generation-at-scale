from pytest import mark
from mofa.fragment import fragment_mof_linkers


@mark.parametrize('nodes', [['CuCu'], ['CuCu', 'ZnZn']])
def test_fragmentation(nodes):
    # fragmentation(nodes)
    # process_fragments(nodes)
    fragment_mof_linkers(nodes)
