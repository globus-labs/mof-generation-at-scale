"""Test method for selecting which computation to run"""

from pytest import raises, fixture

from mofa.db import create_records, mark_in_progress
from mofa.selection.dft import DFTSelector
from mofa.selection.md import MDSelector


@fixture()
def example_coll(coll, example_record):
    # Insert a MOF which has run for too long,
    #  one that's too unstable, and
    #  one that's neither
    example_record.md_trajectory['uff'] = [(10000000, 'a')]
    example_record.structure_stability['uff'] = 0.1
    example_record.name = 'a'
    create_records(coll, [example_record])

    example_record.md_trajectory['uff'] = [(1000, 'b')]
    example_record.structure_stability['uff'] = 1.
    example_record.name = 'b'
    create_records(coll, [example_record])

    example_record.structure_stability['uff'] = 0.01
    example_record.name = 'c'
    create_records(coll, [example_record])
    return coll


def test_select(example_coll, example_record):
    # Make the selector
    selector = MDSelector(
        collection=example_coll,
        max_strain=0.25,
        maximum_steps=10000,
        md_level='uff',
        new_fraction=-1,
    )
    record = selector.select_next([])
    assert record.structure_stability['uff'] == 0.01

    # Ensure it throws an error
    with raises(ValueError, match='criteria'):
        selector.max_strain = -1
        selector.select_next([])
    assert selector.select_next([example_record]) is example_record

    # Ensure that random picks work
    selector.new_fraction = 1.
    selector.max_strain = 1.
    assert selector.select_next([example_record]) is example_record


def test_dft_selector(example_coll):
    selector = DFTSelector(
        collection=example_coll,
        md_level='uff',
        max_strain=0.05
    )
    next_rec = selector.select_next()
    assert next_rec.structure_stability['uff'] == 0.01

    mark_in_progress(example_coll, next_rec, 'dft')
    assert 'dft' in example_coll.find_one({'name': next_rec.name})['in_progress']
    with raises(ValueError, match='criteria'):
        selector.select_next()
