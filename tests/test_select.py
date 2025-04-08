"""Test method for selecting which computation to run"""

from pytest import raises, fixture

from mofa.db import create_records, mark_in_progress
from mofa.selection.dft import DFTSelector
from mofa.selection.md import MDSelector


@fixture()
def example_coll(coll, example_record):
    # Insert a MOF which
    #  has not run at all
    #  has run for too long,
    #  one that's too unstable, and
    #  one that's neither
    example_record.name = '0'
    create_records(coll, [example_record])

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


def test_empty_db(coll):
    selector = MDSelector(collection=coll)
    assert selector.count_available() == 0


def test_md_select(example_coll, example_record):
    # Make the selector
    selector = MDSelector(
        collection=example_coll,
        max_strain=0.25,
        maximum_steps=10000,
        md_level='uff',
        new_fraction=-1,
    )
    # Ensure that it counts both the "unran" and low strain not finished
    assert selector.count_available() == 2, [x['name'] for x in example_coll.aggregate(selector.match_stages)]

    # Remove the unran object
    result = example_coll.delete_one({'structure_stability.uff': {'$exists': False}})
    assert result.deleted_count == 1

    # Make sure it pulls an available structure
    record = selector.select_next([])
    assert record.structure_stability['uff'] == 0.01
    assert record.name == 'c'

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
    # None available at first
    assert selector.count_available() == 0

    # Mark all as relaxed
    example_coll.update_many({}, {'$set': {'times.relaxed': 0.}})
    assert selector.count_available() == 2

    # Drop the unran one
    example_coll.delete_one({'structure_stability.uff': {'$exists': False}})
    assert selector.count_available() == 1

    # Mark all as relaxed
    next_rec = selector.select_next()
    assert next_rec.structure_stability['uff'] == 0.01

    mark_in_progress(example_coll, next_rec, 'dft')
    assert 'dft' in example_coll.find_one({'name': next_rec.name})['in_progress']
    with raises(ValueError, match='criteria'):
        selector.select_next()
