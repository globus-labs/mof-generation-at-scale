"""Test method for selecting which computation to run"""

from pytest import raises

from mofa.db import create_records
from mofa.selection.md import MDSelector


def test_select(coll, example_record):
    # Insert a MOF which has run for too long,
    #  one that's too unstable, and
    #  one that's neither
    example_record.md_trajectory['uff'] = [(10000000, 'a')]
    example_record.structure_stability['uff'] = 0.1
    create_records(coll, [example_record])

    example_record.md_trajectory['uff'] = [(1000, 'b')]
    example_record.structure_stability['uff'] = 1.
    create_records(coll, [example_record])

    example_record.structure_stability['uff'] = 0.01
    create_records(coll, [example_record])

    # Make the selector
    selector = MDSelector(
        collection=coll,
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
