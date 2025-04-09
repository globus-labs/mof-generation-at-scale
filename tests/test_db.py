from mofa.db import create_records, get_records, update_records, count_records, get_all_records
from mofa.model import MOFRecord


def test_initialize(coll):
    assert count_records(coll) == 0


def test_insert_and_find(coll, example_record):
    create_records(coll, [example_record])
    assert count_records(coll) == 1

    copies = get_records(coll, [example_record.name])
    assert len(copies) == 1
    assert copies[0].name == example_record.name
    assert (copies[0].times['created'] - example_record.times['created']).total_seconds() < 1e-6


def test_update(coll, example_record):
    create_records(coll, [example_record])

    example_record.gas_storage['co2'] = (1., 1.)
    update_records(coll, [example_record])

    copies = get_records(coll, [example_record.name])
    assert len(copies) == 1
    assert copies[0].gas_storage['co2'] == [1., 1.]


def test_get_all(coll):
    create_records(coll, [MOFRecord(name=str(x)) for x in range(5)])
    assert len(list(get_all_records(coll))) == 5
