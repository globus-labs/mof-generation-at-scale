"""Test methods for getting finetuning sets"""
import numpy as np

from mofa.db import create_records
from mofa.finetune.difflinker import DiffLinkerCurriculum


def test_difflinker(coll, example_record):
    curr = DiffLinkerCurriculum(collection=coll, max_size=8, min_gas_counts=4)

    # Make a series of records with increasingly-larger strains
    for strain in np.linspace(0., 0.5, 32):
        example_record.structure_stability['mace'] = strain
        create_records(coll, [example_record])

    records = curr.get_training_set()
    assert len(records) == 8
    assert records[0].structure_stability['mace'] < curr.max_strain

    # Add some records with a gas capacity
    for gc in np.linspace(1, 2, 16):
        example_record.gas_storage['CO2'] = gc
        example_record.structure_stability['mace'] = curr.max_strain * 0.9
        create_records(coll, [example_record])

    assert coll.estimated_document_count() == 48
    records = curr.get_training_set()
    assert len(records) == 8
    assert all(r.gas_storage['CO2'] > 1.5 for r in records)

    curr.max_size = 4
    records = curr.get_training_set()
    assert len(records) == 4
    assert all(r.gas_storage['CO2'] > 1.5 for r in records)
