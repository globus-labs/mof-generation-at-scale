"""Utilities for writing data to disk using MongoDB"""

from dataclasses import asdict
from typing import Iterator

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

from mofa.model import MOFRecord


def _row_to_record(row: dict) -> MOFRecord:
    """Convert a Mongo document to a Sequence data record"""
    row.pop("_id")
    return MOFRecord(**row)


def initialize_database(client: MongoClient) -> Collection:
    """Create a collection in which to store sequence information"""

    collection = client.get_database('mofa').get_collection('mofs')

    # Create the indices needed for different operations
    collection.create_index([
        ("name", ASCENDING),
    ])  # Retrieving specific records
    for k in ['structure_stability.uff', 'gas_storage.CO2']:
        collection.create_index([
            (k, ASCENDING),
        ])  # Queries for initial training set (is a prefix index of the following, but I'm being extra sure it gets made
    collection.create_index([
        ('structure_stability.uff', ASCENDING),
        ('gas_storage.CO2', ASCENDING)
    ])  # Queries for training sets
    return collection


def get_records(coll: Collection, name: list[str]) -> list[MOFRecord]:
    """Get the records associated with a list names

    Args:
        coll: Collection holding our MOF data
        name: List of names
    Returns:
        The MOF records for each name. May not be in same order as ``name``
    """

    output = []
    for record in coll.find({'name': {'$in': name}}):
        output.append(_row_to_record(record))
    return output


def get_all_records(coll: Collection) -> Iterator[MOFRecord]:
    """Iterate over all records in the database

    Args:
        coll: Collection holding our MOF data
    """
    for record in coll.find({}):
        yield _row_to_record(record)


def create_records(coll: Collection, records: list[MOFRecord]):
    """Insert new records into the database

    Args:
        coll: Collection holding our MOF data
        records: Records to be inserted
    """

    coll.insert_many(asdict(r) for r in records)


def update_records(coll: Collection, records: list[MOFRecord]):
    """Update the scores and PDB associated with a sequence

    Args:
        coll: Collection holding our MOF data
        records: Records to be updated
    """

    for record in records:
        coll.update_one({'name': record.name}, {'$set': asdict(record)})


def count_records(coll: Collection) -> int:
    """Count the number of records in our database

    Args:
        coll: Collection holding our MOF data
    Returns:
        Number of records
    """
    return coll.estimated_document_count()
