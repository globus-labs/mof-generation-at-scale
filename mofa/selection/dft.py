"""Selecting which DFT calculations to perform"""
from dataclasses import dataclass

from pymongo.collection import Collection

from mofa.db import row_to_record
from mofa.model import MOFRecord


@dataclass(kw_only=True)
class DFTSelector:
    """Pick which DFT calculation to run next"""

    # Criteria
    md_level: str = 'mace'
    """Name of the MD method used to qualify stability"""
    max_strain: float = 0.25
    """Maximum level of strain to allow"""
    collection: Collection
    """Collection of MOF records from previous calculations"""

    def select_next(self) -> MOFRecord:
        """Select which MOF to run next

        Returns:
            The selected MOF record
        """

        # Find a MOF which is not currently running and has a strain below the target
        stages = [
            {'$match': {'in_progress': {'$nin': ['dft']}}},
            {'$match': {f'structure_stability.{self.md_level}': {'$lt': self.max_strain}}},
            {'$sample': {'size': 1}}  # Pick randomly
        ]

        for record in self.collection.aggregate(stages):
            return row_to_record(record)
        raise ValueError('No MOFs match the criteria')
