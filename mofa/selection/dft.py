"""Selecting which DFT calculations to perform"""
from dataclasses import dataclass

from pymongo.collection import Collection

from mofa.db import row_to_record
from mofa.model import MOFRecord


@dataclass(kw_only=True)
class DFTSelector:
    """Pick which DFT calculation to run next

    Find entries which:
        - are not currently running ('dft' not in `in_progress`)
        - have been relaxed (``times.relaxed`` is available)
        - have strains below a threshold
    """

    # Criteria
    md_level: str = 'mace'
    """Name of the MD method used to qualify stability"""
    max_strain: float = 0.25
    """Maximum level of strain to allow"""
    collection: Collection
    """Collection of MOF records from previous calculations"""

    @property
    def match_stages(self) -> list[dict]:
        """Stages used to match applicable records"""
        return [
            {'$match': {'in_progress': {'$nin': ['dft']}}},
            {'$match': {'times.relaxed': {'$exists': True}}},
            {'$match': {f'structure_stability.{self.md_level}': {'$not': {'$gt': self.max_strain}}}}
        ]

    def count_available(self) -> int:
        """Count the number of MOFs available for MD within the database"""
        stages = self.match_stages
        stages.append({'$count': 'available'})
        for result in self.collection.aggregate(stages):
            return result['available']

    def select_next(self) -> MOFRecord:
        """Select which MOF to run next

        Returns:
            The selected MOF record
        """
        stages = self.match_stages
        stages.append({'$sample': {'size': 1}})  # Pick randomly
        for record in self.collection.aggregate(stages):
            return row_to_record(record)
        raise ValueError('No MOFs match the criteria')
