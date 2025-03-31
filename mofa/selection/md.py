"""Modules for selecting which MD calculation to evaluate next"""
from dataclasses import dataclass

from pymongo.collection import Collection
import numpy as np

from mofa.model import MOFRecord
from mofa.db import row_to_record


@dataclass(kw_only=True)
class MDSelector:
    """Pick which MD calculation to run next"""

    maximum_steps: int | None = None
    """Maximum number of MD steps to run for any one MOF"""
    md_level: str = 'mace'
    """Which level of molecular dynamics to perform"""
    max_strain: float = 0.25
    """Maximum level of strain to allow"""
    new_fraction: float = 0.5
    """How often to run a new MOF instead of restarting a previous"""
    collection: Collection
    """Collection of MOF records from previous calculations"""

    def select_next(self, new_mofs: list[MOFRecord]) -> MOFRecord:
        """Select which MOF to run next

        Args:
            new_mofs: List of MOFs yet to be added to the database.
                A MOF will be removed from this list if selected
        Returns:
            The selected MOF record
        """

        if np.random.random() < self.new_fraction:
            return new_mofs.pop()

        # Find a MOF which is still below the target level
        stages = [{'$match': {'in_progress': {'$nin': ['stability']}}}]
        if self.maximum_steps is not None:
            stages.append({'$match': {
                f'md_trajectory.{self.md_level}': {
                    '$not': {'$elemMatch': {'$gte': self.maximum_steps}}
                }}})
        stages.append({
            '$match': {f'structure_stability.{self.md_level}': {'$lt': self.max_strain}}
        })
        stages.append({'$sample': {'size': 1}})

        for record in self.collection.aggregate(stages):
            return row_to_record(record)
        if len(new_mofs) == 0:
            raise ValueError('No MOFs match the criteria')
        return new_mofs.pop()
