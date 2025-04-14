"""Methods to pull training sets from DiffLinker that reflect the best-performing MOFs"""
from dataclasses import dataclass

from pymongo.collection import Collection
import pymongo

from mofa.model import MOFRecord


@dataclass(kw_only=True)
class DiffLinkerCurriculum:
    """Defines how to pull data for DiffLinker finetuning"""

    max_size: int = 3072
    """Maximum number of records to include in a training set"""
    collection: Collection
    """Collection holding MOF records"""

    strain_method: str = 'mace'
    """Which method to use for evaluating strain"""

    min_strain_counts: int = 64
    """How many strains to accumulate before we retrain"""
    max_strain: float = 0.25
    """Maximum allowed strain for a MOF"""
    gas_capacity_field: str = 'gas_storage.CO2'
    """On which field to sort for gas capacity"""
    min_gas_counts: int = 64
    """Minimum number of gas capacity counts before we use it"""

    def get_training_set(self) -> list[MOFRecord]:
        """Pull a training set for DiffLinker finetuning"""

        # Build the filter
        pipeline = [{
            '$project': {'md_trajectory': 0}  # Ignore the big fields
        }]
        query = {
            f'structure_stability.{self.strain_method}': {'$lt': self.max_strain}
        }
        pipeline.append({
            '$match': query
        })

        # Build the sort
        gas_counts = self.collection.count_documents({
            self.gas_capacity_field: {'$exists': True}
        })
        if gas_counts >= self.min_gas_counts:
            # Sort by the gas capacity
            sort_field = self.gas_capacity_field
            sort_order = pymongo.DESCENDING
            pipeline.append({
                '$sort': {sort_field: sort_order}
            })

            # Take only the top half
            pipeline.append({
                '$limit': gas_counts // 2
            })

            # If there are more than the limit, sample down
            if gas_counts > self.max_size:
                pipeline.append({
                    '$sample': {'size': self.max_size}
                })
        else:
            pipeline.append({
                '$sample': {'size': self.max_size}
            })

        # Run the aggregation
        cursor = self.collection.aggregate(pipeline)
        output = []
        for record in cursor:
            for bad in ['_id', 'position']:
                record.pop(bad, None)

            for missing in ['md_trajectory', 'times']:
                record[missing] = {}
            output.append(MOFRecord(**record))
        return output
