"""Metrics for screening a MOF or linker based on its geometry"""
import numpy as np
import ase

from mofa.scoring.base import MOFScorer


class MinimumDistance(MOFScorer):
    """Rate molecules based on the closest distance between atoms"""

    def __call__(self, linker: ase.Atoms) -> float:
        dists: np.ndarray = linker.get_all_distances(mic=True)
        inds = np.triu_indices(dists.shape[0], k=1)
        return np.min(dists[inds])
