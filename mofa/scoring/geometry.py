"""Metrics for screening a MOF based on its geometry"""
import numpy as np
import ase


def get_closest_atomic_distance(atoms: ase.Atoms) -> float:
    """Find the closest distance between atoms

    Args:
        atoms: Structure to be evaluated
    Returns:
        Distance between the closest atoms
    """

    dists: np.ndarray = atoms.get_all_distances(mic=True)
    inds = np.triu_indices(dists.shape[0], k=1)
    return np.min(dists[inds])
