"""Metrics for screening a MOF or linker based on its geometry"""
from io import StringIO

import numpy as np
import ase
from ase.io.vasp import read_vasp

from mofa.model import MOFRecord
from dataclasses import dataclass
from mofa.scoring.base import MOFScorer, Scorer


class MinimumDistance(MOFScorer):
    """Rate molecules based on the closest distance between atoms"""

    def __call__(self, linker: ase.Atoms) -> float:
        dists: np.ndarray = linker.get_all_distances(mic=True)
        inds = np.triu_indices(dists.shape[0], k=1)
        return np.min(dists[inds])


@dataclass
class LatticeParameterChange(Scorer):
    """Score the stability of a MOF based on the lattice parameter change"""

    md_level: str = 'uff'
    """Level of accuracy used for the molecular dynamics simulation"""
    md_length: int = 1000
    """Length of the MD calculation"""

    def score_mof(self, record: MOFRecord) -> float:
        # Get the trajectory from the record
        if self.md_level not in record.md_trajectory:
            raise ValueError(f'No data available for MD simulations at level: "{self.md_level}"')
        traj = record.md_trajectory[self.md_level][str(self.md_length)]

        # Get the initial and final structures
        init_strc = read_vasp(StringIO(traj[0]))
        final_strc: ase.Atoms = read_vasp(StringIO(traj[-1]))

        # Compute the maximum principal strain
        #  Following: https://www.cryst.ehu.es/cryst/strain.html
        strain = np.matmul(init_strc.cell.array, np.linalg.inv(final_strc.cell.array)) - np.eye(3)
        strain = 0.5 * (strain + strain.T)
        strains = np.linalg.eigvals(strain)
        return np.abs(strains).max()
