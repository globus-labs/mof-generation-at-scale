"""Interfaces for functions which produce scores on either the full MOF or just the geometry of the linker"""

import ase
from mofa.model import MOFRecord
import io
import numpy as np
import pandas as pd


class Scorer:
    """Base class for tools which score a MOF

    Most implementations should be subclasses of the :class:`LigandScorer` or
    :class:`MOFScorer`, which provide utility that - for example - extract
    the information about the linker.
    """

    def score_mof(self, record: MOFRecord) -> float:
        """Score a MOF given the full MOF record

        Args:
            record: Record to be scored
        Returns:
            Score value
        """


class LigandScorer(Scorer):
    """Scoring functions which operate on the ligands between nodes in the MOF

    Examples:
        - Verify the ligand is chemically reasonable (e.g., SAScore, SCScore)
    """

    def __call__(self, linker: ase.Atoms) -> float:
        raise NotImplementedError()

    def score_mof(self, record: MOFRecord) -> float:
        assert len(record.ligands) == 1, 'We do not yet know how to score a MOF with >1 type of linker'  # TOOD
        raise NotImplementedError()


class MOFScorer(Scorer):
    """Scoring functions which produce a quick estimate of the quality of a MOF

    Examples:
        - Regression model for MOF -> CO_2 storage capacity
        - Classification model which predicts likelihood of collapse
    """

    def __call__(self, mof: ase.Atoms) -> float:
        raise NotImplementedError()

    def score_mof(self, record: MOFRecord) -> float:
        return self(record.atoms)

# some scripts to tell if a mof is stable
__threshold__ = 0.05  # 5 percent
df = pd.read_json("mofs-25Mar24.json.gz", lines=True)
for i in df.index:
    mof_name = df.at[i, "name"]
    traj_list = df.at[i, "md_trajectory"]["uff"]
    M = [np.loadtxt(io.StringIO("\n".join(list(filter(None, traj_list[x].strip().split("\n")))[2:5]))) for x in range(0, len(traj_list))]
    M = np.array(M)
    A = M[:, 0, :]
    B = M[:, 1, :]
    C = M[:, 2, :]
    a = np.linalg.norm(A, axis=1)
    b = np.linalg.norm(B, axis=1)
    c = np.linalg.norm(C, axis=1)
    alpha = np.arccos(np.sum(B * C, axis=1) / b / c) * 180 / np.pi
    beta = np.arccos(np.sum(A * C, axis=1) / a / c) * 180 / np.pi
    gamma = np.arccos(np.sum(A * B, axis=1) / a / b) * 180 / np.pi
    print(mof_name, np.all((a.mean() - a) / a < __threshold__) & \
            np.all((b.mean() - b) / b < __threshold__) & \
            np.all((c.mean() - c) / c < __threshold__) & \
            np.all((alpha.mean() - alpha) / alpha < __threshold__) & \
            np.all((beta.mean() - beta) / beta < __threshold__) & \
            np.all((gamma.mean() - gamma) / gamma < __threshold__))
