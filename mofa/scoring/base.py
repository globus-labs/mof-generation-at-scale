"""Interfaces for functions which produce scores on either the full MOF or just the geometry of the linker"""

import ase

from mofa.model import MOFRecord


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
