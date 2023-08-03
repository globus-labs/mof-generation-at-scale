"""Interfaces for functions which produce scores on either the full MOF or just the geometry of the linker"""

import ase


class LinkerScorer:
    """Scoring functions which operate on the linkers produced by the generator

    Examples:
        - Verify the linker is chemically reasonable (e.g., SAScore, SCScore)
    """

    def __call__(self, linker: ase.Atoms) -> float:
        raise NotImplementedError()


class MOFScorer:
    """Scoring functions which produce a quick estimate of the quality of a MOF

    Examples:
        - Regression model for MOF -> CO_2 storage capacity
        - Classification model which predicts likelihood of collapse
    """

    def __call__(self, mof: ase.Atoms) -> float:
        raise NotImplementedError()
