"""Run computations backed by DFTB+"""
from pathlib import Path
import os

import ase
from ase.calculators.dftb import Dftb
from ase import units
from ase.filters import UnitCellFilter
from ase.io import Trajectory
from ase.optimize import LBFGS

from mofa.model import MOFRecord
from mofa.utils.conversions import read_from_string


@dataclass
class DFTBPLUSRunner:
    """Interface for running pre-defined DFTB+ workflows"""

    dftbplus_invocation: str = 'dftb+'
    """Invocation used to run DFTB+ on this system"""

    run_dir: Path = Path('dftbplus-runs')
    """Path in which to store DFTB+ files"""

    def _run_dftbplus(self, name: str, atoms: ase.Atoms, action: str, level: str,
                      steps: int = 8, fmax: float = 1e-2, ignore_failure: bool = False) -> tuple[ase.Atoms, Path]:
        """Run DFTB+ in a special directory

        Args:
            name: Name used for the start of the directory
            atoms: Starting structure to use
            action: Which action to perform (single, opt)
            level: Level of accuracy to use
            steps: Number of steps to run
            fmax: Convergence threshold for optimization
            ignore_failure: Whether to ignore failures
        Returns:
            - Relaxed structure
            - Absolute path to the run directory
        """
        # Get the template for this level of computation

        return atoms, out_dir.absolute()
