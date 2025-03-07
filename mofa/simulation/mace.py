"""Run computations backed by MACE"""

from dataclasses import dataclass
from pathlib import Path
import os

import ase
from mace.calculators import mace_mp
from ase import units
from ase.filters import UnitCellFilter
from ase.io import Trajectory
from ase.optimize import LBFGS

from mofa.model import MOFRecord
from mofa.utils.conversions import read_from_string

_mace_options = {
    "default": {
        "model": "medium",  # Can be 'small', 'medium', or 'large'
        "device": "cpu",  # Can be 'cpu' or 'cuda'
        "default_dtype": "float32",
        "dispersion": False,  # Whether to include dispersion corrections
    }
}


def _load_structure(mof: MOFRecord, structure_source: tuple[str, int] | None):
    """Read the appropriate input structure"""
    if structure_source is None:
        return mof.atoms
    else:
        traj, ind = structure_source
        return read_from_string(mof.md_trajectory[traj][ind], "vasp")


@dataclass
class MACERunner:
    """Interface for running pre-defined MACE workflows"""

    run_dir: Path = Path("mace-runs")
    """Path in which to store MACE computation files"""

    def run_single_point(
        self,
        mof: MOFRecord,
        level: str = "default",
        structure_source: tuple[str, int] | None = None,
    ) -> tuple[ase.Atoms, Path]:
        """Perform a single-point computation at a certain level

        Args:
            mof: Structure to be run
            level: Name of the level of computation to perform
            structure_source: Name of the MD trajectory and frame ID from which to source the
                input structure. Default is to use the as-assembled structure
        Returns:
            - Structure with computed properties
            - Path to the run directory
        """
        atoms = _load_structure(mof, structure_source)
        return self._run_mace(mof.name, atoms, "single", level)

    def run_optimization(
        self,
        mof: MOFRecord,
        level: str = "default",
        structure_source: tuple[str, int] | None = None,
        steps: int = 8,
        fmax: float = 1e-2,
    ) -> tuple[ase.Atoms, Path]:
        """Perform a geometry optimization computation

        Args:
            mof: Structure to be run
            level: Name of the level of computation to perform
            structure_source: Name of the MD trajectory and frame ID from which to source the
                input structure. Default is to use the as-assembled structure
            steps: Maximum number of optimization steps
            fmax: Convergence threshold for optimization
        Returns:
            - Relaxed structure
            - Path to the run directory
        """
        atoms = _load_structure(mof, structure_source)
        return self._run_mace(mof.name, atoms, "optimize", level, steps, fmax)

    def _run_mace(
        self,
        name: str,
        atoms: ase.Atoms,
        action: str,
        level: str,
        steps: int = 8,
        fmax: float = 1e-2,
    ) -> tuple[ase.Atoms, Path]:
        """Run MACE in a special directory

        Args:
            name: Name used for the start of the directory
            atoms: Starting structure to use
            action: Which action to perform (single, optimize)
            level: Level of accuracy to use
            steps: Number of steps to run
            fmax: Convergence threshold for optimization
        Returns:
            - Structure with computed properties
            - Absolute path to the run directory
        """
        if level not in _mace_options:
            raise ValueError(f"No presets for {level}")
        options = _mace_options[level]

        # Create and move to output directory
        out_dir = self.run_dir / f"{name}-{action}-{level}"
        start_dir = Path().cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(out_dir)

        try:
            # Initialize MACE calculator
            calc = mace_mp(**options)
            atoms = atoms.copy()
            atoms.calc = calc

            # Run the calculation
            if action == "single":
                atoms.get_potential_energy()
            elif action == "optimize":
                ecf = UnitCellFilter(atoms, hydrostatic_strain=False)
                with Trajectory("relax.traj", mode="w") as traj:
                    dyn = LBFGS(ecf, logfile="relax.log", trajectory=traj)
                    dyn.run(fmax=fmax, steps=steps)
            else:
                raise ValueError(f"Action not supported: {action}")

            # Write the result to disk for easy retrieval
            atoms.write("atoms.json")
        finally:
            os.chdir(start_dir)

        return atoms, out_dir.absolute()
