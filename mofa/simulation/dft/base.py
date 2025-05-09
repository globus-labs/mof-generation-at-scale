"""Base class for DFT computations"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path

import ase
from ase.calculators.calculator import Calculator
from ase.filters import UnitCellFilter
from ase.io import Trajectory
from ase.optimize import LBFGS

from mofa.model import MOFRecord
from mofa.utils.conversions import read_from_string, canonicalize


def _load_structure(mof: MOFRecord, structure_source: tuple[str, int] | None):
    """Read the appropriate input structure"""
    if structure_source is None:
        return mof.atoms
    else:
        traj, ind = structure_source
        return read_from_string(mof.md_trajectory[traj][ind], "vasp")


@dataclass
class BaseDFTRunner:
    """Interface for tools that run density functional theory (DFT) to produce
    the energy and partial charges of atoms in a structure"""

    run_dir: Path = Path('dft-runs')
    """Directory in which to write output files"""
    dft_cmd: str | None = None
    """Command which launches the DFT code"""

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
        return self._run_calc(mof.name, atoms, 'single', level)

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
            - Relaxed structure with computed properties
            - Path to the run directory
        """
        atoms = _load_structure(mof, structure_source)
        return self._run_calc(mof.name, atoms, 'optimize', level, steps, fmax)

    @contextmanager
    def _make_calc(self, level: str, out_dir: Path) -> Calculator:
        """Create the Calculator which drives this DFT code

        Should handle closing the calculator after completion

        Args:
            level: Fidelity level for the computation
            out_dir: Output directory to use for this computation
        Returns:
            Calculator object, ready to compute
        """
        raise NotImplementedError()

    def _run_calc(self, name: str, atoms: ase.Atoms, action: str, level: str,
                  steps: int = 8, fmax: float = 1e-2) -> tuple[ase.Atoms, Path] | None:
        """Run CP2K in a special directory

        Args:
            name: Name used for the start of the directory
            atoms: Starting structure to use
            action: Which action to perform (single, opt)
            level: Level of accuracy to use
            steps: Number of steps to run
            fmax: Convergence threshold for optimization
        Returns:
            - Relaxed structure
            - Absolute path to the run directory
        """

        # Either run in the eventual output directory if one MOF per CP2K,
        #  or run in a separate dir if we will re-use the CP2K executable
        out_dir = self.run_dir / f'{name}-{action}-{level}'
        out_dir = out_dir.absolute()

        # Begin execution
        out_dir.mkdir(parents=True, exist_ok=True)
        with (open(out_dir / 'cp2k.stdout', 'w') as fo, redirect_stdout(fo),
              open(out_dir / 'cp2k.stderr', 'w') as fe, redirect_stderr(fe),
              self._make_calc(level, out_dir) as calc):

            # Run the calculation
            atoms = atoms.copy()
            atoms.calc = calc
            if action == 'single':
                atoms.get_potential_energy()
            elif action == 'optimize':
                ecf = UnitCellFilter(atoms, hydrostatic_strain=False)
                with Trajectory(out_dir / 'relax.traj', mode='w') as traj:
                    dyn = LBFGS(ecf,
                                logfile=out_dir / 'relax.log',
                                trajectory=traj)
                    dyn.run(fmax=fmax, steps=steps)
            else:
                raise ValueError(f'Action not supported: {action}')

            # Write the result to disk for easy retrieval
            atoms.write(out_dir / 'atoms.json')

        return canonicalize(atoms), out_dir
