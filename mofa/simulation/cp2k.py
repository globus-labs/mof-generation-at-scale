"""Run computations backed by CP2K"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from subprocess import run
from pathlib import Path
import time
import sys
import os

import ase
from ase.calculators.cp2k import CP2K
from ase import units
from ase.filters import UnitCellFilter
from ase.io import Trajectory
from ase.optimize import LBFGS

from mofa.model import MOFRecord
from mofa.utils.conversions import read_from_string

_file_dir = Path(__file__).parent / 'files'

_cp2k_options = {
    'pbe': {
        'basis_set': 'DZVP-MOLOPT-SR-GTH',
        'basis_set_file': "BASIS_MOLOPT",
        'pseudo_potential': "GTH-PBE",
        'potential_file': "GTH_POTENTIALS",
        'xc': None,
        'cutoff': 600 * units.Ry
    }
}

# Get the path to the CP2K atomic density guesses
_atomic_density_folder_path = str(Path(sys.prefix) / "share" / "chargemol" / "atomic_densities")
if not _atomic_density_folder_path.endswith("/"):
    _atomic_density_folder_path = _atomic_density_folder_path + "/"


# Utility functions
def _load_structure(mof: MOFRecord, structure_source: tuple[str, int] | None):
    """Read the appropriate input structure"""
    if structure_source is None:
        return mof.atoms
    else:
        traj, ind = structure_source
        return read_from_string(mof.md_trajectory[traj][ind], 'vasp')


def compute_partial_charges(cp2k_path: Path, threads: int | None = 2):
    """Compute partial charges with DDEC

    Args:
        cp2k_path: Path to a CP2K computation which wrote a CUBE file
        threads: Number of threads to use for chargemol
    """

    # Make a copy of the input file
    with open(_file_dir / "chargemol" / "job_control.txt", "r") as rf:
        write_str = rf.read().replace("$$$ATOMIC_DENSITY_DIR$$$", _atomic_density_folder_path)
    with open(cp2k_path / "job_control.txt", "w") as wf:
        wf.write(write_str)

    # my local CP2k is not renaming the output automatically, so an extra renaming step is added here
    if not (cp2k_path / "valence_density.cube").is_file():
        cube_fname = list(cp2k_path.glob('*.cube'))
        if len(cube_fname) != 1:
            raise ValueError(f'Expected 1 cube file. Found {cube_fname}')
        cube_fname[0].rename(cp2k_path / "valence_density.cube")

    # chargemol uses all cores for OpenMP parallelism if no OMP_NUM_THREADS is set
    with open(cp2k_path / 'chargemol.stdout', 'w') as fo, open(cp2k_path / 'chargemol.stderr', 'w') as fe:
        if threads is not None:
            env = os.environ.copy()
            env['OMP_THREADS'] = str(threads)
        else:
            env = None
        proc = run(["chargemol"], cwd=cp2k_path, env=env, stdout=fo, stderr=fe)
        if proc.returncode != 0:
            raise ValueError(f'Chargemol failed in {cp2k_path}')

        # read output of chargemol
        chargemol_out_fname = "DDEC6_even_tempered_net_atomic_charges.xyz"
        with open(cp2k_path / chargemol_out_fname, "r") as cmresf:
            res_lines = cmresf.readlines()
        natoms = int(res_lines[0])
        lat_str = (" ".join(res_lines[1].split("unitcell")[1]
                            .replace("}", "").replace("{", "")
                            .replace("]", "").replace("[", "")
                            .replace(",", "").split()))
        extxyzstr = "".join([res_lines[0], '''Lattice="''' + lat_str + '''" Properties=species:S:1:pos:R:3:q:R:1\n'''] + res_lines[2:natoms + 2])
        return read_from_string(extxyzstr, fmt='extxyz')


@dataclass
class CP2KRunner:
    """Interface for running pre-defined CP2K workflows"""

    cp2k_invocation: str = 'cp2k_shell'
    """Invocation used to run CP2K on this system"""

    run_dir: Path = Path('cp2k-runs')
    """Path in which to store CP2K files"""

    def run_single_point(self, mof: MOFRecord,
                         level: str = 'pbe',
                         structure_source: tuple[str, int] | None = None) -> tuple[ase.Atoms, Path]:
        """Perform a single-point computation at a certain level

        Args:
            mof: Structure to be run
            level: Name of the level of DFT computation to perform
            structure_source: Name of the MD trajectory and frame ID from which to source the
                input structure. Default is to use the as-assembled structure
        Returns:
            - Structure with the
            - Path to the run directory
        """

        atoms = _load_structure(mof, structure_source)
        return self._run_cp2k(mof.name, atoms, 'single', level)

    def run_optimization(self, mof: MOFRecord,
                         level: str = 'pbe',
                         structure_source: tuple[str, int] | None = None,
                         steps: int = 8,
                         fmax: float = 1e-2) -> tuple[ase.Atoms, Path]:
        """Perform a single-point computation at a certain level

        Args:
            mof: Structure to be run
            level: Name of the level of DFT computation to perform
            structure_source: Name of the MD trajectory and frame ID from which to source the
                input structure. Default is to use the as-assembled structure
            steps: Maximum number of optimization steps
            fmax: Convergence threshold for optimization
        Returns:
            - Relaxed structure
            - Path to the run directory
        """

        atoms = _load_structure(mof, structure_source)
        return self._run_cp2k(mof.name, atoms, 'optimize', level, steps, fmax, ignore_failure=True)

    def _run_cp2k(self, name: str, atoms: ase.Atoms, action: str, level: str,
                  steps: int = 8, fmax: float = 1e-2, ignore_failure: bool = False) -> tuple[ase.Atoms, Path]:
        """Run CP2K in a special directory

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
        template_file = _file_dir / f'cp2k-{level}-template.inp'
        if not template_file.is_file():
            raise ValueError(f'Template not found for {level}')
        inp = template_file.read_text()
        if ignore_failure:
            # Does not work with CP2K<2024.1, which is what we're running on Polaris but not what comes with Ubuntu
            inp = inp.replace("&SCF\n", "&SCF\n         IGNORE_CONVERGENCE_FAILURE\n")

        # Get the other settings
        if level not in _cp2k_options:
            raise ValueError(f'No presents for {level}')
        options = _cp2k_options[level]

        # Open then move to the output directory
        #  CP2K does not like long directory names in input files, so we move to the local directory
        out_dir = self.run_dir / f'{name}-{action}-{level}'
        start_dir = Path().cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'cp2k.out').write_text('')  # Clear old content
        os.chdir(out_dir)
        with open('cp2k.stdout', 'w') as fo, redirect_stdout(fo), open('cp2k.stderr', 'w') as fe, redirect_stderr(fe):
            try:
                with CP2K(
                        command=self.cp2k_invocation,
                        directory=".",
                        inp=inp,
                        max_scf=128,
                        **options,
                ) as calc:
                    # Run the calculation
                    atoms = atoms.copy()
                    atoms.calc = calc
                    if action == 'single':
                        atoms.get_potential_energy()
                    elif action == 'optimize':
                        ecf = UnitCellFilter(atoms, hydrostatic_strain=False)
                        with Trajectory('relax.traj', mode='w') as traj:
                            dyn = LBFGS(ecf,
                                        logfile='relax.log',
                                        trajectory=traj)
                            dyn.run(fmax=fmax, steps=steps)
                    else:
                        raise ValueError(f'Action not supported: {action}')

                    # Write the result to disk for easy retrieval
                    atoms.write('atoms.json')
            except AssertionError:
                time.sleep(30)  # Give time for CP2K to exit cleanly
                raise
            finally:
                os.chdir(start_dir)
        return atoms, out_dir.absolute()
