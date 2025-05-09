"""Interfaces for performing DFT calculations"""
import os
from pathlib import Path
from shutil import which
from subprocess import run

import ase

from mofa.simulation.dft.cp2k import _file_dir
from mofa.utils.conversions import read_from_string

# Get the path to the CP2K atomic density guesses
_chargemol_path = which('chargemol')
if _chargemol_path is not None:
    _atomic_density_folder_path = str(Path(_chargemol_path).parent / ".." / "share" / "chargemol" / "atomic_densities")
    if not _atomic_density_folder_path.endswith("/"):
        _atomic_density_folder_path = _atomic_density_folder_path + "/"
else:
    _atomic_density_folder_path = None


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
        return load_atoms_with_charges(cp2k_path)


def load_atoms_with_charges(cp2k_path) -> ase.Atoms:
    """Load the structure labeled with partial charges from a DDEC run

    Assumes chargemol has already been run

    Args:
        cp2k_path: Path to the CP2K run
    Returns:
        Atoms object complete with charges
    """
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
