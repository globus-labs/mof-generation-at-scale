"""Interface definitions"""
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
import subprocess
import shutil

import ase

from mofa.simulation.raspa.utils import get_cif_from_chargemol, write_cif


@dataclass(kw_only=True)
class BaseRaspaRunner:
    """Shared interface for all RASPA implementations"""

    # Settings used by every invocation of RASPA
    run_dir: Path
    """Path in which to store output files"""
    raspa_command: Sequence[str]
    """Command to launch a calculation"""
    delete_finished: bool = False
    """Whether to delete the run files after completing"""
    cutoff: float = 12.8
    """Van der Waals and Coulomb cutoff distance in Angstroms"""

    def run_gcmc(
            self,
            name: str,
            cp2k_dir: Path | str,
            adsorbate: str,
            cycles: int,
            temperature: float,
            pressure: float
    ) -> tuple[float, float, float, float]:
        """Run a GCMC calculation

        Args:
            name: Name of the MOF
            cp2k_dir: Path to a completed CP2K calculation, which contains the partial charges on each atom
                in the output files from a chargemol computation
            adsorbate: Name of the adsorbate
            cycles: Number of monte carlo steps
            temperature: Simulation temperature in Kelvin (K).
            pressure: Simulation pressure in Pascal (Pa).
        Returns:
             Computed uptake (U) and error (E) from RASPA2 in the following order:
                - U (mol/kg)
                - E (mol/kg)
                - U (g/L)
                - E (g/L)
        """
        out_dir = self.run_dir / f"{name}_{adsorbate}_{temperature}_{pressure:0e}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cp2k_dir = Path(cp2k_dir)

        try:
            # Write CIF file with charges
            atoms = get_cif_from_chargemol(cp2k_dir)
            write_cif(atoms, out_dir, "input.cif")

            # Invoke the version specific input writer
            self._write_inputs(out_dir, atoms, cycles, adsorbate, temperature, pressure)

            # Invoke RASPA
            with open(out_dir / "raspa.log", "w") as fp, open(out_dir / "raspa.err", "w") as fe:
                result = subprocess.run(self.raspa_command, cwd=out_dir, stdout=fp, stderr=fe)
            if result.returncode != 0:
                raise ValueError(f'RASPA failed in {out_dir}')

            # Read the results
            return self._read_outputs(out_dir, atoms, adsorbate)
        finally:
            if self.delete_finished:
                shutil.rmtree(out_dir)

    def _write_inputs(self,
                      out_dir: Path,
                      atoms: ase.Atoms,
                      cycles: int,
                      adsorbate: str,
                      temperature: float,
                      pressure: float):
        """Set up RASPA input files in a new directory

        The directory already contains a file named "input.cif"
        with the structure of the MOF

        Args:
            out_dir: Path in which to run RASPA
            atoms: Structure of the MOF
            adsorbate: Name of the adsorbate
            cycles: Number of monte carlo steps
            temperature: Simulation temperature in Kelvin (K).
            pressure: Simulation pressure in Pascal (Pa).
        """
        raise NotImplementedError()

    def _read_outputs(self, out_dir: Path, atoms: ase.Atoms, adsorbate: str) -> tuple[float, float, float, float]:
        """Read the uptakes and errors from a completed directory

        Args:
            out_dir: Directory in which RASPA was run
            atoms: Structure of MOF
            adsorbate: Name of the adsorbate
        Returns:
          Computed uptake (U) and error (E) in the following order:
            - U (mol/kg)
            - E (mol/kg)
            - U (g/L)
            - E (g/L)
        """
        raise NotImplementedError()
