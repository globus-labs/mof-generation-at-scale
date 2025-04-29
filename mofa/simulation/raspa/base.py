"""Interface definitions"""
from pathlib import Path


class BaseRaspaRunner:
    """Shared interface for all RASPA implementations"""

    # Settings used by every invocation of RASPA
    run_path: Path
    """Path in which to store output files"""
    command: str
    """Command to launch a calculation"""
    delete_finished: bool = False
    """Whether to delete the run files after completing"""
    cutoff: float = 12.8
    """Van der Waals and Coulomb cutoff distance in Angstroms"""

    def run_gcmc(
            self,
            name: str,
            cp2k_dir: Path,
            adsorbate: str,
            steps: int,
            temperature: float,
            pressure: float,
    ) -> tuple[float, float, float, float]:
        """Run a GCMC calculation

        Args:
            name: Name of the MOF
            cp2k_dir: Path to a completed CP2K calculation, which contains the partial charges on each atom
                in the output files from a chargemol computation
            adsorbate: Name of the adsorbate
            steps: Number of monte carlo steps
            temperature: Simulation temperature in Kelvin (K).
            pressure: Simulation pressure in Pascal (Pa).
        Returns:
             Computed uptake (U) and error (E) from RASPA2 in the following order:
                - U (mol/kg)
                - E (mol/kg)
                - U (g/L)
                - E (g/L)
        """
        raise NotImplementedError()
