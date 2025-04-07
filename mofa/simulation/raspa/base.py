"""Interface definitions"""

from pathlib import Path

from mofa.model import MOFRecord


class BaseRaspaRunner:
    """Shared interface for all RASPA implementations"""

    # Settings used by every invocation of RASPA
    run_path: Path
    """Path in which to store output files"""
    command: str
    """Command to launch a calculation"""
    delete_finished: bool = False
    """Whether to delete the run files after completing"""

    def run_gcmc(
            self,
            mof: MOFRecord,
            cp2k_dir: Path,
            adsorbate: str,
            steps: int,
            **kwargs  # Add more kwargs for runtime options specific to version of RASPA
    ) -> dict[str, tuple[float, float]]:
        """Run a GCMC calculation

        Args:
            mof: Record of MOF, which contains the name and the structure of the MOF
            cp2k_dir: Path to a completed CP2K calculation, which contains the partial charges on each atom
                in the output files from a chargemol computation
            adsorbate: Name of the adsorbate
            steps: Number of monte carlo steps
        Returns:
             Map of the adsorption property computed by RASPA (e.g., U) to the mean and standard deviation
        """
        raise NotImplementedError()
