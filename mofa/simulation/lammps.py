"""Simulation operations that involve LAMMPS"""
import ase

from mofa.model import MOFRecord


class LAMMPSRunner:
    """Interface for running pre-defined LAMMPS workflows

    Args:
        lammps_command: Command used to launch LAMMPS
    """

    def __init__(self, lammps_command: str):
        self.lammps_command: str = lammps_command

    def run_molecular_dynamics(self, mof: MOFRecord, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run a molecular dynamics trajectory

        Args:
            mof: Record describing the MOF. Includes the structure in CIF format, which includes the bonding information used by UFF
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """

        raise NotImplementedError()
