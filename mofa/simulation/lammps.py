"""Simulation operations that involve LAMMPS"""
import ase
import io, os
from cif2lammps.main_conversion import single_conversion
from cif2lammps.UFF4MOF_construction import UFF4MOF


class LAMMPSRunner:
    """Interface for running pre-defined LAMMPS workflows

    Args:
        lammps_command: Command used to launch LAMMPS
    """

    def __init__(self, lammps_command: str):
        self.lammps_command: str = lammps_command

    def prep_molecular_dynamics(self, mof: cif_str) -> str:
        """Run a molecular dynamics trajectory
        Args:
            mof: Starting structure
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """
        
        raise NotImplementedError()

    def run_molecular_dynamics(self, mof: ase.Atoms, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run a molecular dynamics trajectory

        Args:
            mof: Starting structure
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """

        raise NotImplementedError()
