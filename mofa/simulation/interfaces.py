"""Interfaces for performing specific types of simulations"""

import ase

from mofa.model import MOFRecord


class MDInterface:
    """Interface for tools which perform molecular dynamics"""

    traj_name: str
    """Name to use for MD performed with this class"""

    def run_molecular_dynamics(self,
                               mof: MOFRecord,
                               timesteps: int,
                               report_frequency: int) -> list[tuple[int, ase.Atoms]]:
        """Run NPT molecular dynamics

        Start from the last structure of the previous trajectory

        Args:
            mof: Record describing the MOF. Includes the structure in CIF format, which includes the bonding information used by UFF
            timesteps: How many total timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """
        raise NotImplementedError()
