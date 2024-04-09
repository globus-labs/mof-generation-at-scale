"""Simulation operations that involve RASPA"""
from typing import Sequence
from subprocess import run, CompletedProcess
from pathlib import Path
import os

import ase
import io
import shutil
import logging
import pandas as pd
from ase.io.lammpsrun import read_lammps_dump_text

from .cif2lammps.main_conversion import single_conversion
from .cif2lammps.UFF4MOF_construction import UFF4MOF

from mofa.model import MOFRecord

logger = logging.getLogger(__name__)


class RASPARunner:
    """Interface for running pre-defined RASPA workflows

    Args:
        raspa_command: Command used to launch RASPA
        raspa_sims_root_path: Scratch directory for RASPA simulations
        raspa_environ: Additional environment variables to provide to RASPA
        delete_finished: Whether to delete run files once completed
    """

    def __init__(self,
                 raspa_command: Sequence[str] = (pathlib.Path(sys.prefix) / "bin" / "simulate",),
                 raspa_sims_root_path: str = "raspa_sims",
                 raspa_environ: dict[str, str] | None = None,
                 delete_finished: bool = True):
        self.raspa_command = raspa_command
        self.raspa_sims_root_path = raspa_sims_root_path
        os.makedirs(self.raspa_sims_root_path, exist_ok=True)
        self.raspa_environ = raspa_environ.copy()
        self.delete_finished = delete_finished

    def prep_molecular_dynamics_single(self, run_name: str, atoms: ase.Atoms, timesteps: int, report_frequency: int, stepsize_fs: float = 0.5) -> str:
        """Use cif2lammps to assign force field to a single MOF and generate input files for raspa simulation

        Args:
            run_name: Name of the run directory
            atoms: Starting structure
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
            stepsize_fs: Timestep size
        Returns:
            raspa_path: a directory with the raspa simulation input files
        """

        # Convert the cif_path to string, as that's what the underlying library uses
        raspa_path = os.path.join(self.raspa_sims_root_path, run_name)
        os.makedirs(raspa_path, exist_ok=True)

        # Write the cif file to disk
        cif_path = os.path.join(raspa_path, f'{run_name}.cif')
        atoms.write(cif_path, 'cif')
        try:
            single_conversion(cif_path,
                              force_field=UFF4MOF,
                              ff_string='UFF4MOF',
                              small_molecule_force_field=None,
                              outdir=raspa_path,
                              charges=False,
                              parallel=False,
                              replication='2x2x2',
                              read_cifs_pymatgen=True,
                              add_molecule=None,
                              small_molecule_file=None)
            in_file_name = [x for x in os.listdir(raspa_path) if x.startswith("in.") and not x.startswith("in.lmp")][0]
            data_file_name = [x for x in os.listdir(raspa_path) if x.startswith("data.") and not x.startswith("data.lmp")][0]
            logger.info("Reading data file for element list: " + os.path.join(raspa_path, data_file_name))
            with io.open(os.path.join(raspa_path, data_file_name), "r") as rf:
                df = pd.read_csv(io.StringIO(rf.read().split("Masses")[1].split("Pair Coeffs")[0]), sep=r"\s+", header=None)
                element_list = df[3].to_list()
            os.remove(os.path.join(raspa_path, in_file_name))


        except Exception as e:
            shutil.rmtree(raspa_path)
            raise e

        return raspa_path

    def run_molecular_dynamics(self, mof: MOFRecord, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run a molecular dynamics trajectory

        Args:
            mof: Record describing the MOF. Includes the structure in CIF format, which includes the bonding information used by UFF
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """

        # Generate the input files
        raspa_path = self.prep_molecular_dynamics_single(mof.name, mof.atoms, timesteps, report_frequency)

        # Invoke raspa
        try:
            ret = self.invoke_raspa(raspa_path)
            if ret.returncode != 0:
                raise ValueError('RASPA failed.' + ('' if self.delete_finished else f'Check the log files in: {raspa_path}'))

            # Read the output file
            with open(Path(raspa_path) / 'dump.lammpstrj.all') as fp:
                return read_lammps_dump_text(fp, slice(None))
        finally:
            if self.delete_finished:
                shutil.rmtree(raspa_path)

    def invoke_raspa(self, raspa_path: str | Path) -> CompletedProcess:
        """Invoke RASPA in a specific run directory

        Args:
            raspa_path: Path to the RASPA run directory
        Returns:
            Log from the completed process
        """

        raspa_path = Path(raspa_path)
        with open(raspa_path / 'stdout.raspa', 'w') as fp, open(raspa_path / 'stderr.raspa', 'w') as fe:
            env = None
            if self.raspa_environ is not None:
                env = os.environ.copy()
                env.update(self.raspa_environ)
            return run(list(self.raspa_command), cwd=raspa_path, stdout=fp, stderr=fe, env=env)
