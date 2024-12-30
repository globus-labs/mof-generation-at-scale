"""Simulation operations that involve LAMMPS"""
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


class LAMMPSRunner:
    """Interface for running pre-defined LAMMPS workflows

    Args:
        lammps_command: Command used to launch LAMMPS
        lmp_sims_root_path: Scratch directory for LAMMPS simulations
        lammps_environ: Additional environment variables to provide to LAMMPS
        delete_finished: Whether to delete run files once completed
    """

    def __init__(self,
                 lammps_command: Sequence[str] = ("lmp_serial",),
                 lmp_sims_root_path: str = "lmp_sims",
                 lammps_environ: dict[str, str] | None = None,
                 delete_finished: bool = True):
        self.lammps_command = lammps_command
        self.lmp_sims_root_path = lmp_sims_root_path
        os.makedirs(self.lmp_sims_root_path, exist_ok=True)
        #self.lammps_environ = lammps_environ.copy()
        self.lammps_environ = lammps_environ
        self.delete_finished = delete_finished

    def prep_molecular_dynamics_single(self, run_name: str, atoms: ase.Atoms, timesteps: int, report_frequency: int, stepsize_fs: float = 0.5) -> str:
        """Use cif2lammps to assign force field to a single MOF and generate input files for lammps simulation

        Args:
            run_name: Name of the run directory
            atoms: Starting structure
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
            stepsize_fs: Timestep size
        Returns:
            lmp_path: a directory with the lammps simulation input files
        """

        # Convert the cif_path to string, as that's what the underlying library uses
        lmp_path = os.path.join(self.lmp_sims_root_path, run_name)
        os.makedirs(lmp_path, exist_ok=True)

        # Write the cif file to disk
        cif_path = os.path.join(lmp_path, f'{run_name}.cif')
        atoms.write(cif_path, 'cif')
        try:
            single_conversion(cif_path,
                              force_field=UFF4MOF,
                              ff_string='UFF4MOF',
                              small_molecule_force_field=None,
                              outdir=lmp_path,
                              charges=False,
                              parallel=False,
                              replication='2x2x2',
                              read_cifs_pymatgen=True,
                              add_molecule=None,
                              small_molecule_file=None)
            in_file_name = [x for x in os.listdir(lmp_path) if x.startswith("in.") and not x.startswith("in.lmp")][0]
            data_file_name = [x for x in os.listdir(lmp_path) if x.startswith("data.") and not x.startswith("data.lmp")][0]
            in_file_rename = "in.lmp"
            data_file_rename = "data.lmp"
            logger.info("Reading data file for element list: " + os.path.join(lmp_path, data_file_name))
            with io.open(os.path.join(lmp_path, data_file_name), "r") as rf:
                df = pd.read_csv(io.StringIO(rf.read().split("Masses")[1].split("Pair Coeffs")[0]), sep=r"\s+", header=None)
                element_list = df[3].to_list()
            with io.open(os.path.join(lmp_path, in_file_rename), "w") as wf:
                logger.info("Writing input file: " + os.path.join(lmp_path, in_file_rename))
                with io.open(os.path.join(lmp_path, in_file_name), "r") as rf:
                    logger.info("Reading original input file: " + os.path.join(lmp_path, in_file_name))
                    wf.write(rf.read().replace(data_file_name, data_file_rename) + f"""

# simulation

fix                 fxnpt all npt temp 300.0 300.0 $(200.0*dt) tri 1.0 1.0 $(800.0*dt)
variable            Nevery equal {report_frequency}

thermo              ${{Nevery}}
thermo_style        custom step cpu dt time temp press pe ke etotal density xlo ylo zlo cella cellb cellc cellalpha cellbeta cellgamma
thermo_modify       flush yes

minimize            1.0e-10 1.0e-10 10000 100000
reset_timestep      0

dump                trajectAll all custom ${{Nevery}} dump.lammpstrj.all id type element x y z q
dump_modify         trajectAll element {" ".join(element_list)}

timestep            {stepsize_fs}
run                 {timesteps}
undump              trajectAll
write_restart       relaxing.*.restart
write_data          relaxing.*.data

""")
            os.remove(os.path.join(lmp_path, in_file_name))
            shutil.move(os.path.join(lmp_path, data_file_name), os.path.join(lmp_path, data_file_rename))

        except Exception as e:
            shutil.rmtree(lmp_path)
            raise e

        return lmp_path

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
        lmp_path = self.prep_molecular_dynamics_single(mof.name, mof.atoms, timesteps, report_frequency)

        # Invoke lammps
        try:
            ret = self.invoke_lammps(lmp_path)
            if ret.returncode != 0:
                raise ValueError('LAMMPS failed.' + ('' if self.delete_finished else f'Check the log files in: {lmp_path}'))

            # Read the output file
            with open(Path(lmp_path) / 'dump.lammpstrj.all') as fp:
                return read_lammps_dump_text(fp, slice(None))
        finally:
            if self.delete_finished:
                shutil.rmtree(lmp_path)

    def invoke_lammps(self, lmp_path: str | Path) -> CompletedProcess:
        """Invoke LAMMPS in a specific run directory

        Args:
            lmp_path: Path to the LAMMPS run directory
        Returns:
            Log from the completed process
        """

        lmp_path = Path(lmp_path)
        with open(lmp_path / 'stdout.lmp', 'w') as fp, open(lmp_path / 'stderr.lmp', 'w') as fe:
            env = None
            if self.lammps_environ is not None:
                env = os.environ.copy()
                env.update(self.lammps_environ)
            return run(list(self.lammps_command) + ['-i', 'in.lmp'], cwd=lmp_path, stdout=fp, stderr=fe, env=env)
