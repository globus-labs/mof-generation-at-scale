"""Simulation operations that involve RASPA"""
from typing import Sequence
from subprocess import run, CompletedProcess
from pathlib import Path
import os
import sys

import ase
import io
import shutil
import logging
import pandas as pd
from ase.io.lammpsrun import read_lammps_dump_text
from ase.geometry.cell import cell_to_cellpar

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

    def prep_common_files(self, run_name: str, raspa_path: str | Path, mof_ase_atoms: ase.Atoms):
        # MOF cif file with partial charge and labeled in RASPA convention
        cifdf = pd.DataFrame(mof_ase_atoms.get_scaled_positions(), columns=["xs", "ys", "zs"])
        cifdf["q"] = mof_ase_atoms.arrays["q"]
        cifdf["el"] = mof_ase_atoms.symbols
        a, b, c, alpha, beta, gamma = cell_to_cellpar(mof_ase_atoms.cell)
        
        label_map = {}
        for val, subdf in cifdf.groupby('el'):
            newmap = dict(zip(subdf.index.tolist(), subdf["el"] + [str(x) for x in range(0, len(subdf))]))
            if type(label_map) == type(None):
                label_map = dict(newmap)
            else:
                label_map.update(newmap)
        cifdf["label"] = cifdf.index.map(label_map)
        cifdf = cifdf[["label", "el", "xs", "ys", "zs", "q"]]
        cifstr = """
_audit_author_name 'xyan11@uic.edu'

_cell_length_a       """ + "%.8f" % a + """
_cell_length_b       """ + "%.8f" % b + """
_cell_length_c       """ + "%.8f" % c + """
_cell_angle_alpha    """ + "%.8f" % alpha + """
_cell_angle_beta     """ + "%.8f" % beta + """
_cell_angle_gamma    """ + "%.8f" % gamma + """
_cell_volume         """ + "%.8f" % mof_ase_atoms.cell.volume + """

_symmetry_cell_setting             triclinic
_symmetry_space_group_name_Hall    'P 1'
_symmetry_space_group_name_H-M     'P 1'
_symmetry_Int_Tables_number        1

loop_
_symmetry_equiv_pos_as_xyz
 'x,y,z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
""" + cifdf.to_string(header=None, index=None) + "\n"
        # Write the cif file to disk
        with open(Path(raspa_path) / f'{run_name}.cif', "w") as wf:
            wf.write(cifstr)

        # meta information about force field
        with open(Path(raspa_path) / "force_field.def", "w") as wf:
            wf.write("""# rules to overwrite
0
# number of defined interactions
0
# mixing rules to overwrite
0
""")

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
            os.remove(os.path.join(raspa_path, data_file_name))

        except Exception as e:
            shutil.rmtree(raspa_path)
            raise e

        read_str = None
        with io.open(os.path.join(raspa_path, data_file_rename), "r") as rf2:
            read_str = rf2.read()
        mass_df = read_lmp_sec_str2df(read_str.split(
            "Masses")[1].split("Pair Coeffs")[0].strip())
        pair_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Pair Coeffs")[1].split("Bond Coeffs")[0].strip())
        bond_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Bond Coeffs")[1].split("Angle Coeffs")[0].strip())
        angle_coeff_df = read_lmp_sec_str2df(read_str.split(
            "Angle Coeffs")[1].split("Dihedral Coeffs")[0].strip())
        dihedral_coeff_df = read_lmp_sec_str2df(read_str.split("Dihedral Coeffs")[
            1].split("Improper Coeffs")[0].strip())
        improper_coeff_df = read_lmp_sec_str2df(
            read_str.split("Improper Coeffs")[1].split("Atoms")[0].strip())
        cifbox = read_str.split("Atoms")[1].split("$$$atoms$$$")[0].strip()
        atom_df = read_lmp_sec_str2df(read_str.split("$$$atoms$$$")[
            1].split("Bonds")[0].strip())
        atom_df.columns = [
            'id', 'mol', 'type', 'q', 'x', 'y', 'z', '#', "comment"]
        _atom_df = pd.read_csv(
            io.StringIO(
                "\n".join(
                    atom_df["comment"].to_list())), sep=r"\s+", header=None, index_col=None, names=[
                "comment", "fx", "fy", "fz"])
        atom_df = pd.concat([atom_df[['id', 'mol', 'type', 'q', 'x', 'y', 'z', '#']].reset_index(
            drop=True), _atom_df.reset_index(drop=True)], axis=1)
        bond_df = read_lmp_sec_str2df(
            read_str.split("Bonds")[1].split("Angles")[0].strip())
        angle_df = read_lmp_sec_str2df(read_str.split(
            "Angles")[1].split("Dihedrals")[0].strip())
        dihedral_df = read_lmp_sec_str2df(read_str.split(
            "Dihedrals")[1].split("Impropers")[0].strip())
        improper_df = read_lmp_sec_str2df(
            read_str.split("Impropers")[1].strip())


    def run_He_void_single(self, run_name: str, atoms: ase.Atoms, timesteps: int, report_frequency: int, mof_ase_atoms) -> str:
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
        self.prep_common_files(raspa_path, mof_ase_atoms)
        with open(Path(raspa_path) / "helium.def", "w") as wf:
            wf.write("""# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
5.2
228000.0
-0.39
# Number Of Atoms
1
# Number Of Groups
1
# Alkane-group
flexible
# number of atoms
1
# atomic positions
0 He
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
               0    0            0    0            0       0        0            0         0            0         0               0            0        0            0
# Number of config moves
0
""")

        
        
        # He void fraction input
        with open(Path(raspa_path) / "simulation.input", "w") as wf:
            wf.write("""SimulationType                       MonteCarlo
NumberOfCycles                       40000
PrintEvery                           1000
PrintPropertiesEvery                 1000

Forcefield                           local

Framework 0
FrameworkName mof
UnitCells 2 2 2
ExternalTemperature 300.0

Component 0 MoleculeName             helium
            MoleculeDefinition       local
            WidomProbability         1.0
            CreateNumberOfMolecules  0
""")

        

        return raspa_path

    def run_gcmc(self, mof: MOFRecord, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run a molecular dynamics trajectory

        Args:
            mof: Record describing the MOF. Includes the structure in CIF format, which includes the bonding information used by UFF
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """

        # Generate the input files
        raspa_path = self.prep_gcmc_single(mof.name, mof.atoms, timesteps, report_frequency)
        with open(Path(raspa_path) / "CO2.def", "w") as wf:
            wf.write("""# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
304.1282
7377300.0
0.22394
#Number Of Atoms
3
# Number of groups
1
# CO2-group
rigid
# number of atoms
3
# atomic positions
0 O_co2     0.0           0.0           1.149
1 C_co2     0.0           0.0           0.0
2 O_co2     0.0           0.0          -1.149
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond""" + " " + \
          """Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
             0    2            0    0            0       0        0            0         0""" + "            " + \
          """0         0               0            0        0            0
# Bond stretch: atom n1-n2, type, parameters
0 1 RIGID_BOND
1 2 RIGID_BOND
# Number of config moves
0
""")
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
