"""Simulation operations that involve LAMMPS"""
import ase
import io
import os
import shutil
import pandas as pd
from cif2lammps.main_conversion import single_conversion
from cif2lammps.UFF4MOF_construction import UFF4MOF


class LAMMPSRunner:
    """Interface for running pre-defined LAMMPS workflows

    Args:
        lammps_command: Command used to launch LAMMPS
    """
    lammps_command = "npt_tri"
    lmp_sims_root_path = "lmp_sims"
    cif_files_root_path = "cif_files"
    cif_files_paths = []

    def __init__(self, lammps_command: str = "npt_tri", lmp_sims_root_path: str = "lmp_sims", cif_files_root_path: str = "cif_files"):
        self.lammps_command = lammps_command
        self.lmp_sims_root_path = lmp_sims_root_path
        print("Making LAMMPS simulation root path at: " + os.path.join(os.getcwd(), self.lmp_sims_root_path))
        os.makedirs(self.lmp_sims_root_path, exist_ok=True)
        print("Scanning cif files at: " + os.path.join(os.getcwd(), self.cif_files_root_path))
        self.cif_files_root_path = cif_files_root_path
        self.cif_files_paths = [os.path.join(self.cif_files_root_path, x) for x in os.listdir(self.cif_files_root_path) if x.endswith(".cif")]
        print("Found " + "%d" % len(self.cif_files_paths) + " files with .cif extension! \n")

    def prep_molecular_dynamics_single(self, cif_path: str, timesteps: int, report_frequency: int, stepsize_fs: float = 0.5) -> str:
        """Run a molecular dynamics trajectory
        Args:
            cif_path: starting structure's cif file path
            timesteps: Number of timesteps to run
            report_frequency: How often to report structures
        Returns:
            Structures produced at specified intervals
        """
        cif_name = os.path.split(cif_path)[-1]
        lmp_path = os.path.join(self.lmp_sims_root_path, cif_name.replace(".cif", ""))
        os.makedirs(lmp_path, exist_ok=True)
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
            print("Reading data file for element list: " + os.path.join(lmp_path, data_file_name))
            with io.open(os.path.join(lmp_path, data_file_name), "r") as rf:
                df = pd.read_csv(io.StringIO(rf.read().split("Masses")[1].split("Pair Coeffs")[0]), sep=r"\s+", header=None)
                element_list = df[3].to_list()
            with io.open(os.path.join(lmp_path, in_file_rename), "w") as wf:
                print("Writing input file: " + os.path.join(lmp_path, in_file_rename))
                with io.open(os.path.join(lmp_path, in_file_name), "r") as rf:
                    print("Reading original input file: " + os.path.join(lmp_path, in_file_name))
                    wf.write(rf.read().replace(data_file_name, data_file_rename) + """

# simulation

fix                 fxnpt all npt temp 300.0 300.0 $(200.0*dt) tri 1.0 1.0 $(800.0*dt)
variable            Nevery equal """ + "%d" % report_frequency + """

thermo              ${Nevery}
thermo_style        custom step cpu dt time temp press pe ke etotal density xlo ylo zlo cella cellb cellc cellalpha cellbeta cellgamma
thermo_modify       flush yes

minimize            1.0e-10 1.0e-10 10000 100000
reset_timestep      0

dump                trajectAll all custom ${Nevery} dump.lammpstrj.all id type element x y z q
dump_modify         trajectAll element """ + " ".join(element_list) + """

timestep            0.5
run                 """ + "%d" % timesteps + """
undump              trajectAll
write_restart       relaxing.*.restart
write_data          relaxing.*.data

""")
            os.remove(os.path.join(lmp_path, in_file_name))
            shutil.move(os.path.join(lmp_path, data_file_name), os.path.join(lmp_path, data_file_rename))
            print("Success!!\n\n")

        except Exception as e:
            print(e)
            print("Failed!! Removing files...\n\n")
            shutil.rmtree(lmp_path)
        return lmp_path

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


if __name__ == "__main__":
    LMPrunner = LAMMPSRunner(lammps_command="npt_tri", lmp_sims_root_path="lmp_sims", cif_files_root_path="cif_files")
    test_file_index = 0
    lmp_path = LMPrunner.prep_molecular_dynamics_single(os.path.join(LMPrunner.cif_files_root_path,
                                                                     (LMPrunner.cif_file_names)[test_file_index]),
                                                        timesteps=200000, report_frequency=1000, stepsize_fs=0.5)
