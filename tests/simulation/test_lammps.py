import os
from mofa.simulation.lammps import LAMMPSRunner


def test_LMPRunner():
    LMPrunner = LAMMPSRunner(lammps_command="npt_tri",
                             lmp_sims_root_path="lmp_sims",
                             cif_files_root_path="mofa/simulation/cif_files")
    for test_file in LMPrunner.cif_files_paths:
        lmp_path = LMPrunner.prep_molecular_dynamics_single(test_file,
                                                            timesteps=1000, report_frequency=100, stepsize_fs=0.5)
        print(lmp_path)
