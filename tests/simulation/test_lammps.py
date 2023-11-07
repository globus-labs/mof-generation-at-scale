from mofa.simulation.lammps import LAMMPSRunner
import logging


def test_lammps_runner(cif_files):
    lmprunner = LAMMPSRunner(
        lammps_command="npt_tri",
        lmp_sims_root_path="lmp_sims",
        cif_files_root_path=cif_files
    )

    for test_file in lmprunner.cif_files_paths:
        try:
            lmp_path = lmprunner.prep_molecular_dynamics_single(test_file,
                                                                timesteps=1000,
                                                                report_frequency=100,
                                                                stepsize_fs=0.5)
            logging.info(lmp_path)
        except Exception as e:
            logging.error(e)
