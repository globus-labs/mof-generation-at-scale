from pathlib import Path
from mofa.simulation.lammps import LAMMPSRunner


def test_lammps_runner(cif_files):
    lmprunner = LAMMPSRunner(
        lammps_command="npt_tri",
        lmp_sims_root_path="lmp_sims",
    )

    for test_file in cif_files.glob("*.cif"):
        lmp_path = lmprunner.prep_molecular_dynamics_single(test_file,
                                                            timesteps=1000,
                                                            report_frequency=100,
                                                            stepsize_fs=0.5)
        assert Path(lmp_path).exists()
