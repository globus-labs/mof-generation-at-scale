from pathlib import Path

from pytest import mark

from mofa.model import MOFRecord
from mofa.simulation.lammps import LAMMPSRunner


@mark.parametrize('cif_name', ['hMOF-0'])
def test_lammps_runner(cif_name, cif_dir, tmpdir):
    # Make a LAMMPS simulator that reads and writes to a
    lmprunner = LAMMPSRunner(
        lammps_command=["lmp_serial"],
        lmp_sims_root_path=tmpdir / "lmp_sims",
    )

    # Make sure the preparation works
    test_file = cif_dir / f'{cif_name}.cif'
    lmp_path = lmprunner.prep_molecular_dynamics_single(test_file,
                                                        timesteps=1000,
                                                        report_frequency=100,
                                                        stepsize_fs=0.5)

    assert Path(lmp_path).exists()

    # Make sure that it runs
    ret = lmprunner.invoke_lammps(lmp_path)
    assert ret.returncode == 0

    # Test the full pipeline
    record = MOFRecord.from_file(test_file)
    traj = lmprunner.run_molecular_dynamics(record, timesteps=200, report_frequency=100)
    assert len(traj) == 3
