from pathlib import Path

from pytest import mark

from mofa.model import MOFRecord
from mofa.simulation.lammps import LAMMPSRunner


@mark.parametrize('cif_name', ['hMOF-0', 'hMOF-5000000'])
def test_lammps_runner(cif_name, cif_dir, tmpdir):
    # Make a LAMMPS simulator that reads and writes to a
    lmprunner = LAMMPSRunner(
        lammps_command=["lmp"],
        lmp_sims_root_path=tmpdir / "lmp_sims",
        lammps_environ={'OMP_NUM_THREADS': '1'}
    )

    # Make sure the preparation works
    test_file = cif_dir / f'{cif_name}.cif'
    record = MOFRecord.from_file(test_file)
    lmp_path = lmprunner.prep_molecular_dynamics_single(record.name,
                                                        record.atoms,
                                                        timesteps=1000,
                                                        report_frequency=100,
                                                        stepsize_fs=0.5)

    assert Path(lmp_path).exists()

    # Make sure that it runs
    ret = lmprunner.invoke_lammps(lmp_path)
    assert ret.returncode == 0

    # Test the full pipeline, forcing deletion on the end
    lmprunner.delete_finished = True
    traj = lmprunner.run_molecular_dynamics(record, timesteps=200, report_frequency=100)
    assert len(traj) == 3
    assert not Path(lmp_path).exists()
