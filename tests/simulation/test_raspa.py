from pytest import mark
from mofa.simulation.raspa import RASPARunner
from ase.io import read
from pathlib import Path


@mark.parametrize('extxyz_name', ['test-zn'])
def test_raspa_runner(extxyz_name, cif_dir, tmpdir):
    # Make a RASPA simulator that reads and writes to a temporary directory
    runner = RASPARunner()
    test_file = Path(cif_dir) / f'{extxyz_name}.extxyz'
    ase_atoms = read(test_file, format="extxyz")
    ads_mean, ads_std = runner.run_GCMC_single(ase_atoms, run_name="some_random_string", timesteps=200, report_frequency=1)
    assert isinstance(ads_mean, float)
    assert isinstance(ads_std, float)
