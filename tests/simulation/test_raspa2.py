from mofa.simulation.raspa2 import RASPA2Runner
from pytest import mark


#@mark.skip()
@mark.parametrize("adsorbate,temperature,pressure", [("CO2", 298, 1e4), ("H2", 160, 1e4)])
def test_raspa2_runner(adsorbate, temperature, pressure):
    """Test for RASPA2 runner"""

    # Can use different sets of parameters
    params = {
        "name": "test",
        "cp2k_path": ".",
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "n_cycle": 100,
    }
    raspa2_command = ("simulate")
    r2r = RASPA2Runner()
    r2r.raspa2_command = raspa2_command
    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = r2r.run_gcmc(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)