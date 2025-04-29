from shutil import which

from pytest import mark

from mofa.simulation.raspa.graspa import gRASPARunner

graspa_path = which('graspa.x')


# TODO (wardlt): Spoof the outputs if gRASPA not available
@mark.skipif(graspa_path is None, reason='graspa not found')
@mark.parametrize("adsorbate,temperature,pressure", [("CO2", 298, 1e4), ("H2", 160, 5e5)])
def test_graspa_runner(adsorbate, temperature, pressure):
    """Test for gRASPA runner"""

    # Can use different sets of parameters
    params = {
        "name": "test",
        "cp2k_path": ".",
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "n_cycle": 100,
    }
    gr = gRASPARunner()
    gr.graspa_command = graspa_path
    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = gr.run_graspa(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)
