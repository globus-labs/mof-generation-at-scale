from mofa.simulation.graspa import gRASPARunner
from pytest import mark


@mark.skip()
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
    graspa_command = (
        "/opt/cray/pals/1.3.4/bin/mpiexec -n 1 --cpu-bind list:0 "
        "/eagle/projects/HPCBot/thang/soft/gRASPA/src_clean/nvc_main.x"
    )
    gr = gRASPARunner()
    gr.graspa_command = graspa_command
    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = gr.run_graspa(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)
