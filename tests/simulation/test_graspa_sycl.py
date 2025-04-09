from pathlib import Path
from mofa.simulation.graspa_sycl import GRASPASyclRunner
from pytest import mark


_file_path = Path(__file__).parent


@mark.skip()
@mark.parametrize(
    "adsorbate,temperature,pressure", [("CO2", 298, 1e4), ("H2", 160, 1e4)]
)
def test_graspa_sycl_runner(adsorbate, temperature, pressure):
    """Test for gRASPA-sycl runner"""

    # Can use different sets of parameters
    params = {
        "name": "test",
        "cp2k_path": _file_path,
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "n_cycle": 100,
    }
    graspa_sycl_command = "sycl.out"
    gr = GRASPASyclRunner()
    gr.graspa_sycl_command = graspa_sycl_command

    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = gr.run_gcmc(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)
