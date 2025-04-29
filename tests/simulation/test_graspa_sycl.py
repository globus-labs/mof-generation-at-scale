from pathlib import Path
from shutil import which

from mofa.simulation.raspa.graspa_sycl import GRASPASyclRunner
from pytest import mark

_file_path = Path(__file__).parent

has_graspa_scyl = which('sycl.out') is not None


# TODO (wardlt): Spoof the outputs if gRASPA not available
@mark.skipif(not has_graspa_scyl, reason='gRASPA-sycl not found')
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
    graspa_sycl_command = [which("sycl.out")]
    gr = GRASPASyclRunner()
    gr.graspa_command = graspa_sycl_command

    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = gr.run_gcmc(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)
