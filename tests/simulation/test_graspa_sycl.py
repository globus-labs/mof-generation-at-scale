from pathlib import Path
from shutil import which

from mofa.simulation.raspa.graspa_sycl import GRASPASyclRunner
from pytest import mark

_file_path = Path(__file__).parent
graspa_scyl_path = which('sycl.out')
_cache_dir = Path(__file__).parent / 'gRASPA-sycl-runs' / 'cached'


@mark.parametrize(
    "adsorbate,temperature,pressure", [("CO2", 298, 1e4), ("H2", 160, 1e4)]
)
def test_graspa_sycl_runner(adsorbate, temperature, pressure):
    """Test for gRASPA-sycl runner"""

    # Can use different sets of parameters
    params = {
        "name": "test",
        "cp2k_dir": _file_path,
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "cycles": 100,
    }

    if graspa_scyl_path is None:
        name = "{name}_{adsorbate}_{temperature}_{pressure:0e}".format(**params)
        graspa_command = ["cp", f"{_cache_dir.absolute() / name}.log", "raspa.log"]
    else:
        graspa_command = [graspa_scyl_path]
    gr = GRASPASyclRunner(run_dir=_file_path, raspa_command=graspa_command)

    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = gr.run_gcmc(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)
