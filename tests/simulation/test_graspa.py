from pathlib import Path
from shutil import which

from pytest import mark

from mofa.simulation.raspa.graspa import gRASPARunner

_file_path = Path(__file__).parent
graspa_path = which('graspa.x')
_cache_dir = Path(__file__).parent / 'graspa-runs' / 'cached'


@mark.parametrize("adsorbate,temperature,pressure", [("CO2", 298, 1e4), ("H2", 160, 5e5)])
def test_graspa_runner(adsorbate, temperature, pressure):
    """Test for gRASPA runner"""

    # Can use different sets of parameters
    params = {
        "name": "test",
        "cp2k_dir": _file_path,
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "cycles": 100,
    }
    if graspa_path is None:
        name = "{name}_{adsorbate}_{temperature}_{pressure:0e}".format(**params)
        graspa_command = f"cp {_cache_dir.absolute() / name}.log raspa.log".split()
    else:
        graspa_command = graspa_path
    gr = gRASPARunner(raspa_command=graspa_command, run_dir=_file_path)

    uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L = gr.run_gcmc(**params)
    assert isinstance(uptake_mol_kg, float)
    assert isinstance(error_mol_kg, float)
    assert isinstance(uptake_g_L, float)
    assert isinstance(error_g_L, float)
