from mofa.simulation.graspa import gRASPARunner
from ase.io import read
from pytest import mark
from pathlib import Path

@mark.parametrize('adsorbate,temperature,pressure,unit', 
    [
        ("CO2",298,1e4,"mol/kg"),
        ("H2",160,5e5,"g/L")
    ]
)
def test_graspa_runner(adsorbate, temperature, pressure, unit):
    """Test for gRASPA runner"""

    # Can use different sets of parameters
    params = {'name': 'test',
              'cp2k_path': '.',
              'adsorbate': adsorbate,
              'temperature': temperature,
              'pressure': pressure,
              'n_cycle': 100,
              'write_output': True, # Can set to False to store outputs in stdout and stderr. Probably not worth it.
              'unit': unit
              }
    graspa_command = "/opt/cray/pals/1.3.4/bin/mpiexec -n 1 --cpu-bind list:0 /eagle/projects/HPCBot/thang/soft/gRASPA/src_clean/nvc_main.x" # Change this to your graspa command.
    gr = gRASPARunner()
    gr.graspa_command = graspa_command
    uptake, error = gr.run_graspa(**params)

    assert isinstance(uptake, float)
    assert isinstance(error, float)
