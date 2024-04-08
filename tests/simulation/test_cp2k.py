from pytest import mark

from mofa.model import MOFRecord
from mofa.simulation.cp2k import CP2KRunner, compute_partial_charges
from mofa.utils.conversions import write_to_string


@mark.parametrize('cif_name', ['hMOF-0'])
def test_cp2k_runner(cif_name, cif_dir, tmpdir):
    # Make a CP2k simulator that reads and writes to a temporary directory
    runner = CP2KRunner(
        cp2k_invocation="cp2k_shell.psmp",
    )

    test_file = cif_dir / f'{cif_name}.cif'
    record = MOFRecord.from_file(test_file)
    record.md_trajectory['uff'] = [write_to_string(record.atoms, 'vasp')]
    cp2k_path = runner.run_single_point(record, structure_source=('uff', -1))

    charged_mof = compute_partial_charges(cp2k_path, threads=2)
    assert charged_mof.arrays["q"].shape[0] == charged_mof.arrays["positions"].shape[0]
    
