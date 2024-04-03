from pytest import mark

from mofa.model import MOFRecord
from mofa.simulation.cp2k import CP2KRunner


@mark.parametrize('cif_name', ['hMOF-0'])
def test_cp2k_runner(cif_name, cif_dir, tmpdir):
    # Make a CP2k simulator that reads and writes to a temporary directory
    runner = CP2KRunner(
        cp2k_invocation="cp2k_shell.psmp",
        run_ddec=True,
    )

    test_file = cif_dir / f'{cif_name}.cif'
    record = MOFRecord.from_file(test_file)
    runner.run_single_point(record)
