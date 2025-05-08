import os
import shutil

from pytest import mark

from mofa.model import MOFRecord
from mofa.simulation.cp2k import CP2KRunner, compute_partial_charges
from mofa.utils.conversions import write_to_string

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@mark.parametrize('cif_name', ['hMOF-0'])
def test_cp2k_single(cif_name, cif_dir, tmpdir):
    # Make a CP2k simulator that reads and writes to a temporary directory
    runner = CP2KRunner(
        cp2k_invocation="cp2k_shell",
        close_cp2k=False,
    )

    test_file = cif_dir / f'{cif_name}.cif'
    record = MOFRecord.from_file(test_file)
    record.md_trajectory['uff'] = [write_to_string(record.atoms, 'vasp')]
    _, cp2k_path = runner.run_single_point(record, structure_source=('uff', -1))

    charged_mof = compute_partial_charges(cp2k_path, threads=2)
    assert charged_mof.arrays["q"].shape[0] == charged_mof.arrays["positions"].shape[0]

    # Run a second time to test re-using the executable
    old_run_dir = runner._calc.run_dir
    old_runtime = cp2k_path.joinpath('cp2k.out').stat().st_mtime
    _, cp2k_path = runner.run_single_point(record, structure_source=('uff', -1))
    assert old_run_dir == runner._calc.run_dir  # No changed path
    assert old_runtime != cp2k_path.joinpath('cp2k.out').stat().st_mtime  # Changed mod times


@mark.skipif(IN_GITHUB_ACTIONS, reason='Too expensive for CI')
@mark.parametrize('cif_name', ['hMOF-0'])
def test_cp2k_optimize(cif_name, cif_dir, tmpdir):
    shutil.rmtree("cp2k-runs")

    # Make a CP2k simulator that reads and writes to a temporary directory
    runner = CP2KRunner(
        cp2k_invocation="cp2k_shell",
        close_cp2k=True
    )

    test_file = cif_dir / f'{cif_name}.cif'
    record = MOFRecord.from_file(test_file)
    record.md_trajectory['uff'] = [write_to_string(record.atoms, 'vasp')]
    _, cp2k_path = runner.run_optimization(record, steps=2, fmax=0.1)
    assert cp2k_path.exists()
    assert cp2k_path.is_absolute()
    assert 'opt' in cp2k_path.name

    # Make sure IGNORE_CONVERGENCE was turned on
    assert 'IGNORE_CONVER' in (cp2k_path / 'cp2k.inp').read_text()

    charged_mof = compute_partial_charges(cp2k_path, threads=2)
    assert charged_mof.arrays["q"].shape[0] == charged_mof.arrays["positions"].shape[0]
