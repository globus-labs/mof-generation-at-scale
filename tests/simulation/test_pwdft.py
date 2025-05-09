import os
import shutil
from shutil import which

from pytest import mark
from mofa.model import MOFRecord
from mofa.simulation.dft.pwdft import PWDFTRunner
from mofa.utils.conversions import write_to_string

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def has_pwdft():
    """Check if pwdft binary is available in PATH"""
    return which('pwdft') is not None


@mark.slow
@mark.skipif(not has_pwdft(), reason="PWDFT binary not found in PATH")
@mark.parametrize("cif_name", ["hMOF-0"])
def test_pwdft_single(cif_name, cif_dir, tmpdir):
    # Make a PWDFT simulator that reads and writes to a temporary directory
    runner = PWDFTRunner(dft_cmd="mpiexec -n 12 pwdft")

    test_file = cif_dir / f"{cif_name}.cif"
    record = MOFRecord.from_file(test_file)
    record.md_trajectory["uff"] = [write_to_string(record.atoms, "vasp")]
    atoms, pwdft_path = runner.run_single_point(record, structure_source=("uff", -1))

    # Check that the computation completed and produced expected outputs
    assert pwdft_path.exists()
    assert pwdft_path.is_absolute()
    assert "single" in pwdft_path.name
    assert (pwdft_path / "atoms.json").exists()
    assert atoms.get_potential_energy() is not None


@mark.slow
@mark.skipif(IN_GITHUB_ACTIONS, reason="Too expensive for CI")
@mark.skipif(not has_pwdft(), reason="PWDFT binary not found in PATH")
@mark.parametrize("cif_name", ["hMOF-0"])
def test_pwdft_optimize(cif_name, cif_dir, tmpdir):
    shutil.rmtree("pwdft-runs", ignore_errors=True)

    runner = PWDFTRunner(dft_cmd="mpiexec -n 12 pwdft")

    test_file = cif_dir / f"{cif_name}.cif"
    record = MOFRecord.from_file(test_file)
    record.md_trajectory["uff"] = [write_to_string(record.atoms, "vasp")]
    atoms, pwdft_path = runner.run_optimization(record, steps=2, fmax=0.1)

    # Check that optimization produced expected changes and outputs
    assert atoms != record.atoms
    assert pwdft_path.exists()
    assert pwdft_path.is_absolute()
    assert "optimize" in pwdft_path.name
    assert (pwdft_path / "atoms.json").exists()
    assert (pwdft_path / "relax.traj").exists()
    assert (pwdft_path / "relax.log").exists()
    assert atoms.get_potential_energy() is not None


@mark.skipif(not has_pwdft(), reason="PWDFT binary not found in PATH")
@mark.parametrize("level", ["default"])
def test_pwdft_options(level, cif_dir):
    """Test that different PWDFT options work"""
    runner = PWDFTRunner(dft_cmd="mpirun -n 12 pwdft")

    test_file = cif_dir / "hMOF-0.cif"
    record = MOFRecord.from_file(test_file)
    atoms, pwdft_path = runner.run_single_point(record, level=level)
    assert atoms.get_potential_energy() is not None
