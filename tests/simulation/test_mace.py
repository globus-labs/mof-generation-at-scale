import os
import shutil

from pytest import mark, importorskip

# Try to import MACE and check if models are available
mace = importorskip("mace")
try:
    from mace.calculators import mace_mp

    calc = mace_mp(
        model="medium", dispersion=True, default_dtype="float32", device="cpu"
    )
    MACE_AVAILABLE = True
except (ImportError, ValueError, RuntimeError):
    MACE_AVAILABLE = False

from mofa.model import MOFRecord
from mofa.simulation.mace import MACERunner
from mofa.utils.conversions import write_to_string

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@mark.skipif(not MACE_AVAILABLE, reason="MACE models not available")
@mark.parametrize("cif_name", ["hMOF-0"])
def test_mace_single(cif_name, cif_dir, tmpdir):
    # Make a MACE simulator that reads and writes to a temporary directory
    runner = MACERunner()

    test_file = cif_dir / f"{cif_name}.cif"
    record = MOFRecord.from_file(test_file)
    record.md_trajectory["uff"] = [write_to_string(record.atoms, "vasp")]
    atoms, mace_path = runner.run_single_point(record, structure_source=("uff", -1))

    # Check that the computation completed and produced expected outputs
    assert mace_path.exists()
    assert mace_path.is_absolute()
    assert "single" in mace_path.name
    assert (mace_path / "atoms.json").exists()
    assert atoms.get_potential_energy() is not None


@mark.skipif(not MACE_AVAILABLE, reason="MACE models not available")
@mark.skipif(IN_GITHUB_ACTIONS, reason="Too expensive for CI")
@mark.parametrize("cif_name", ["hMOF-0"])
def test_mace_optimize(cif_name, cif_dir, tmpdir):
    shutil.rmtree("mace-runs", ignore_errors=True)

    # Make a MACE simulator that reads and writes to a temporary directory
    runner = MACERunner()

    test_file = cif_dir / f"{cif_name}.cif"
    record = MOFRecord.from_file(test_file)
    record.md_trajectory["uff"] = [write_to_string(record.atoms, "vasp")]
    atoms, mace_path = runner.run_optimization(record, steps=2, fmax=0.1)

    # Check that optimization produced expected changes and outputs
    assert atoms != record.atoms
    assert mace_path.exists()
    assert mace_path.is_absolute()
    assert "optimize" in mace_path.name
    assert (mace_path / "atoms.json").exists()
    assert (mace_path / "relax.traj").exists()
    assert (mace_path / "relax.log").exists()
    assert atoms.get_potential_energy() is not None


@mark.skipif(not MACE_AVAILABLE, reason="MACE models not available")
@mark.parametrize("level", ["default"])
def test_mace_options(level, cif_dir):
    """Test that different MACE options work"""
    runner = MACERunner()

    test_file = cif_dir / "hMOF-0.cif"
    record = MOFRecord.from_file(test_file)
    atoms, mace_path = runner.run_single_point(record, level=level)
    assert atoms.get_potential_energy() is not None
