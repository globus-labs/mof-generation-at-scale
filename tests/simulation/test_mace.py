import os
import shutil
from pathlib import Path

from pytest import mark
from mofa.model import MOFRecord
from mofa.simulation.mace import MACERunner
from mofa.utils.conversions import write_to_string

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@mark.parametrize("cif_name", ["hMOF-0"])
def test_mace_single(cif_name, cif_dir, tmpdir):
    # Make a MACE simulator that reads and writes to a temporary directory
    runner = MACERunner()

    test_file = cif_dir / f"{cif_name}.cif"
    record = MOFRecord.from_file(test_file)
    record.md_trajectory["uff"] = [(0, write_to_string(record.atoms, "vasp"))]
    atoms, mace_path = runner.run_single_point(record, structure_source=("uff", -1))

    # Check that the computation completed and produced expected outputs
    assert mace_path.exists()
    assert mace_path.is_absolute()
    assert "single" in mace_path.name
    assert (mace_path / "atoms.extxyz").exists()


@mark.skipif(IN_GITHUB_ACTIONS, reason="Too expensive for CI")
@mark.parametrize("cif_name", ["hMOF-0"])
def test_mace_optimize(cif_name, cif_dir, tmpdir):
    shutil.rmtree("mace-runs", ignore_errors=True)

    # Make a MACE simulator that reads and writes to a temporary directory
    runner = MACERunner()

    test_file = cif_dir / f"{cif_name}.cif"
    record = MOFRecord.from_file(test_file)
    record.md_trajectory["uff"] = [(0, write_to_string(record.atoms, "vasp"))]
    atoms, mace_path = runner.run_optimization(record, steps=2, fmax=0.1)

    # Check that optimization produced expected changes and outputs
    assert atoms != record.atoms
    assert mace_path.exists()
    assert mace_path.is_absolute()
    assert "optimize" in mace_path.name
    assert (mace_path / "atoms.extxyz").exists()
    assert (mace_path / "relax.traj").exists()
    assert (mace_path / "relax.log").exists()


@mark.parametrize("level", ["default"])
def test_mace_options(level, cif_dir):
    """Test that different MACE options work"""
    runner = MACERunner(delete_finished=True)

    test_file = cif_dir / "hMOF-0.cif"
    record = MOFRecord.from_file(test_file)
    atoms, mace_path = runner.run_single_point(record, level=level)

    assert not mace_path.exists()


def test_mace_md(cif_dir):
    test_file = cif_dir / "hMOF-0.cif"
    record = MOFRecord.from_file(test_file)

    # Initial run
    lammps_path = Path("qnviq")
    model_path = Path('../../input-files/mace/mace-mp0_medium-lammps.pt').absolute()
    runner = MACERunner(
        lammps_cmd=f"{lammps_path} -k on g 1 -sf kk".split() if lammps_path.exists() else None,
        model_path=model_path,
        delete_finished=True
    )
    output = runner.run_molecular_dynamics(
        mof=record,
        timesteps=4,
        report_frequency=2
    )
    assert len(output) == 3
    record.md_trajectory[runner.traj_name] = [
        (f, write_to_string(a, 'vasp')) for f, a in output
    ]

    # Continuation
    output = runner.run_molecular_dynamics(
        mof=record,
        timesteps=8,
        report_frequency=2
    )
    assert len(output) == 2
