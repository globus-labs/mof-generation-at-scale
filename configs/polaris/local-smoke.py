"""Single-node Polaris configuration for an end-to-end MOFA smoke test."""

import os
from shlex import quote
from pathlib import Path

from pydantic import computed_field

from mofa.hpc.config import LocalConfig, RASPAVersion


ROOT = Path.cwd().resolve()
if not (ROOT / "run_parallel_workflow.py").is_file():
    raise RuntimeError(
        "Load this configuration from the MOFA repository root. "
        f"Current directory: {ROOT}"
    )


class Config(LocalConfig):
    """Run each accelerator-backed stage sequentially on one Polaris GPU."""

    lammps_cmd: tuple[str, ...] = (
        str(ROOT / "bin/run-lammps-polaris.sh"),
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "on", "neigh", "half",
    )

    raspa_version: RASPAVersion = "raspa2"
    raspa_cmd: tuple[str, ...] = (
        str(Path(os.environ.get("MOFA_ENV") or ROOT / "mofa_env_py312").resolve() / "bin/simulate"),
    )

    # LocalConfig otherwise permits DFT on the CPU helper executor as well.
    dft_executors: list[str] = ["gpu"]
    raspa_executors: list[str] = ["gpu"]

    @computed_field
    @property
    def dft_cmd(self) -> str:
        return (
            "env CP2K_BINARY=cp2k_shell.ssmp "
            + quote(str(ROOT / "bin/run-cp2k-polaris.sh"))
        )


hpc_config = Config()
