"""Polaris scaling with a MOFA Conda environment and external simulations."""

import os
from shlex import quote
from pathlib import Path
from subprocess import Popen

from parsl import Config as ParslConfig
from parsl.launchers import WrappedLauncher
from pydantic import Field, computed_field

from mofa.hpc.config import RASPAVersion, SingleJobHPCConfig


ROOT = Path.cwd().resolve()
if not (ROOT / "run_parallel_workflow.py").is_file():
    raise RuntimeError(
        "Load this configuration from the MOFA repository root. "
        f"Current directory: {ROOT}"
    )


# These executors use mpiexec only to place independent Parsl managers on each
# node. Skip Slingshot VNI allocation so they do not consume CXI resources.
NO_VNI_EXECUTORS = {"inf", "train", "lammps", "helper"}
CP2K_CXI_PID_BASE = 5
# Capture overrides so the same paths reach managers on every compute node.
RUNTIME_ENV = {name: os.environ[name] for name in (
    "MOFA_ENV", "MOFA_CONDA_MODULE", "LAMMPS_ROOT", "LAMMPS_VENV",
    "CP2K_ROOT", "CP2K_DATA_DIR",
) if name in os.environ}
RUNTIME_ENV["MOFA_ENV"] = str(Path(os.environ.get("MOFA_ENV") or ROOT / "mofa_env_py312").resolve())


class Config(SingleJobHPCConfig):
    """Use ML-IAP, distributed CP2K, and RASPA2 on Polaris."""

    lammps_per_gpu: int = Field(default=2, init=False)
    # Preserve the original two-node CP2K layout (four ranks per node).
    nodes_per_cp2k: int = Field(default=2, init=False)

    lammps_cmd: tuple[str, ...] = (
        str(ROOT / "bin/run-lammps-polaris.sh"),
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "on", "neigh", "half",
    )
    raspa_version: RASPAVersion = "raspa2"
    raspa_cmd: tuple[str, ...] = (str(Path(RUNTIME_ENV["MOFA_ENV"]) / "bin/simulate"),)

    worker_init: str = "\n".join([
        "set -euo pipefail",
        *(f"export {name}={quote(value)}" for name, value in RUNTIME_ENV.items()),
        f"source {quote(str(ROOT / 'bin/activate-mofa-polaris.sh'))}",
        f"cd {quote(str(ROOT))}",
        "which python",
        "hostname",
    ])

    @computed_field
    @property
    def dft_cmd(self) -> str:
        """Launch four GPU-bound CP2K ranks on each assigned node."""
        if self.run_dir is None:
            raise ValueError("run_dir must be set before constructing dft_cmd")
        affinity = ROOT / "bin/set-affinity-gpu-polaris.sh"
        cp2k = ROOT / "bin/run-cp2k-polaris.sh"
        hostfiles = self.run_dir.absolute() / "cp2k-hostfiles"
        return (
            f"env MPICH_OFI_CXI_PID_BASE={CP2K_CXI_PID_BASE} "
            f"mpiexec -n {self.nodes_per_cp2k * self.gpus_per_node} "
            f"--ppn {self.gpus_per_node} --cpu-bind depth --depth 8 "
            f"-env OMP_NUM_THREADS=8 -env CP2K_BINARY=cp2k_shell.psmp --hostfile "
            f"{quote(str(hostfiles))}/local_hostfile.`printf %04d $PARSL_WORKER_RANK` "
            f"{quote(str(affinity))} {quote(str(cp2k))}"
        )

    def launch_monitor_process(self, freq: int = 20) -> Popen:
        """Launch one independent utilization monitor per node without a VNI."""
        log_dir = self.run_dir / "logs"
        return Popen(
            args=(
                f"mpiexec --no-vni -n {len(self.hosts)} --ppn 1 "
                f"--depth={self.cpus_per_node} --cpu-bind depth "
                f"monitor_utilization --frequency {freq} {log_dir.absolute()}"
            ).split()
        )

    def make_parsl_config(self) -> ParslConfig:
        """Build the standard Polaris layout with this environment on workers."""
        config = super().make_parsl_config()
        configured_launchers = set()
        for executor in config.executors:
            provider = getattr(executor, "provider", None)
            if provider is not None and hasattr(provider, "worker_init"):
                provider.worker_init = self.worker_init

            if executor.label not in NO_VNI_EXECUTORS:
                continue
            launcher = getattr(provider, "launcher", None)
            if not isinstance(launcher, WrappedLauncher):
                raise TypeError(
                    f"Executor {executor.label!r} must use WrappedLauncher "
                    "to configure its Polaris VNI behavior"
                )
            if not launcher.prepend.startswith("mpiexec "):
                raise ValueError(
                    f"Executor {executor.label!r} has an unexpected launcher: "
                    f"{launcher.prepend!r}"
                )
            launcher.prepend = launcher.prepend.replace(
                "mpiexec ", "mpiexec --no-vni ", 1
            )
            configured_launchers.add(executor.label)

        missing = NO_VNI_EXECUTORS - configured_launchers
        if missing:
            raise RuntimeError(
                "Missing Polaris executors requiring --no-vni: "
                + ", ".join(sorted(missing))
            )
        return config


hpc_config = Config()
