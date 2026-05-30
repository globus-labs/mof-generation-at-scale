"""CPU-only single-host configuration for cloud VMs / non-HPC linux hosts.

Used by ``run_parallel_workflow.py --compute-config configs/cloud-vm.py``.
LAMMPS and CP2K resolve from $PATH (typically the active conda env's
``bin/``), with MOFA_LAMMPS_BIN / MOFA_CP2K_BIN env vars as overrides.
See [envs/chameleon-stream.md] §9 for the cloud-VM provisioning runbook.
"""
from mofa.hpc.config import CloudVMConfig


hpc_config = CloudVMConfig()
