#!/bin/bash
# Source this file to configure the repository and external installations.
MOFA_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export MOFA_REPO_ROOT
export MOFA_ENV="${MOFA_ENV:-${MOFA_REPO_ROOT}/mofa_env_py312}"
export MOFA_CONDA_MODULE="${MOFA_CONDA_MODULE:-conda}"
export LAMMPS_ROOT="${LAMMPS_ROOT:-/lus/eagle/projects/ChemGraph/thang/soft/lammps}"
export LAMMPS_VENV="${LAMMPS_VENV:-${LAMMPS_ROOT}/.venv}"
export CP2K_ROOT="${CP2K_ROOT:-/lus/eagle/projects/ChemGraph/thang/soft/cp2k}"
export CP2K_DATA_DIR="${CP2K_DATA_DIR:-${CP2K_ROOT}/data}"
