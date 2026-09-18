#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N cp2k-diagnostic
#PBS -A ChemGraph

set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the MOFA repository root}"
export MOFA_ENV="${MOFA_ENV:-${PWD}/mofa_env_py312}"
source bin/activate-mofa-polaris.sh
exec python -u cp2k-test/diagnose.py run "$@"
