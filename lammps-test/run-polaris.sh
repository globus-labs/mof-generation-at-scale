#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N lammps-diagnostic
#PBS -A ChemGraph

# Submit from the repository root. Can also run inside a one-node allocation.
set -euo pipefail
cd "${PBS_O_WORKDIR:?Run this script inside a PBS allocation}"
repo_root=$PWD
if [[ ! -f "${repo_root}/lammps-test/diagnose.py" ]]; then
    echo "Submit from the MOFA repository root." >&2
    exit 2
fi
if [[ $(sort -u "${PBS_NODEFILE:?}" | wc -l) -ne 1 ]]; then
    echo "This diagnostic requires exactly one allocated node." >&2
    exit 2
fi

source "${repo_root}/bin/polaris-paths.sh"
# The controller needs only the Python standard library. Do not activate MOFA.
diagnostic_python="${LAMMPS_VENV}/bin/python"
if [[ ! -x "${diagnostic_python}" ]]; then
    echo "Missing LAMMPS Python: ${diagnostic_python}" >&2
    exit 2
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
args=(--input-dir "${LAMMPS_TEST_DIR:-${repo_root}/lammps-test}"
      --max-per-gpu "${LAMMPS_TEST_MAX_PER_GPU:-4}")
if [[ "${LAMMPS_TEST_FULL:-0}" == 1 ]]; then
    args+=(--full)
fi
exec "${diagnostic_python}" -u "${repo_root}/lammps-test/diagnose.py" "${args[@]}" "$@"
