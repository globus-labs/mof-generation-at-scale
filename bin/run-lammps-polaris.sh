#!/bin/bash
# ML-IAP must use the Python environment belonging to the external build.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "${script_dir}/polaris-paths.sh"
lammps_exe="${LAMMPS_ROOT}/build-mliap-no-mpi/lmp"
if [[ ! -x "${lammps_exe}" || ! -f "${LAMMPS_VENV}/bin/activate" ]]; then
    echo "Missing external LAMMPS executable or environment: ${lammps_exe}, ${LAMMPS_VENV}" >&2
    exit 2
fi
source "${script_dir}/polaris-simulation-env.sh" >&2
source "${LAMMPS_VENV}/bin/activate"
exec "${lammps_exe}" "$@"
