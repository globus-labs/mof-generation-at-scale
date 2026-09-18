#!/bin/bash
# Source in the job shell and in every Parsl worker.
source "$(dirname -- "${BASH_SOURCE[0]}")/polaris-paths.sh"
if [[ ! -x "${MOFA_ENV}/bin/python" ]]; then
    echo "Missing MOFA Python: ${MOFA_ENV}/bin/python. Expected the validated mofa_env_py312 environment." >&2
    return 2
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
# The site's Conda unload hook unsets every exported variable whose name
# contains CONDA. Preserve our module selection across module reset.
_mofa_module_choice=${MOFA_CONDA_MODULE}

# Parsl may export the conda function without its private helper functions.
# Restore them before module reset runs the site's conda unload hook.
if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
fi
module reset
export MOFA_CONDA_MODULE=${_mofa_module_choice}
unset _mofa_module_choice
module use /soft/modulefiles
module load "${MOFA_CONDA_MODULE}"
source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
unset PYTHONHOME PYTHONPATH
conda activate "${MOFA_ENV}"
module load cray-mpich-abi

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
runtime_cache="${MOFA_REPO_ROOT}/.runtime-cache/${PBS_JOBID:-local}"
mkdir -p "${runtime_cache}/matplotlib" "${runtime_cache}/xdg"
export MPLCONFIGDIR="${runtime_cache}/matplotlib"
export XDG_CACHE_HOME="${runtime_cache}/xdg"
