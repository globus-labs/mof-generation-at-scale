#!/bin/bash -l
# Create a separate MOFA environment; external simulations are not rebuilt.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/bin/polaris-paths.sh"

usage() {
    cat <<'EOF'
Usage: bash create_conda.sh [--prefix PATH]

Create and validate a new MOFA Conda environment on Polaris.
Default target: <repository>/mofa_env_py312
Relative paths are resolved from the directory where this command is run.
An existing target is never updated or removed. Use an unused --prefix.
MOFA_ENV does not select the build target; use --prefix explicitly.

Options:
  --prefix PATH  Location of the new environment
  -h, --help     Show this help without loading modules or installing packages

MOFA_CONDA_MODULE selects the bootstrap module (default: conda).
LAMMPS and CP2K remain in their separately managed installations.
EOF
}

build_prefix="${MOFA_REPO_ROOT}/mofa_env_py312"
while (($#)); do
    case "$1" in
        --prefix)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "--prefix requires a path." >&2
                exit 2
            fi
            build_prefix=$2
            shift 2
            ;;
        --prefix=*)
            build_prefix=${1#--prefix=}
            if [[ -z "${build_prefix}" ]]; then
                echo "--prefix requires a path." >&2
                exit 2
            fi
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -e "${build_prefix}" || -L "${build_prefix}" ]]; then
    echo "Refusing to overwrite existing target: ${build_prefix}. Choose an unused --prefix." >&2
    exit 2
fi
build_prefix=$(realpath -m -- "${build_prefix}")
if [[ -e "${build_prefix}" || -L "${build_prefix}" ]]; then
    echo "Refusing to overwrite existing target: ${build_prefix}. Choose an unused --prefix." >&2
    exit 2
fi
# This affects only this installer process, not the caller or job defaults.
export MOFA_ENV="${build_prefix}"
cd "${MOFA_REPO_ROOT}"

mkdir -p "${MOFA_REPO_ROOT}/.runtime-cache"
build_log=$(mktemp "${MOFA_REPO_ROOT}/.runtime-cache/mofa-env-build.XXXXXXXX.log")
exec > >(tee -a "${build_log}") 2>&1
trap 'status=$?; if ((status != 0)); then printf "Build/validation failed (exit %s). See %s. The target is retained for diagnosis.\n" "${status}" "${build_log}" >&2; fi' EXIT
printf 'MOFA target: %s\nBuild log: %s\n' "${MOFA_ENV}" "${build_log}"
ulimit -c 0

# Isolate the installer before Conda invokes the target Python/pip. Waiting
# until activate-mofa-polaris.sh is too late for dependency installation.
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
unset PYTHONHOME PYTHONPATH
bootstrap_module=${MOFA_CONDA_MODULE}
# Restore helpers if this shell inherited only the exported conda function.
if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
fi
module reset
module use /soft/modulefiles
# A site Conda unload hook can unset exported names containing CONDA.
module load "${bootstrap_module}"
export MOFA_CONDA_MODULE=${bootstrap_module}
# XALT injects a shared library and Python startup hooks. Keep these out of
# package metadata/wheel build subprocesses; this is local to the installer.
module unload xalt
unset PYTHONHOME PYTHONPATH
source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
conda activate base
printf 'Bootstrap modules:\n'
module -t list 2>&1
printf 'Build isolation: PYTHONNOUSERSITE=%s TMPDIR=%s\n' "${PYTHONNOUSERSITE}" "${TMPDIR}"
printf 'LD_PRELOAD=%s\nLD_LIBRARY_PATH=%s\n' "${LD_PRELOAD:-}" "${LD_LIBRARY_PATH:-}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MOFA_REPO_ROOT}/.cache/conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MOFA_REPO_ROOT}/.cache/pip}"
conda env create --file envs/environment-polaris.yml --prefix "${MOFA_ENV}"
conda activate "${MOFA_ENV}"
python -m pip install --no-deps -e .

source bin/activate-mofa-polaris.sh
# Record what was installed even if the following validation fails.
conda env export --prefix "${MOFA_ENV}" > "${MOFA_ENV}/mofa-environment.yml"
conda list --explicit --prefix "${MOFA_ENV}" > "${MOFA_ENV}/mofa-conda-explicit.txt"
python -m pip freeze > "${MOFA_ENV}/mofa-pip-freeze.txt"
module -t list > "${MOFA_ENV}/mofa-modules.txt" 2>&1
printf '%s\n' "${build_log}" > "${MOFA_ENV}/mofa-build-log.txt"

python bin/check-polaris.py --models --external
python run_parallel_workflow.py --help > /dev/null
printf 'MOFA CPU/environment checks passed: %s\n' "${MOFA_ENV}"
printf 'Compute-node workflow validation is still required; no jobs were submitted.\n'
printf 'To select this environment:\n  export MOFA_ENV=%q\n  source %q\n' \
    "${MOFA_ENV}" "${MOFA_REPO_ROOT}/bin/activate-mofa-polaris.sh"
