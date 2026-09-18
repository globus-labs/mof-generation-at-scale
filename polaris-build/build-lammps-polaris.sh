#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N lammps-build
#PBS -A ChemGraph
#PBS -j oe

# Submit from the MOFA repository root; see instruction.md for source checkout.
set -euo pipefail
case "${1:-}" in
    -h|--help)
        cat <<'HELP'
Build the external LAMMPS checkout for Polaris A100 GPUs (Kokkos/ML-IAP, no MPI).
From the MOFA repository root:
  qsub -v LAMMPS_ROOT=/absolute/path/to/lammps polaris-build/build-lammps-polaris.sh
Required: LAMMPS_ROOT, a separate checkout at the revision in instruction.md.
Optional: LAMMPS_VENV (default: $LAMMPS_ROOT/.venv), LAMMPS_BUILD_JOBS (32),
          MOFA_CONDA_MODULE (conda).
Creates Python 3.12.13 in $LAMMPS_ROOT/.python, then the venv and build-mliap-no-mpi.
Existing Python environments or build directories are refused. No sources are cloned.
HELP
        exit 0 ;;
    '') ;;
    *) echo "Unknown argument: $1; use --help." >&2; exit 2 ;;
esac

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Run this script inside a Polaris PBS compute-node job." >&2
    exit 2
fi
repo_root=$(cd -- "${PBS_O_WORKDIR:-$(dirname -- "${BASH_SOURCE[0]}")/..}" && pwd -P)
if [[ ! -f "${repo_root}/bin/polaris-simulation-env.sh" ]]; then
    echo "Submit from the MOFA repository root (PBS_O_WORKDIR)." >&2
    exit 2
fi
if [[ "${LAMMPS_ROOT:-}" != /* ]]; then
    echo "Set LAMMPS_ROOT to an absolute external source-checkout path." >&2
    exit 2
fi
lammps_root=$(realpath -e -- "${LAMMPS_ROOT}")
case "${lammps_root}/" in
    "${repo_root}/"*) echo "LAMMPS_ROOT must be outside the MOFA checkout." >&2; exit 2 ;;
esac
expected_revision=d51bbd4983a26e2da6f1550e1b41592690b02a90
if [[ ! -f "${lammps_root}/cmake/CMakeLists.txt" ||
      "$(git -C "${lammps_root}" rev-parse HEAD)" != "${expected_revision}" ]]; then
    echo "Expected LAMMPS revision ${expected_revision}; see polaris-build/instruction.md." >&2
    exit 2
fi
export LAMMPS_VENV="${LAMMPS_VENV:-${lammps_root}/.venv}"
if [[ "${LAMMPS_VENV}" != /* ]]; then
    echo "LAMMPS_VENV must be an absolute path." >&2
    exit 2
fi
LAMMPS_VENV=$(realpath -m -- "${LAMMPS_VENV}")
case "${LAMMPS_VENV}/" in
    "${repo_root}/"*) echo "LAMMPS_VENV must be outside the MOFA checkout." >&2; exit 2 ;;
esac
python_prefix="${lammps_root}/.python"
build_dir="${lammps_root}/build-mliap-no-mpi"
build_jobs="${LAMMPS_BUILD_JOBS:-32}"
if [[ ! "${build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LAMMPS_BUILD_JOBS must be a positive integer." >&2
    exit 2
fi
exec 9>"${lammps_root}/.build-lammps.lock"
if ! flock -n 9; then
    echo "Another LAMMPS build is active for ${lammps_root}." >&2
    exit 3
fi
for target in "${python_prefix}" "${LAMMPS_VENV}" "${build_dir}"; do
    if [[ -e "${target}" || -L "${target}" ]]; then
        echo "Refusing existing build/environment: ${target}. Use a fresh external checkout." >&2
        exit 2
    fi
done

# Keep the interpreter and its shared library accessible alongside LAMMPS,
# rather than creating a venv backed by a Python installation in a private home.
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
bootstrap_module="${MOFA_CONDA_MODULE:-conda}"
if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
fi
module reset
module use /soft/modulefiles
module load "${bootstrap_module}"
module unload xalt
if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
fi
unset PYTHONHOME PYTHONPATH
export CONDA_PKGS_DIRS="${lammps_root}/.cache/conda"
export PIP_CACHE_DIR="${lammps_root}/.cache/pip"
conda create --yes --override-channels --channel conda-forge \
    --prefix "${python_prefix}" python=3.12.13 pip
"${python_prefix}/bin/python" -m venv "${LAMMPS_VENV}"

# Compile with the same module stack used by bin/run-lammps-polaris.sh.
source "${repo_root}/bin/polaris-simulation-env.sh"
module load spack-pe-base cmake
source "${LAMMPS_VENV}/bin/activate"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L${CUDA_HOME}/lib64 -lcudart ${LDFLAGS:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export OMP_NUM_THREADS=1
module list
nvcc_version=$(nvcc --version)
if [[ "${nvcc_version}" != *"release 13.0"* || "${CRAY_ACCEL_TARGET:-}" != nvidia80 ]]; then
    echo "Expected CUDA 13.0.x and the Polaris A100 target nvidia80." >&2
    exit 1
fi

python -m pip install -r "${lammps_root}/python/wheel_requirements.txt"
python -m pip install -r "${repo_root}/polaris-build/lammps-requirements.txt"
python -m pip check

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${build_dir}" \
    -D BUILD_MPI=OFF \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_C_COMPILER=cc \
    -D CMAKE_CXX_COMPILER="${lammps_root}/lib/kokkos/bin/nvcc_wrapper" \
    -D Python_EXECUTABLE="${LAMMPS_VENV}/bin/python" \
    -D Python3_EXECUTABLE="${LAMMPS_VENV}/bin/python" \
    -D Python_FIND_VIRTUALENV=ONLY \
    -D PKG_KOKKOS=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_EXTRA-MOLECULE=ON \
    -D PKG_KSPACE=ON \
    -D PKG_ML-SNAP=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_PYTHON=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_AMPERE80=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ARCH_AMDAVX=ON \
    -D FFT_KOKKOS=CUFFT \
    -D FFT_SINGLE=yes \
    -D CMAKE_EXE_LINKER_FLAGS="-target-accel=nvidia80" \
    -S "${lammps_root}/cmake" \
    -B "${build_dir}"
cmake --build "${build_dir}" --parallel "${build_jobs}"
cmake --build "${build_dir}" --target install-python

if [[ ! -x "${build_dir}/lmp" ]]; then
    echo "Missing LAMMPS executable: ${build_dir}/lmp" >&2
    exit 1
fi
lammps_help=$("${build_dir}/lmp" -help)
for package in ML-IAP ML-SNAP KOKKOS PYTHON EXTRA-MOLECULE; do
    if ! grep -qw -- "${package}" <<<"${lammps_help}"; then
        echo "LAMMPS is missing package ${package}." >&2
        exit 1
    fi
done
python -c "from lammps import lammps; lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none']); lmp.close()"
python -m pip freeze > "${lammps_root}/mofa-lammps-pip-freeze.txt"
module -t list > "${lammps_root}/mofa-lammps-modules.txt" 2>&1
git -C "${lammps_root}" rev-parse HEAD > "${lammps_root}/mofa-lammps-revision.txt"
echo "LAMMPS installed: ${build_dir}/lmp"
echo "Runtime environment: ${LAMMPS_VENV} (keep ${python_prefix} alongside it)."
echo "Next: prepare the MACE model as described in polaris-build/instruction.md."
