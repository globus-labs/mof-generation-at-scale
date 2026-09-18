#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N cp2k-build
#PBS -A ChemGraph
#PBS -j oe

# Submit from the MOFA repository root; see instruction.md for source checkout.
set -euo pipefail
case "${1:-}" in
    -h|--help)
        cat <<'HELP'
Build the patched external CP2K 2025.2 checkout for Polaris A100 GPUs.
From the MOFA repository root:
  qsub -v CP2K_ROOT=/absolute/path/to/cp2k polaris-build/build-cp2k-polaris.sh
Required: CP2K_ROOT, a separate checkout at the revision in instruction.md.
Optional: CP2K_BUILD_JOBS (8), CP2K_CLEAN_TOOLCHAIN (0; set 1 to rebuild dependencies).
Existing CP2K outputs are moved to timestamped backups. Sources are not cloned.
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
if [[ "${CP2K_ROOT:-}" != /* ]]; then
    echo "Set CP2K_ROOT to an absolute external source-checkout path." >&2
    exit 2
fi
cp2k_root=$(realpath -e -- "${CP2K_ROOT}")
case "${cp2k_root}/" in
    "${repo_root}/"*) echo "CP2K_ROOT must be outside the MOFA checkout." >&2; exit 2 ;;
esac
expected_revision=34658f03b9867a3335ace1b4af7b7736c33523bb
if [[ ! -f "${cp2k_root}/CMakeLists.txt" ||
      "$(git -C "${cp2k_root}" rev-parse HEAD)" != "${expected_revision}" ]]; then
    echo "Expected patched CP2K revision ${expected_revision}; see polaris-build/instruction.md." >&2
    exit 2
fi
build_jobs="${CP2K_BUILD_JOBS:-8}"
if [[ ! "${build_jobs}" =~ ^[1-9][0-9]*$ || ! "${CP2K_CLEAN_TOOLCHAIN:-0}" =~ ^[01]$ ]]; then
    echo "CP2K_BUILD_JOBS must be positive; CP2K_CLEAN_TOOLCHAIN must be 0 or 1." >&2
    exit 2
fi

# Prevent concurrent jobs from moving or rebuilding the same directories.
lock_file="${cp2k_root}/.build-cp2k.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: Another CP2K build is active for ${cp2k_root}."
    exit 3
fi

cd "${cp2k_root}"
echo "PBS job: ${PBS_JOBID}"
echo "CP2K source: ${cp2k_root}"

echo "========== Configuring Polaris modules =========="
# Build and runtime wrappers deliberately share the compiler/CUDA module stack.
source "${repo_root}/bin/polaris-simulation-env.sh"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L${CUDA_HOME}/lib64 -lcudart ${LDFLAGS:-}"

module list

echo
echo "========== Compiler environment =========="
echo "PE_ENV=${PE_ENV:-}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CRAY_ACCEL_TARGET=${CRAY_ACCEL_TARGET:-}"
cc --version | head -n 1
CC --version | head -n 1
ftn --version | head -n 1
nvcc --version | tail -n 1

if ! nvcc --version | grep -q 'release 13\.0'; then
    echo "ERROR: CP2K must be built with CUDA 13.0.x on Polaris"
    echo "       (cray-mpich 9.1.0 GTL requires libcudart.so.13)."
    nvcc --version
    exit 1
fi
if [[ "${CRAY_ACCEL_TARGET:-}" != "nvidia80" ]]; then
    echo "ERROR: Expected the Polaris A100 target nvidia80."
    exit 1
fi
if ! cc --cray-print-opts=libs | grep -q -- '-lmpi_gtl_cuda'; then
    echo "ERROR: The Cray compiler wrapper is not configured for CUDA-aware MPI."
    echo "cc --cray-print-opts=libs returned:"
    cc --cray-print-opts=libs
    exit 1
fi

# Preflight: actually link and compile something before spending ~25 minutes on
# dependencies. The version checks above only compare strings; these catch the
# real failure mode, which is a module set that looks right but cannot link.
#
# This exists because ALCF can change a module default out from under the
# script. When cudatoolkit-standalone defaulted to 12.2.2, every link in the
# toolchain failed (cray-mpich's libmpi_gtl_cuda needs libcudart.so.13) and the
# first symptom was 28 undefined symbols inside OpenBLAS's getarch probe,
# twelve minutes in and three layers from the cause.
echo
echo "========== Preflight: toolchain sanity =========="
preflight_dir=$(mktemp -d)
trap 'rm -rf "${preflight_dir}"' EXIT

# 1. Can we link at all? craype-accel-nvidia80 injects -lmpi_gtl_cuda into
#    every link line, so this fails if the CUDA runtime does not match.
echo 'int main(void) { return 0; }' > "${preflight_dir}/link.c"
if ! cc "${preflight_dir}/link.c" -o "${preflight_dir}/link.out" \
        > "${preflight_dir}/link.log" 2>&1; then
    echo "ERROR: Preflight link failed. The loaded CUDA runtime is probably"
    echo "       incompatible with the cray-mpich CUDA GTL."
    echo "       CUDA_HOME=${CUDA_HOME}"
    echo "       libmpi_gtl_cuda requires:"
    objdump -p "$(cc --cray-print-opts=libs | tr ' ' '\n' |
        sed -n 's/^-L//p' | while read -r d; do
            [[ -e "${d}/libmpi_gtl_cuda.so" ]] && echo "${d}/libmpi_gtl_cuda.so" && break
        done)" 2>/dev/null | grep 'NEEDED.*cudart' || true
    echo "       CUDA_HOME provides:"
    ls "${CUDA_HOME}"/lib64/libcudart.so.* 2>/dev/null || true
    echo "--- link output ---"
    cat "${preflight_dir}/link.log"
    exit 1
fi

# 2. Can nvcc parse the loaded gcc's headers, and emit code for this GPU?
#    CUDA 12.2 chokes on gcc-native/14 headers (_Float128, 0.0bf16), and CUDA
#    13 dropped the Pascal arch that some dependencies still default to.
cat > "${preflight_dir}/probe.cu" << 'PREFLIGHT_EOF'
#include <math.h>
__global__ void probe_kernel(void) {}
int main(void) { return 0; }
PREFLIGHT_EOF
if ! nvcc -arch=sm_80 -std=c++14 -allow-unsupported-compiler \
        -c "${preflight_dir}/probe.cu" -o "${preflight_dir}/probe.o" \
        > "${preflight_dir}/nvcc.log" 2>&1; then
    echo "ERROR: Preflight nvcc compile failed for sm_80."
    echo "       nvcc: $(nvcc --version | tail -n 2 | head -n 1)"
    echo "       gcc:  $(cc --version | head -n 1)"
    echo "--- nvcc output ---"
    cat "${preflight_dir}/nvcc.log"
    exit 1
fi

rm -rf "${preflight_dir}"
trap - EXIT
echo "Preflight OK: link succeeds and nvcc emits sm_80."

# COSMA and COSTA require the LibSci installation prefix. Derive the active
# GNU/GCC 14 path from the Cray compiler wrapper instead of hard-coding it.
libsci_libdir=$(
    cc --cray-print-opts=libs |
        tr ' ' '\n' |
        sed -n 's/^-L//p' |
        grep '/libsci/' |
        head -n 1 || true
)
if [[ -z "${libsci_libdir}" || ! -d "${libsci_libdir}" ]]; then
    echo "ERROR: Could not determine the active Cray LibSci directory."
    exit 1
fi
case "${libsci_libdir}" in
    */lib|*/lib64)
        export CRAY_LIBSCI_PREFIX_DIR="${libsci_libdir%/*}"
        ;;
    *)
        echo "ERROR: Unexpected LibSci library path: ${libsci_libdir}"
        exit 1
        ;;
esac
export CRAY_PE_LIBSCI_PREFIX_DIR="${CRAY_LIBSCI_PREFIX_DIR}"
echo "CRAY_LIBSCI_PREFIX_DIR=${CRAY_LIBSCI_PREFIX_DIR}"

echo
echo "========== Preserving previous build =========="
# By default only the CP2K object/exe trees are moved aside. The toolchain
# build and install trees are left in place so install_cp2k_toolchain.sh can
# reuse the per-package install_successful lock files and skip dependencies it
# already built: a full rebuild is ~25 minutes (libxc alone is ~12) and several
# GB, which is a steep price for re-running after a failure in a late stage.
#
# Set CP2K_CLEAN_TOOLCHAIN=1 to force a from-scratch dependency rebuild. Do
# that whenever the compiler or CUDA version changes, since the lock files do
# not track the toolchain environment and stale objects would be reused.
backup_tag="prev-$(date -u +%Y%m%dT%H%M%SZ)-${PBS_JOBID}"
backup_paths=(
    obj/local
    lib/local
    exe/local
    obj/local_cuda
    lib/local_cuda
    exe/local_cuda
)
if [[ "${CP2K_CLEAN_TOOLCHAIN:-0}" == "1" ]]; then
    echo "CP2K_CLEAN_TOOLCHAIN=1: dependencies will be rebuilt from scratch."
    backup_paths+=(tools/toolchain/install tools/toolchain/build)
else
    echo "Reusing existing toolchain dependencies where possible."
    echo "Set CP2K_CLEAN_TOOLCHAIN=1 to force a full dependency rebuild."
fi
for old_path in "${backup_paths[@]}"; do
    if [[ -e "${old_path}" ]]; then
        echo "Moving ${old_path} to ${old_path}.${backup_tag}"
        mv "${old_path}" "${old_path}.${backup_tag}"
    fi
done
if [[ -f tools/toolchain/install.log ]]; then
    mv tools/toolchain/install.log \
        "tools/toolchain/install.log.${backup_tag}"
fi

echo
echo "========== Installing CP2K dependencies =========="
cd tools/toolchain
./install_cp2k_toolchain.sh \
    --enable-cray=yes \
    --enable-cuda \
    --math-mode=cray \
    --gpu-ver=A100 \
    --target-cpu=znver3 \
    --mpi-mode=mpich \
    --with-elpa=install \
    --with-sirius=no \
    --with-openblas=no \
    -j "${build_jobs}" 2>&1 | tee install.log

cp install/arch/* ../../arch/
cd "${cp2k_root}"

echo
echo "========== Building CP2K =========="
# The generated toolchain setup appends to CP_* variables without first
# defining them. Temporarily disable nounset while importing it, then restore
# the build driver's strict shell settings.
set +u
source ./tools/toolchain/install/setup
set -u
export MPICH_GPU_SUPPORT_ENABLED=1

if ! nvcc --version | grep -q 'release 13\.0'; then
    echo "ERROR: The generated toolchain changed the active CUDA version."
    nvcc --version
    exit 1
fi

make -j "${build_jobs}" ARCH=local VERSION="ssmp psmp"
make -j "${build_jobs}" ARCH=local_cuda VERSION="ssmp psmp"

echo
echo "========== Verifying executables =========="
executables=(
    exe/local/cp2k.ssmp
    exe/local/cp2k.psmp
    exe/local/cp2k_shell.ssmp
    exe/local/cp2k_shell.psmp
    exe/local_cuda/cp2k.ssmp
    exe/local_cuda/cp2k.psmp
    exe/local_cuda/cp2k_shell.ssmp
    exe/local_cuda/cp2k_shell.psmp
)
for exe in "${executables[@]}"; do
    if [[ ! -x "${exe}" ]]; then
        echo "ERROR: Missing executable: ${exe}"
        exit 1
    fi
    echo "--- ${exe} ---"
    "./${exe}" --version | sed -n '1,8p'
done

echo
echo "========== Verifying CUDA runtime and MPI GTL =========="
for exe in exe/local_cuda/cp2k.psmp exe/local_cuda/cp2k_shell.psmp; do
    dynamic_section=$(readelf -d "${exe}")
    if [[ "${dynamic_section}" != *"cuda-13.0.1"* ]]; then
        echo "ERROR: ${exe} does not contain the CUDA 13.0.1 runpath."
        exit 1
    fi
    if [[ "${dynamic_section}" == *"cuda-12."* ]]; then
        echo "ERROR: ${exe} still contains a CUDA 12.x runpath."
        exit 1
    fi

    ldd_output=$(ldd "${exe}")
    if [[ "${ldd_output}" == *"not found"* ]]; then
        echo "ERROR: ${exe} has unresolved shared libraries:"
        echo "${ldd_output}"
        exit 1
    fi
    if [[ "${ldd_output}" != *"libmpi_gtl_cuda"* ]]; then
        echo "ERROR: ${exe} is not linked to the Cray MPI CUDA GTL."
        exit 1
    fi
    if ! grep 'libnvrtc' <<<"${ldd_output}" | grep -q 'cuda-13.0.1'; then
        echo "ERROR: ${exe} does not resolve NVRTC from CUDA 13.0.1."
        echo "${ldd_output}" | grep -E 'cuda|nvrtc|gtl' || true
        exit 1
    fi

    libsci_count=0
    while IFS= read -r libsci_path; do
        [[ -n "${libsci_path}" ]] || continue
        libsci_count=$((libsci_count + 1))
        resolved_libsci=$(readlink -f "${libsci_path}")
        case "${resolved_libsci}" in
            "${CRAY_LIBSCI_PREFIX_DIR}"/*) ;;
            *)
                echo "ERROR: ${exe} resolves LibSci outside the active installation:"
                echo "  ${resolved_libsci}"
                exit 1
                ;;
        esac
    done < <(awk '/libsci/ {print $3}' <<<"${ldd_output}")
    if (( libsci_count == 0 )); then
        echo "ERROR: ${exe} has no resolved Cray LibSci dependency."
        exit 1
    fi

    echo "--- ${exe}: CUDA, NVRTC, MPI GTL, and LibSci ---"
    echo "${ldd_output}" | grep -Ei 'cuda|nvrtc|gtl|libsci'
done

echo
git -C "${cp2k_root}" rev-parse HEAD > "${cp2k_root}/mofa-cp2k-revision.txt"
module -t list > "${cp2k_root}/mofa-cp2k-modules.txt" 2>&1
echo "CP2K installed: ${cp2k_root}/exe/local_cuda"
echo "Use CP2K_ROOT=${cp2k_root} and CP2K_DATA_DIR=${cp2k_root}/data for MOFA."
echo "Previous build suffix: ${backup_tag}"
