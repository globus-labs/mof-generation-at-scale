#!/bin/bash
# Runtime shared by the external CP2K and LAMMPS builds (September 2026).
if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
fi
module reset
module use /soft/modulefiles
if module -t list 2>&1 | grep -q '^PrgEnv-nvidia/'; then
    module swap PrgEnv-nvidia PrgEnv-gnu
elif ! module -t list 2>&1 | grep -q '^PrgEnv-gnu/'; then
    module load PrgEnv-gnu
fi
module load gcc-native/14
module unload cray-libsci cray-fftw 2>/dev/null || true
module load cray-libsci cray-fftw
module load cuda
module load craype-accel-nvidia80
module unload cuda
module load cudatoolkit-standalone/13.0.1
module unload darshan
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MAX_THREAD_SAFETY=multiple
unset PYTHONHOME PYTHONPATH
