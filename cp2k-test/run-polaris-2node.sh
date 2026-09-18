#!/bin/bash -l

# Validate distributed (2-node) CP2K, matching the MOFA config's original
# nodes_per_cp2k=2 launch. Run this from an active Polaris PBS allocation with
# at least two nodes, e.g.:
#   qsub -I -l select=2:system=polaris -l walltime=00:30:00 \
#        -l filesystems=home:eagle -q debug -A ChemGraph
# then run this script from inside the allocation.
#
# Everything except the launch (module stack, env, binary, affinity, success
# check) is identical to run-polaris.sh so this isolates the multi-node variable.
set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Run this script inside a Polaris PBS job."
    exit 2
fi
if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
    echo "PBS_NODEFILE is not available; run inside a PBS allocation."
    exit 2
fi

num_nodes=$(wc -l < "${PBS_NODEFILE}")
if (( num_nodes < 2 )); then
    echo "This test needs at least 2 nodes; allocation has ${num_nodes}."
    exit 2
fi

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${test_dir}/.." && pwd -P)
source "${repo_root}/bin/polaris-paths.sh"
cp2k_exe="${repo_root}/bin/run-cp2k-polaris.sh"
affinity_script="${repo_root}/bin/set-affinity-gpu-polaris.sh"

if [[ ! -x "${cp2k_exe}" ]]; then
    echo "Missing CP2K executable: ${cp2k_exe}"
    exit 2
fi
if [[ ! -f "${test_dir}/cp2k.inp" ]]; then
    echo "Missing CP2K input: ${test_dir}/cp2k.inp"
    exit 2
fi

# Each rank loads the external build's matching modules through the wrapper.
export OMP_NUM_THREADS=8

run_dir="${test_dir}/run-2node-${PBS_JOBID}"
if [[ -e "${run_dir}" ]]; then
    echo "Run directory already exists: ${run_dir}"
    exit 2
fi
mkdir "${run_dir}"
cp "${test_dir}/cp2k.inp" "${run_dir}/"
cd "${run_dir}"

# Build a 2-node hostfile the same way MOFA batches cp2k-hostfiles/local_hostfile.NNNN.
hostfile="${run_dir}/local_hostfile"
head -n 2 "${PBS_NODEFILE}" > "${hostfile}"

echo "CP2K executable: ${cp2k_exe}"
echo "Run directory: ${run_dir}"
echo "Hostfile:"
cat "${hostfile}"

env MPICH_OFI_CXI_PID_BASE=5 \
    mpiexec -n 8 --ppn 4 --cpu-bind depth --depth 8 \
    -env OMP_NUM_THREADS=8 -env CP2K_BINARY=cp2k.psmp --hostfile "${hostfile}" \
    "${affinity_script}" "${cp2k_exe}" \
    -i cp2k.inp -o cp2k.out \
    > launcher.stdout 2> launcher.stderr

if ! grep -q 'PROGRAM ENDED AT' cp2k.out; then
    echo "CP2K exited without its successful-termination marker."
    tail -n 100 cp2k.out
    exit 3
fi

echo "CP2K simulation completed successfully: ${run_dir}/cp2k.out"
