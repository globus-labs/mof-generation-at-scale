#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N mofa-smoke
#PBS -A ChemGraph

set -euo pipefail

cd "${PBS_O_WORKDIR:?Submit this script from the MOFA repository root}"
repo_root=$PWD
echo "Repository: ${repo_root}"
export MOFA_ENV="${repo_root}/mofa_env_py312"
# Runtime for the MOFA workflow. External simulation launchers load their
# matching modules and, for LAMMPS, Python environment in child processes.
source "${repo_root}/bin/activate-mofa-polaris.sh"
python "${repo_root}/bin/check-polaris.py" --gpu --external
mace_model="${repo_root}/input-files/mace/mace-mp0_medium-mliap_lammps.pt"

redis_log="${repo_root}/redis-${PBS_JOBID}.log"
redis_dir="${repo_root}/.runtime-cache/${PBS_JOBID}/redis"
mkdir -p "${redis_dir}"
redis-server \
    --bind 127.0.0.1 \
    --dir "${redis_dir}" \
    --save "" \
    --appendonly no \
    --protected-mode no \
    --logfile "${redis_log}" &
redis_pid=$!

cleanup() {
    kill "${redis_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 30); do
    if redis-cli -h 127.0.0.1 ping 2>/dev/null | grep -q PONG; then
        break
    fi
    if ! kill -0 "${redis_pid}" 2>/dev/null; then
        echo "Redis exited during startup. See ${redis_log}." >&2
        exit 2
    fi
    sleep 1
done
redis-cli -h 127.0.0.1 ping | grep -q PONG

python run_parallel_workflow.py \
    --node-path input-files/zn-paddle-pillar/node.json \
    --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
    --generator-config-path models/geom-300k/config-tf32-a100.yaml \
    --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
    --compute-config configs/polaris/local-smoke.py \
    --mace-model-path "${mace_model}" \
    --redis-host 127.0.0.1 \
    --molecule-sizes 9 12 \
    --num-samples 8 \
    --gen-batch-size 8 \
    --minimum-ligand-pool 2 \
    --simulation-budget "${MOFA_SIMULATION_BUDGET:-8}" \
    --retrain-freq 1000 \
    --num-epochs 1 \
    --md-timesteps 100 \
    --md-snapshots-freq 50 \
    --dft-opt-steps 1 \
    --raspa-timesteps 100
