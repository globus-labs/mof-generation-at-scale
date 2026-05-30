#!/usr/bin/env bash
# Run run_parallel_workflow.py on a single cloud VM (no PBS/Slurm).
#
# Usage:
#   bin/run-cloud-vm.sh mofka     # uses bin/start-bedrock.py
#   bin/run-cloud-vm.sh octopus   # uses cached Globus tokens (see §5)
#   bin/run-cloud-vm.sh files     # local-file backend, no daemon
#
# Trimmed budget so a CPU-only end-to-end pass fits in ~30 min on a typical
# 8-core cloud VM. Bump --simulation-budget and --md-timesteps later for
# real evaluation runs.

set -euo pipefail
BACKEND="${1:-files}"
case "$BACKEND" in
    mofka|octopus|files) ;;
    *) echo "usage: $0 {mofka|octopus|files}" >&2; exit 2 ;;
esac

export CONDA_PREFIX="${CONDA_PREFIX:-$HOME/conda-envs/mofa-stream}"
if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "Activate the mofa-stream env first (CONDA_PREFIX=$CONDA_PREFIX has no python)." >&2
    exit 1
fi
# Make the env's CLI entry points (monitor_utilization, mongod from conda's
# cp2k.ssmp, etc.) resolvable for subprocess.Popen() calls inside the workflow.
export PATH="$CONDA_PREFIX/bin:$PATH"

# Cap BLAS/thread pools — keeps numpy from starving librdkafka's broker
# threads ("Unable to create broker thread"). See §0 of envs/chameleon-stream.md.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# CP2K via ASE needs a command whose name contains "cp2k_shell" (ASE asserts
# that) and CP2K_DATA_DIR set — neither true for conda-forge's bare cp2k.ssmp.
# bin/install-cp2k-shell-wrapper.sh drops a cp2k_shell.ssmp wrapper that fixes
# both; default MOFA_CP2K_BIN at it. See §9 of envs/chameleon-stream.md.
export MOFA_CP2K_BIN="${MOFA_CP2K_BIN:-cp2k_shell.ssmp}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Backend-specific setup.
MOFKA_ARGS=()
if [[ "$BACKEND" == "mofka" ]]; then
    MOFKA_DIR="${MOFKA_DIR:-/tmp/mofka-run}"
    GROUP_FILE="$(python bin/start-bedrock.py "$MOFKA_DIR")"
    MOFKA_ARGS=(--mofka-group-file "$GROUP_FILE")
    # libyokan-server.so dlopens lmdb/leveldb without declaring them in DT_NEEDED.
    export LD_PRELOAD="$CONDA_PREFIX/lib/liblmdb.so:$CONDA_PREFIX/lib/libleveldb.so${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Sanity checks for shared services. The workflow Popen's mongod itself
# (see run_parallel_workflow.py). ProxyStore-on-Redis is disabled by default,
# so no redis-server is needed.
command -v mongod >/dev/null || { echo "mongod not on PATH (apt install mongodb-org)" >&2; exit 1; }
command -v "${MOFA_CP2K_BIN%% *}" >/dev/null || {
    echo "CP2K command '${MOFA_CP2K_BIN%% *}' not on PATH — run bin/install-cp2k-shell-wrapper.sh" >&2
    exit 1
}

PYTHON="$CONDA_PREFIX/bin/python"

exec "$PYTHON" run_parallel_workflow.py \
    --node-path input-files/zn-paddle-pillar/node.json \
    --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
    --generator-config-path models/geom-300k/config-tf32-a100.yaml \
    --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
    --retrain-freq 1 \
    --num-epochs 1 \
    --num-samples 4 \
    --gen-batch-size 4 \
    --molecule-sizes 8 12 \
    --simulation-budget 1 \
    --md-timesteps 50 \
    --dft-opt-steps 1 \
    --compute-config configs/cloud-vm.py \
    --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
    --stream-engine "$BACKEND" \
    "${MOFKA_ARGS[@]}"
