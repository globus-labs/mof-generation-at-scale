#!/usr/bin/env bash
# Timed streaming-benchmark run of run_parallel_workflow.py on a single cloud VM.
#
# Runs one stream backend for a fixed wall-clock duration with a large
# --simulation-budget (so it never finishes early), then stops it gracefully so
# the always-on DiasporaQueues benchmark traces flush. For octopus it also
# cleans up the run's Kafka topics afterward.
#
# Usage:
#   bin/run-bench-cloud-vm.sh {mofka|octopus|files} [duration_sec] [out_dir]
#
# Defaults: duration 3600s (60 min), out_dir run/bench-<backend>-<timestamp>.
# Summarize afterward with:  bin/analyze-bench.py <out_dir>/benchmark
#
# Prereqs are the same as bin/run-cloud-vm.sh (see envs/chameleon-stream.md §9):
# the mofa-stream conda env, MOFA_LAMMPS_BIN (defaults to the ACEsuit lmp at
# /home/cc/lammps-mace/build/lmp), the cp2k_shell.ssmp wrapper, and mongod.
set -uo pipefail

BACKEND="${1:-}"
DURATION="${2:-3600}"
case "$BACKEND" in
    mofka|octopus|files) ;;
    *) echo "usage: $0 {mofka|octopus|files} [duration_sec] [out_dir]" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
TS="$(date +%d%b%y-%H%M%S)"
OUTDIR="${3:-$REPO/run/bench-$BACKEND-$TS}"
mkdir -p "$OUTDIR/benchmark"
BENCH_BASE="$OUTDIR/benchmark/diaspora-trace.log"

# Force the mofa-stream env prefix. We deliberately do NOT honor an inherited
# CONDA_PREFIX (a base-conda one leaks in from non-interactive shells and would
# hide the cp2k_shell.ssmp wrapper); override the env location with MOFA_ENV_PREFIX.
export CONDA_PREFIX="${MOFA_ENV_PREFIX:-$HOME/conda-envs/mofa-stream}"
if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "mofa-stream env not found at CONDA_PREFIX=$CONDA_PREFIX (set MOFA_ENV_PREFIX)." >&2
    exit 1
fi
export PATH="$CONDA_PREFIX/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MOFA_CP2K_BIN="${MOFA_CP2K_BIN:-cp2k_shell.ssmp}"
export MOFA_LAMMPS_BIN="${MOFA_LAMMPS_BIN:-/home/cc/lammps-mace/build/lmp}"

command -v mongod >/dev/null || { echo "mongod not on PATH (apt install mongodb-org)" >&2; exit 1; }
command -v "${MOFA_CP2K_BIN%% *}" >/dev/null || {
    echo "CP2K command '${MOFA_CP2K_BIN%% *}' not on PATH — run bin/install-cp2k-shell-wrapper.sh" >&2; exit 1; }

PYTHON="$CONDA_PREFIX/bin/python"
PREFIX="$("$PYTHON" -c 'import secrets;print("mofa_"+secrets.token_hex(3))')"
echo "$PREFIX" > "$OUTDIR/queue-prefix.txt"

MOFKA_ARGS=()
if [[ "$BACKEND" == "mofka" ]]; then
    GROUP_FILE="$("$PYTHON" bin/start-bedrock.py "$OUTDIR/mofka")"
    MOFKA_ARGS=(--mofka-group-file "$GROUP_FILE")
    # libyokan-server.so dlopens lmdb/leveldb without declaring them in DT_NEEDED.
    export LD_PRELOAD="$CONDA_PREFIX/lib/liblmdb.so:$CONDA_PREFIX/lib/libleveldb.so${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[bench] backend=$BACKEND duration=${DURATION}s prefix=$PREFIX outdir=$OUTDIR"
"$PYTHON" -u run_parallel_workflow.py \
    --node-path input-files/zn-paddle-pillar/node.json \
    --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
    --generator-config-path models/geom-300k/config-tf32-a100.yaml \
    --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
    --retrain-freq 2 --num-epochs 2 \
    --num-samples 8 --gen-batch-size 8 --molecule-sizes 8 10 12 \
    --simulation-budget 100000 --md-timesteps 200 --dft-opt-steps 2 \
    --compute-config configs/cloud-vm.py \
    --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
    --queue-prefix "$PREFIX" \
    --benchmark-file "$BENCH_BASE" \
    --stream-engine "$BACKEND" \
    "${MOFKA_ARGS[@]}" > "$OUTDIR/workflow.out" 2>&1 &
WF_PID=$!
echo "$WF_PID" > "$OUTDIR/workflow.pid"
echo "[bench] workflow pid=$WF_PID; running for ${DURATION}s ..."

for ((i=0; i<DURATION; i++)); do
    kill -0 "$WF_PID" 2>/dev/null || { echo "[bench] workflow exited early at ${i}s"; break; }
    sleep 1
done

# Graceful stop: SIGINT triggers run_parallel_workflow's finally + atexit so the
# benchmark MemoryHandlers flush. Fall back to SIGKILL for anything left over.
if kill -0 "$WF_PID" 2>/dev/null; then
    echo "[bench] sending SIGINT for graceful shutdown ..."
    kill -INT "$WF_PID" 2>/dev/null
    for ((i=0; i<120; i++)); do kill -0 "$WF_PID" 2>/dev/null || break; sleep 1; done
fi
pkill -9 -f "[r]un_parallel_workflow.py" 2>/dev/null
[[ "$BACKEND" == "mofka" ]] && pkill -9 -f "[b]edrock-config-single" 2>/dev/null
pkill -9 -f "[m]ongod.*parallel-" 2>/dev/null
pkill -9 -f "[p]arsl" 2>/dev/null
pkill -9 -f "process_worker_pool" 2>/dev/null
# Compute binaries that Parsl workers spawn get reparented to PID 1 and keep
# burning cores when their worker is SIGKILLed — clean them up explicitly.
pkill -9 -x lmp 2>/dev/null
pkill -9 -x cp2k.ssmp 2>/dev/null
pkill -9 -x cp2k.psmp 2>/dev/null
pkill -9 -f "monitor_utilization" 2>/dev/null
sleep 2

if [[ "$BACKEND" == "octopus" ]]; then
    echo "[bench] cleaning up octopus topics for prefix=$PREFIX"
    "$PYTHON" tests/smoke_octopus.py --prefix "$PREFIX" --no-rotate-keys >> "$OUTDIR/cleanup.out" 2>&1 || true
fi

echo "[bench] done. traces under $OUTDIR/benchmark/"
"$PYTHON" bin/analyze-bench.py "$OUTDIR/benchmark" | tee "$OUTDIR/bench-summary.txt"
