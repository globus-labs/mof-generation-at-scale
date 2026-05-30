# Chameleon Diaspora Stream env

> branch `diaspora-debug` · python 3.12 · last verified **2026-05-30** on a
> Chameleon Ubuntu 24.04 VM (glibc 2.39, 48-core CPU, no GPU).

Build a conda env for the MOFA workflow plus the
[mochi-hpc](https://github.com/mochi-hpc) **mofka** and **octopus** backends
used by `mofa.diaspora.DiasporaQueues`, then run `run_parallel_workflow.py`
end-to-end and benchmark the streaming layer.

Primary target: a single **Chameleon Cloud Ubuntu 24.04 VM** treated as an HPC
login node (no PBS/Slurm; the workflow runs as a plain process). The mochi
prebuilt binaries need glibc ≥ 2.32, so Chameleon and a Polaris login node
(glibc 2.38) run them natively without a container. Polaris differences are
flagged inline.

The mochi packages are not on conda-forge; they ship as per-release tarballs at
[mochi-hpc/mochi-conda-packages](https://github.com/mochi-hpc/mochi-conda-packages/releases).
This guide pins **`TAG=2026-05-30`** (mofka 0.9.0 / yokan 0.9.2 / bedrock 0.16.1),
verified end-to-end here; a newer release works too as long as yokan and
`diaspora-stream-octopus` still agree on the libuuid / util-linux pin (see
[Troubleshooting](#troubleshooting)).

---

## 1. One-time host + env setup

### 1a. Host prerequisites

```bash
# miniconda (first time only)
curl -L -o /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

# accept the Anaconda ToS (conda >=24 aborts the solve otherwise)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# system packages: mongod (workflow spawns it) + libXrender (openbabel dlopens it)
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | sudo gpg -o /etc/apt/keyrings/mongodb-server-8.0.gpg --dearmor
echo "deb [signed-by=/etc/apt/keyrings/mongodb-server-8.0.gpg] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" \
    | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
sudo apt update && sudo apt install -y mongodb-org libxrender1
```

**Always export the BLAS/thread caps** in any shell that runs this guide —
otherwise numpy/MKL starve librdkafka's broker threads (`Unable to create
broker thread`):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

> **Polaris:** login nodes enforce a `pids.max=128` cgroup, so use plain `conda`
> (not mamba) with `export CONDA_FETCH_THREADS=1 CONDA_EXTRACT_THREADS=1`, and
> load conda via `module use /soft/modulefiles/ && module load conda` (ToS
> pre-accepted there).

### 1b. Stage the mochi channel and create the env

Conda does not expand vars in channel URLs, so the channel must live at the
absolute path hard-coded in `envs/environment-chameleon-stream.yml`
(`/home/cc/mochi-conda-packages`). On a host where `$USER != cc`, symlink it or
sed-replace the URL first.

```bash
mkdir -p /home/cc/mochi-conda-packages && cd /home/cc/mochi-conda-packages
TAG=2026-05-30
curl -L "https://github.com/mochi-hpc/mochi-conda-packages/releases/download/$TAG/mochi-conda-channel-$TAG-linux-64-py3.12.tar.gz" \
    | tar -xz --strip-components=1     # expect: linux-64/  noarch/  channeldata.json

cd ~/mof-generation-at-scale
conda env create --file envs/environment-chameleon-stream.yml --prefix ~/conda-envs/mofa-stream
conda activate ~/conda-envs/mofa-stream
pip install ConfigSpace                # only for `mofkactl config generate`

python tests/smoke_stream_env.py       # expect: 8/8 checks passed
```

`smoke_stream_env.py` checks every import the stream workflow needs
(`diaspora_stream.api`, `mochi.mofka`, `liboctopus.so`, `colmena`, `torch`,
`ase`, `mofa.diaspora`). Any failure makes the env unusable.

### 1c. LAMMPS with `pair_style mace`

`CloudVMConfig.lammps_pkg='ml-mace'` needs `pair_style mace` from the
[ACEsuit/lammps](https://github.com/ACEsuit/lammps) fork to load the
libtorch-format MACE model. Build once:

```bash
# CPU libtorch, cxx11-abi (NOT conda's CUDA libtorch)
curl -L -o /tmp/libtorch.zip \
    https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
unzip -q /tmp/libtorch.zip -d /home/cc/          # -> /home/cc/libtorch

git clone -b mace https://github.com/ACEsuit/lammps.git /home/cc/lammps-mace
mkdir -p /home/cc/lammps-mace/build && cd /home/cc/lammps-mace/build
cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH=/home/cc/libtorch -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include \
    -DPKG_ML-MACE=on -DPKG_PYTHON=off
cmake --build . -j$(nproc)
./lmp -h | grep -i mace                          # expect "mace" in pair_style list

# generate the libtorch model pair_style mace consumes; move into input-files/mace/
~/conda-envs/mofa-stream/bin/mace_create_lammps_model --format libtorch --dtype float32 \
    ~/mof-generation-at-scale/input-files/mace/mace-mp0_medium.model

export MOFA_LAMMPS_BIN=/home/cc/lammps-mace/build/lmp   # persist in ~/.bashrc
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is needed for the fork's old cmake policy
syntax; `-DMKL_INCLUDE_DIR` lets the FFT pkg find conda's `mkl_dfti.h`.

### 1d. CP2K shell wrapper

ASE's CP2K calculator asserts the launch command contains `cp2k_shell`, and
CP2K needs `CP2K_DATA_DIR` — neither true for conda-forge's bare `cp2k.ssmp`
(pinned `*nompi*` → serial cp2k 2024.2; the env yml does this because cp2k
2026.1 is MPI-only). The wrapper fixes both and raises CP2K's OpenMP threads:

```bash
conda activate ~/conda-envs/mofa-stream
bin/install-cp2k-shell-wrapper.sh        # creates $CONDA_PREFIX/bin/cp2k_shell.ssmp
```

### 1e. Octopus tokens (octopus backend only)

Octopus authenticates via Globus + AWS MSK. Bootstrap once (interactive Globus
login, cached in `~/.diaspora/storage.db`):

```bash
~/conda-envs/mofa-stream/bin/python tests/smoke_octopus.py
```

Subsequent runs are non-interactive. The same script also wipes stale topics
(see [Cleanup](#5-cleanup)).

---

## 2. Smoke tests

### 2a. mofka bedrock daemon

mofka needs a local bedrock daemon. `bin/start-bedrock.py` writes a minimal
config, uses `ofi+tcp://127.0.0.1` (Parsl-fork-safe; `na+sm` is Yama-blocked and
deadlocks forked workers — see
[ADR-0001](../docs/adr/0001-mofka-bedrock-transport.md)), patches the yokan
backend to in-memory `map` (`BEDROCK_YOKAN_MAP=1`, default), and `LD_PRELOAD`s
the lmdb/leveldb that `libyokan-server.so` dlopens:

```bash
conda activate ~/conda-envs/mofa-stream
python bin/start-bedrock.py /tmp/mofka-test      # stdout: /tmp/mofka-test/mofka.flock.json
# teardown:  pkill -f bedrock-config-single.json ; rm -rf /tmp/mofka-test/data_* /tmp/mofka-test/metadata
```

### 2b. Colmena round-trip — `mofa/diaspora.py`

Round-trips `send_inputs → get_task → send_result → get_result` over all five
task classes. Expect a final `RESULT: PASS` and exit 0.

```bash
P=~/conda-envs/mofa-stream/lib; PRELOAD="$P/liblmdb.so:$P/libleveldb.so"
PY=~/conda-envs/mofa-stream/bin/python

# mofka (needs bedrock from 2a)
LD_PRELOAD=$PRELOAD LD_LIBRARY_PATH=$P $PY -u mofa/diaspora.py mofka \
    --group-file /tmp/mofka-test/mofka.flock.json

# octopus (needs tokens from 1e); first run vs a fresh topic can take >60 s
$PY -u mofa/diaspora.py octopus
```

Benign noise (not failures): mofka `HG_Core_register() Overwriting RPC callback`
once per topic; octopus `BatchSize ignored by consumer` and `PARTCNT` lines on
topic delete.

---

## 3. Run the full workflow

`bin/run-cloud-vm.sh` drives a short **smoke** run (`--simulation-budget 1`) of
the full pipeline (generation → ligand-validate → assembly → LAMMPS+MACE →
CP2K) for one backend:

```bash
bin/run-cloud-vm.sh mofka     # auto-starts bedrock
bin/run-cloud-vm.sh octopus   # needs tokens (1e)
bin/run-cloud-vm.sh files     # local-file backend, no daemon
```

It sets `PATH`, the BLAS caps, `MOFA_LAMMPS_BIN`, `MOFA_CP2K_BIN=cp2k_shell.ssmp`
(and `LD_PRELOAD`/`LD_LIBRARY_PATH` for mofka), and uses
[configs/cloud-vm.py](../configs/cloud-vm.py) → `CloudVMConfig` (CPU-only). The
workflow Popens its own `mongod`. CP2K on CPU is the long pole — one MOF's SCF
can dominate wall time (a non-converging one grinds to `max_scf=128`); that cost
is independent of the stream backend.

mofka smoke exits rc=0 in ~8 min. octopus is slower (~15 min + CP2K) — Kafka
round-trip latency makes generation→assembly take ~7 min.

For a longer evaluation run, call `run_parallel_workflow.py` directly under the
same env (raise `--simulation-budget`, `--num-samples`, `--md-timesteps`); for a
single CPU VM don't exceed ~`--simulation-budget 4 --md-timesteps 200` (a
`--simulation-budget 8` run deadlocked in the shutdown drain after ~14 h).

---

## 4. Benchmark the streaming layer

`DiasporaQueues` benchmarking is **always on**
([`mofa/diaspora.py`](../mofa/diaspora.py) `log_benchmark`): every timed queue op
is written as one JSON record per line to a per-PID trace file. Default location
is `<run_dir>/benchmark/diaspora-trace-pid<PID>.log`; override the base with
`--benchmark-file`. (`run_parallel_workflow.py` enables the trace
unconditionally — `--benchmark-file` only moves the output.)

```json
{"task_name":"PUSH","engine":"octopus","queue":"mofa_x_requests",
 "msg_size":37566,"start_ns":...,"end_ns":...,"duration_ns":...,
 "pid":318142,"wall_ns":...}
```

| `task_name` | meaning | backends |
|---|---|---|
| `CONNECT` / `DISCONNECT` | driver/dispatcher open & teardown | all |
| `CREATE_TOPIC` / `GET_TOPIC` | first topic open (incl. propagation wait) / cached lookup | files, octopus |
| `PUSH` / `FLUSH` | `producer.push().wait()` / `producer.flush().wait()` | files, octopus |
| `SEND_MSG` | full caller-visible send | all |
| `EVENT_WAIT` / `EVENT_ACK` | `pull().wait()` that returned a msg / `event.acknowledge()` | files, octopus |
| `MOFKA_PUSH` / `MOFKA_EVENT_WAIT` / `MOFKA_EVENT_ACK` | the same broker ops, measured **inside the mofka dispatcher thread** | mofka |
| `GET_MSG` | full caller-visible receive (**includes idle blocking** until a message is available) | all |

- `duration`s use `perf_counter_ns` (monotonic, per-process); `wall_ns`/`pid`
  let traces from the thinker and server processes merge onto one timeline.
- For **mofka** the broker push/pull run on a dedicated dispatcher thread, so
  `SEND_MSG` folds in the inter-thread handoff while `MOFKA_PUSH` is the raw
  broker push. For **octopus/files** `SEND_MSG` = `PUSH` + `FLUSH`.
- `GET_MSG` is *not* a transport metric — the workflow is compute-gated on CPU
  MD/DFT, so a consumer often blocks seconds–minutes for the next task. Use
  `PUSH`/`MOFKA_PUSH`/`EVENT_WAIT`/`EVENT_ACK` for transport latency.

Summarize any trace dir (recurses for `*trace*.log`; tab-separated columns for
spreadsheet paste): `bin/analyze-bench.py <dir> [--by-queue]`.

### 4a. Native (no docker)

`bin/run-bench-cloud-vm.sh` runs one backend for a fixed duration with a large
`--simulation-budget` (so it never exits early), stops it gracefully so traces
flush, cleans up orphaned `lmp`/`cp2k`/octopus-topics, then prints the summary:

```bash
bin/run-bench-cloud-vm.sh mofka   3600     # 1 hr; traces -> run/bench-mofka-<timestamp>/benchmark/
bin/run-bench-cloud-vm.sh octopus 3600
# the runner prints a summary on exit; re-analyze the dir it created (newest match):
bin/analyze-bench.py "$(ls -dt run/bench-mofka-*/benchmark | head -1)"   # per-op latency + throughput
```

A 1-hour run is mostly idle: the pipeline is compute-gated on CPU DFT, so it
streams a burst early then trickles (a slow/non-converging DFT can stall mofka
after ~10 min). Per-op **latency** is stable regardless; longer runs mainly grow
the message-count sample. Don't read the message *throughput* as a transport
number — it tracks DFT speed, not the queue.

### 4b. Containerized (thinker/server split)

The [mofa-mini-app](https://github.com/diaspora-project/mofa-mini-app) runs the
workflow as a thinker + server split per backend. Its `docker-compose.yml`
bind-mounts `./bench-out/<profile>` over the workflow's `run/`, so traces land
on the host by default. Build picks up `diaspora-debug` from the remote
(`--build-arg CACHEBUST=$(date +%s)` re-pulls after a push):

```bash
cd ~/mofa-mini-app
# Rebuild to bake in scripts/start.sh changes (cheap COPY tail). Add
# --build-arg CACHEBUST=$(date +%s) only when re-pulling MOFA after a source push.
docker compose --profile mofka build

# 1-hr run. Container length is WALL-CLOCK — there is no duration flag: you start
# it, wait, then `down`. MOFA_SIM_BUDGET is NOT a timer; it's the simulation count
# that ends the run early, so set it huge to keep the workflow alive for the whole
# hour. MOFA_QUEUE_PREFIX is the shared topic prefix both halves meet on.
MOFA_SIM_BUDGET=100000 MOFA_QUEUE_PREFIX=mofa_bench \
    docker compose --profile mofka up -d
sleep 3600                                   # 1 hr, then stop
docker compose --profile mofka down

~/mof-generation-at-scale/bin/analyze-bench.py ~/mofa-mini-app/bench-out/mofka
```

For the **octopus** profile, seed the container's cached tokens once — either
the interactive bootstrap
(`docker compose --profile octopus run --rm octopus-thinker python tests/smoke_octopus.py`)
or, headless, copy the host's `~/.diaspora/storage.db` into the
`diaspora-storage` volume. Use a fresh `MOFA_QUEUE_PREFIX` per run and clean its
topics afterward (see Cleanup). Container traces may be root-owned; `sudo chown`
`bench-out/` if `analyze-bench.py` can't read them.

**Measured (Chameleon 48-core CPU VM, four 1-hour runs, 2026-05-30):** mofka
broker push p50 **~0.56–0.59 ms** (loopback bedrock) vs octopus **~51–121 ms**
(AWS MSK) — roughly **90–210× slower per send**. Native and container run
identical workflow params (the only intended difference is the thinker/server
split), so the two columns are a controlled A/B.

---

## 5. Cleanup

`bin/run-bench-cloud-vm.sh` does this automatically; for a manual run after
`MOFAThinker completed` (or a hang):

```bash
pkill -9 -f "[r]un_parallel_workflow.py"
pkill -9 -f "[b]edrock-config-single"        # mofka only
pkill -9 -f "[m]ongod.*parallel-"
pkill -9 -f "[p]arsl"
pkill -9 -x lmp ; pkill -9 -x cp2k.ssmp       # Parsl orphans these to PID 1
```

(Use bracketed patterns so a `pkill -f` doesn't match its own shell.)

After an octopus run, delete only that run's topics (its prefix is logged at
start-up as `mofa_` + 6 hex, or you set it via `--queue-prefix`):

```bash
~/conda-envs/mofa-stream/bin/python tests/smoke_octopus.py --prefix <prefix> --no-rotate-keys
```

Do **not** run `smoke_octopus.py` with no `--prefix` — it wipes every topic in
the namespace, including ones other sessions created.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Unable to create broker thread` / `OpenBLAS pthread_create failed` | Export the four `*_NUM_THREADS=1` caps (§1a); `pkill -f diaspora.py` stale runs; wipe stale octopus topics |
| `conda env create` fails: `mochi-yokan/mofka` need `libuuid >=2.42` but `diaspora-stream-octopus` needs `util-linux <2.42` | A release where the two disagree on the libuuid/util-linux pin (seen on `2026-03-30`..`04`). The pinned **`2026-05-30`** rebuilds both against `util-linux 2.42` so they coexist — keep `TAG=2026-05-30` (or a newer release that holds the agreement); if a future tag regresses, pin back to `2026-05-30` |
| `bedrock` exits: `undefined symbol: mdb_env_create` / `leveldb::...` | `libyokan-server.so` doesn't declare lmdb/leveldb in `DT_NEEDED` — `LD_PRELOAD` them (`start-bedrock.py` and `run-*.sh` do this) |
| `Yokan error: Invalid mode` on `create_topic` | upstream lmdb/rocksdb/leveldb backends reject mofka's `put_direct` on Ubuntu's libstdc++ ABI — `BEDROCK_YOKAN_MAP=1` (start-bedrock.py default) rewrites yokan to in-memory `map` |
| bedrock `na+sm`: `Operation not permitted` / Yama; or parsl-forked workers hang | use `ofi+tcp` (start-bedrock.py default); see [ADR-0001](../docs/adr/0001-mofka-bedrock-transport.md) |
| CP2K `AssertionError` (`'cp2k_shell' in command`) or `basis set ... not found in BASIS_MOLOPT` | install + use the `cp2k_shell.ssmp` wrapper (§1d); `run-*.sh` default `MOFA_CP2K_BIN` to it |
| `from openbabel import pybel` aborts: `libXrender.so.1: cannot open` | `sudo apt install -y libxrender1` (§1a) |
| `CondaToSNonInteractiveError` on a fresh miniconda | run the `conda tos accept` lines (§1a) |
| octopus run leaves stale topics → later `broker thread` errors | `smoke_octopus.py --prefix <prefix> --no-rotate-keys` (§5) |
</content>
