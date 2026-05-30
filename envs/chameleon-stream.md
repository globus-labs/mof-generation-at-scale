# Chameleon Diaspora Stream env

> branch: `diaspora-debug` · date: 2026-05-27 · python: 3.12
>
> **Last verified 2026-05-27** (Chameleon Ubuntu 24.04 VM): §3 env 8/8 ·
> §6 round-trip mofka + octopus PASS · §9 smoke **mofka PASS** (rc=0, ~8 min) ·
> §9 smoke **octopus** runs the full pipeline (generation → MD → CP2K) after
> the parsl idle-worker fix (§8); reaching `MOFAThinker completed` is then
> gated on MOF-dependent CP2K SCF wall time. CP2K works via the §9
> `cp2k_shell.ssmp` wrapper.
>
> **Re-verified 2026-05-29** (env + image recreated): §3 8/8. Native mofka &
> octopus exercise generation → MD (LAMMPS+MACE) → CP2K (CP2K SCF is the long
> pole; runs stopped at ~15 min). Both docker compose profiles boot and
> round-trip the thinker/server split (mofka loopback-bedrock + shared netns;
> octopus dual-container on AWS MSK). One drift fixed: conda-forge `cp2k` is now
> MPI-only, so §2 pins `=*=*nompi*` → serial cp2k 2024.2 (ships `cp2k.ssmp`).

Build a conda env for the MOFA workflow plus the
[mochi-hpc](https://github.com/mochi-hpc) **mofka** and **octopus** backends
used by `mofa.diaspora.DiasporaQueues`. Primary target: a **Chameleon Cloud
Ubuntu 24.04 VM** (glibc 2.39) treated as an "HPC login node" — no
PBS/Slurm, `run_parallel_workflow.py` runs as a plain process. The mochi
prebuilt binaries need glibc ≥ 2.32, so both Chameleon and a Polaris login
node (glibc 2.38) run them natively without a container; Polaris-specific
differences are flagged inline.

The mochi packages are not on conda-forge. They ship as a per-release tarball
at [mochi-hpc/mochi-conda-packages/releases](https://github.com/mochi-hpc/mochi-conda-packages/releases).
The env yml ([environment-chameleon-stream.yml](environment-chameleon-stream.yml))
does not pin `mofka`, `diaspora-stream-api`, or `diaspora-stream-octopus`, so
whatever the staged channel contains is what you get. The snippet in §1 pins
the channel to **`TAG=2026-03-30`** — the last release whose yokan and
diaspora-stream-octopus agree on a libuuid pin (see §8 for the conflict, which
is still present in the latest release `2026-05-12`, so the pin stands).

---

## 0. Host prerequisites

On a fresh Chameleon Ubuntu 24.04 VM, install miniconda, accept the
Anaconda repo terms of service, and cap thread pools. The thread caps are a
defensive default everywhere in this guide — BLAS/MKL otherwise starve
librdkafka's broker threads (`Unable to create broker thread`).

```bash
# miniconda (first-time only)
curl -L -o /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

# mandatory ToS accept before §2 (the conda module on Polaris ships pre-accepted)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# re-export in every shell that runs this guide
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

**Polaris note.** Login nodes additionally enforce a `pids.max = 128` cgroup,
so `mamba`'s parallel solver and parallel conda fetches blow up the per-user
PID budget. Use plain `conda` and add
`export CONDA_FETCH_THREADS=1 CONDA_EXTRACT_THREADS=1`. Load conda from the
module system instead of miniconda:
`module use /soft/modulefiles/ && module load conda` — that module ships
with the Anaconda ToS already accepted, so the `conda tos accept` lines
above are unnecessary there.

---

## 1. Stage the mochi-hpc conda channel

Conda does not expand `$HOME` or env vars in channel URLs, so the channel
must live at the absolute path hardcoded in
[environment-chameleon-stream.yml:12](environment-chameleon-stream.yml#L12)
(`/home/cc/mochi-conda-packages`, matching the default Chameleon user). If
you're on a host where `$USER != cc`, either symlink
`/home/cc/mochi-conda-packages` to your own staging directory or sed-replace
the URL in the yml before §2.

```bash
mkdir -p /home/cc/mochi-conda-packages && cd /home/cc/mochi-conda-packages

# Pinned: 2026-04-27 through the latest release (2026-05-12) break the env
# (libuuid conflict, see §8; re-verified 2026-05-29). Bump past 2026-03-30 only
# once a release rebuilds diaspora-stream-octopus against libuuid/util-linux >=2.42.
TAG=2026-03-30
curl -L "https://github.com/mochi-hpc/mochi-conda-packages/releases/download/$TAG/mochi-conda-channel-$TAG-linux-64-py3.12.tar.gz" \
    | tar -xz --strip-components=1

ls   # expect: linux-64/  noarch/  channeldata.json
```

To re-test a newer tag, set `TAG=` to the candidate and re-run — it overwrites
the extracted channel in place. Wipe `~/conda-envs/mofa-stream` before
recreating with §2 so conda actually re-solves against the new channel.

---

## 2. Create the conda env

First-time create (~10 min, mostly chargemol/raspa2):

```bash
cd ~/mof-generation-at-scale
conda env create \
    --file envs/environment-chameleon-stream.yml \
    --prefix ~/conda-envs/mofa-stream
conda activate ~/conda-envs/mofa-stream
pip install ConfigSpace   # only needed for `mofkactl config generate`
```

After a new mochi-conda-packages release, re-stage with §1 and pull just the
mochi packages forward:

```bash
conda update --override-channels \
    -c file:///home/cc/mochi-conda-packages -c conda-forge \
    --prefix ~/conda-envs/mofa-stream --yes \
    mofka diaspora-stream-api diaspora-stream-octopus
```

---

## 3. Verify the env

```bash
~/conda-envs/mofa-stream/bin/python tests/smoke_stream_env.py
```

Checks every import the stream workflow needs —
`diaspora_stream.api.Driver`, `mochi.mofka`, `liboctopus.so` (dlopen),
`diaspora_event_sdk.Client`, `colmena.queue.base`, `torch`, `ase`,
`mofa.diaspora`. Expected: `8/8 checks passed`. Any failure makes the env
unusable for stream workflows.

---

## 4. Start a local Mofka bedrock daemon (mofka mode only)

Mofka needs a bedrock daemon. `mofkactl`'s minimal config needs several
tweaks before bedrock will accept it (JSON array → object, flock
`bootstrap` → `self`, yokan database → in-memory `map` on non-Polaris
hosts), and the daemon needs `LD_PRELOAD` for lmdb + leveldb that
`libyokan-server.so` dlopens without listing in `DT_NEEDED`.
[bin/start-bedrock.py](../bin/start-bedrock.py) does all of that:

```bash
# start-bedrock.py reads $CONDA_PREFIX to locate the bedrock binary and the
# lmdb/leveldb libraries it LD_PRELOADs, so the env must be activated first —
# invoking the env's bin/python directly is *not* enough (the script exits
# early with "CONDA_PREFIX not set — activate the mofa-stream env first").
conda activate ~/conda-envs/mofa-stream
python bin/start-bedrock.py /tmp/mofka-test
# stdout: /tmp/mofka-test/mofka.flock.json
# stderr: # bedrock pid=<n> log=/tmp/mofka-test/bedrock.log
```

The helper uses `ofi+tcp://127.0.0.1` (single-host, Parsl-fork-safe; the
faster `na+sm` transport is blocked by Linux Yama on cloud images and
deadlocks Parsl-forked children — see
[../docs/adr/0001-mofka-bedrock-transport.md](../docs/adr/0001-mofka-bedrock-transport.md)).
`BEDROCK_YOKAN_MAP=1` (default) patches the yokan backend to in-memory
`map` because the upstream lmdb/leveldb/rocksdb backends reject mofka's
`put_direct` mode on the libstdc++ ABI Chameleon (and other non-Polaris
hosts) ship; unset it once
[mochi-conda-packages](https://github.com/mochi-hpc/mochi-conda-packages)
fixes the upstream ABI mismatch.

When done, tear bedrock down:

```bash
pkill -f bedrock-config-single.json
rm -rf /tmp/mofka-test/data_* /tmp/mofka-test/metadata
```

---

## 5. Octopus tokens (one-time setup)

Octopus authenticates via Globus + AWS MSK Kafka. Bootstrap your tokens by
running [tests/smoke_octopus.py](../tests/smoke_octopus.py) once:

```bash
~/conda-envs/mofa-stream/bin/python tests/smoke_octopus.py
```

The first invocation walks you through an interactive Globus login and
caches tokens in `~/.diaspora/storage.db`. The same script also rotates AWS
keys via `create_key` and deletes any topics in your namespace — on a fresh
account both are no-ops, so first-time setup is safe. Subsequent runs are
non-interactive (see §7 for the cleanup use case).

**Expected output** on first-time setup (Globus prompt elided):

```
delete_user: …
create_key: access=AKIA…
list_namespaces: {}
deleted=0 remaining=0
```

---

## 6. Run [mofa/diaspora.py](../mofa/diaspora.py) — 5-topic Colmena round-trip

Exercises `DiasporaQueues` end-to-end: `send_inputs → get_task →
send_result → get_result` on each of `generation`, `lammps`, `cp2k`,
`training`, `assembly`. The mofka mode talks to the bedrock daemon from §4;
octopus talks to AWS MSK via cached Globus tokens.

"5-topic" counts the five task classes round-tripped. The underlying topic
count is higher: mofka pre-wires **7** topics up front and logs
`7 topic(s) pre-wired` — `<prefix>_requests`, a `<prefix>_default_result`
(colmena's always-present `default` queue), and one `<prefix>_<class>_result`
per task class. octopus creates topics lazily and deletes the **6** it
touched (`<prefix>_requests` + the five `*_result`) on exit; it never
materialises `default_result`.

Shared shell setup (re-export §0 caps, then):

```bash
P=~/conda-envs/mofa-stream/lib
PRELOAD="$P/liblmdb.so:$P/libleveldb.so"
PY=~/conda-envs/mofa-stream/bin/python
```

| Mode | Command | What it exercises |
|---|---|---|
| `mofka` | `LD_PRELOAD=$PRELOAD LD_LIBRARY_PATH=$P $PY -u mofa/diaspora.py mofka --group-file /tmp/mofka-test/mofka.flock.json` | Opens `Driver(backend="mofka")` against bedrock, creates topics, auto-adds a memory partition per topic, round-trips one task+result per topic over margo/mochi RPCs. |
| `octopus` | `$PY -u mofa/diaspora.py octopus` | Opens `Driver(backend="octopus")` against AWS MSK: pulls Globus tokens, signs IAM, creates Kafka topics, round-trips over librdkafka, then deletes its prefix-scoped topics on exit. |

**Expected output** (both modes), final two log lines:

```
… __main__ INFO assembly: <hash> -> {'input': <hash>, 'output': <2*hash>}
… __main__ INFO RESULT: PASS (5 round-trips on <mode>)
```

The logger is `__main__` (not `mofa.diaspora`): the file is run as a script.

Process exits 0. `mofa/diaspora.py` uses `os._exit` because the diaspora
Driver's C++ destructors hang on regular interpreter shutdown. The first
octopus invocation against a fresh Kafka topic may take >60 s to propagate;
re-run if it times out.

**Benign log noise** — neither of these is a failure; only `RESULT: PASS`
(exit 0) vs `FAIL` (exit 1) decides the run:

- mofka: `mercury->cls [warning] … HG_Core_register() Overwriting RPC
  callback for a previously registered RPC ID …`, once per topic.
- octopus: `[octopus:warning] BatchSize ignored by consumer`,
  `MaxNumBatches ignored by consumer`, and `Could not read info topic …
  using default validator, serializer, and partition selector` (topics are
  created without an info record). On exit, `%5|…|PARTCNT|rdkafka…| Topic …
  partition count changed from 1 to 0` accompanies each topic delete.

Last re-verified **2026-05-27**, both modes PASS: mofka returns in ~1 s
(single local bedrock); octopus takes ~60 s end-to-end (Kafka topic
propagation dominates — ~10 s per task class).

---

## 7. Wipe lingering kafka topics — [tests/smoke_octopus.py](../tests/smoke_octopus.py)

`diaspora.py octopus` deletes its own topics on exit, but if a run dies
before its cleanup phase the prefix-scoped topics stay around and eat into
the per-user kafka topic budget (symptom: subsequent runs fail with `Unable
to create broker thread`). Re-run the same script from §5 to walk
`delete_user → create_key → list_namespaces → delete_topic` over every
topic in the caller's namespace. Destructive — it also rotates your AWS
access/secret keys.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PY tests/smoke_octopus.py
```

**Expected output** (N = number of stale topics found):

```
delete_user: …
create_key: access=AKIA…
list_namespaces: {'<ns>': ['dq_xxxx_requests', 'dq_xxxx_generation_result', …]}
delete_topic ns=<ns> topic=…: …
delete_topic ns=<ns> topic=…: …
deleted=<N> remaining=0
```

Exits 0 if `remaining == 0`. A non-zero exit means the server reported
topics still present after the delete loop — re-run, or check for a leaked
producer/consumer still holding handles to the topic.

---

## 8. Known issues

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: can't start new thread` during `conda env create` | 128-PID cgroup cap + mamba's solver | Use `conda`, set `CONDA_FETCH_THREADS=1 CONDA_EXTRACT_THREADS=1` |
| `OpenBLAS blas_thread_init: pthread_create failed` | numpy/scipy default to all cores | Export the four `*_NUM_THREADS=1` caps from §0 |
| `Unable to create broker thread` / segfault partway through `diaspora.py octopus` | librdkafka can't spawn enough broker threads — usually BLAS workers stole the per-user thread budget, or leftover python processes are still holding threads | Apply §0 caps; `pkill -f diaspora.py` to drop stale runs; run §7 to wipe lingering kafka topics |
| `bedrock` exits with `undefined symbol: mdb_env_create` or `leveldb::WriteBatch::~WriteBatch` | `libyokan-server.so` doesn't declare lmdb/leveldb in `DT_NEEDED` | `LD_PRELOAD="$P/liblmdb.so:$P/libleveldb.so"` |
| `bedrock` exits with `cannot use key() for non-object iterators` | `mofkactl config generate` emits a JSON array | `bin/start-bedrock.py` unwraps it; the snippet is preserved inline if you bypass the helper |
| `ImportError: No module named 'ConfigSpace'` from `mofkactl config generate` | optional dep | `pip install ConfigSpace` |
| `SyntaxError: future feature annotations is not defined` when running `mofkactl` directly | bash wrapper resolves to system python 3.6 | `python -m mochi.mofka.mofkactl …` |
| `No ABT-IO provider provided or found in server` from `add_default_partition` (mofka 0.9+) | `add_default_partition` now requires an abt-io provider and a `partition_config` path | `mofa.diaspora.DiasporaQueues` calls `add_memory_partition` instead — works against the minimal bedrock config and is appropriate for ephemeral test data |
| `mochi-yokan`/`mofka`/`mochi-warabi` require `libuuid >=2.42` but `diaspora-stream-octopus` requires `util-linux >=2.41.3,<2.42` during `conda env create` (mochi-conda-packages ≥ 2026-04-27; **still present in the latest release `2026-05-12`** — re-verified 2026-05-29) | conda-forge ships `libuuid` and `util-linux` from one feedstock at a single version, so `libuuid >=2.42` and `util-linux <2.42` are mutually exclusive. Upstream bumped the mochi packages (yokan 0.9.2, mofka 0.9.0, warabi 0.7.1) to `libuuid >=2.42` but `diaspora-stream-octopus 0.2.0` is unchanged (identical build `h3fd9d12_0` from 2026-03-30 through 2026-05-12) and still pins `util-linux <2.42` — the two no longer co-install | §1 pins `TAG=2026-03-30`; revisit when a release rebuilds `diaspora-stream-octopus` against `util-linux`/`libuuid >=2.42` |
| `Yokan error: Invalid mode` / bedrock logs `yk_put_direct_ult:132: mode not supported by database` on `create_topic` (mochi-conda-packages 2026-03-17 build) | mofka 0.8.2 in that build calls `put_direct` with a mode the bundled yokan 0.9.1 backends reject | use `TAG=2026-03-30` — same versions but rebuilt; the runtime regression is gone |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` from a fresh Miniconda install | a fresh Miniconda doesn't pre-accept the Anaconda repo ToS (the Polaris `conda` module does) | Run the `conda tos accept …` lines in §0 before §2 |
| `bedrock` (`na+sm`) fails: `na_sm_process_vm_writev() ... Operation not permitted` / `Kernel Yama configuration does not allow cross-memory attach` (Ubuntu 24.04 has `kernel.yama.ptrace_scope >= 1` by default) — *or* the parsl-forked task server hangs against an `na+sm` bedrock | `na+sm` uses `process_vm_writev` (Yama-blocked) **and** keys its `/dev/shm` segment by margo address (parsl-forked children contend with the parent and deadlock) | Use `ofi+tcp` instead — `bin/start-bedrock.py` does this by default. See [docs/adr/0001-mofka-bedrock-transport.md](../docs/adr/0001-mofka-bedrock-transport.md) |
| `Yokan error: Invalid mode` *still* fires on TAG=2026-03-30 on Ubuntu 24.04 | the upstream lmdb/rocksdb/leveldb yokan backends reject mofka's `put_direct` mode against Ubuntu 24.04's libstdc++ ABI | Already handled by `bin/start-bedrock.py` — `BEDROCK_YOKAN_MAP=1` (default) rewrites the yokan provider's `database` to `{"type":"map","config":{}}`. Unset it once mochi-conda-packages ships an ABI-compatible build |
| `pybel.py` aborts at import with `libXrender.so.1: cannot open shared object file` on a fresh VM | a handful of openbabel format `.so`s dlopen libXrender even though MOFA never reads those formats | `sudo apt install -y libxrender1` (see §9) |
| `store_cp2k - WARNING - Task run_optimization failed ... AssertionError()` during a §9 workflow run | ASE's CP2K calculator asserts `'cp2k_shell' in command`, but conda-forge's serial cp2k ships only the bare `cp2k.ssmp` binary, so MOFA's `cp2k.ssmp --shell` command fails the assert before CP2K runs. (Getting `cp2k.ssmp` at all now needs the `=*=*nompi*` pin in the §2 env yml; unpinned `cp2k` resolves to the MPI-only 2026.1 build, which has no `cp2k.ssmp`) | Install the `cp2k_shell.ssmp` wrapper (`bin/install-cp2k-shell-wrapper.sh`) and point `MOFA_CP2K_BIN` at it — see §9. `run-cloud-vm.sh` does this by default |
| `store_cp2k - WARNING - Task run_optimization failed ... ValueError("invalid literal for int() with base 10: ''")`; CP2K `cp2k.out` shows `[ABORT] ... basis set <DZVP-MOLOPT-SR-GTH> ... not found in BASIS_MOLOPT` | conda-forge does not set `CP2K_DATA_DIR`, so CP2K can't find its basis-set/potential files; ASE then chokes parsing the empty response | The `cp2k_shell.ssmp` wrapper from §9 exports `CP2K_DATA_DIR=$CONDA_PREFIX/share/cp2k/data`; or set it yourself |
| **(FIXED 2026-05-27)** `octopus` §9 smoke workflow never completes — generation/assembly keep round-tripping but no LAMMPS result ever lands; `sim/interchange.log` shows `HOLD_WORKER` then `Too many heartbeats missed for manager - removing manager` minutes before the first MD task is submitted | `LocalConfig.make_parsl_config()` left the executors on parsl's default provider (`min_blocks=0`) + `simple` strategy, which reaps an idle executor's worker after `max_idletime` (~120 s). octopus's per-message latency makes the generation→assembly pipeline take ~7 min to produce the first MOF, so the idle LAMMPS/CP2K (`sim`) worker is reaped before the first MD task arrives. mofka's near-instant generation gets a task in before the timeout, which is why only octopus hit it | Fixed: `CloudVMConfig.make_parsl_config()` now pins every executor with `LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1)` (same as the HPC configs), so the worker is never reaped. octopus now runs MD + CP2K end-to-end |

---

## 9. Run the full workflow

Extra prerequisites needed to run `run_parallel_workflow.py` end-to-end on
Chameleon. The env yml in §2 already pulls in CP2K, RASPA2, chargemol,
redis-server (now unused — ProxyStore is disabled, but `bin/run-cloud-vm.sh`
still starts it and `--redis-host` is still passed/parsed), and a CPU LAMMPS
build (`pair_style mliap` only — see "LAMMPS source build" below for the fork
that supplies `pair_style mace`).
System-level packages the env doesn't install:

```bash
# mongodb-org is in mongodb-org's own apt repo
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | sudo gpg -o /etc/apt/keyrings/mongodb-server-8.0.gpg --dearmor
echo "deb [signed-by=/etc/apt/keyrings/mongodb-server-8.0.gpg] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" \
    | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
sudo apt update && sudo apt install -y mongodb-org libxrender1
```

`mongodb-org` provides `mongod` (the env yml omits the conda package because
it conflicts with mochi-margo's libuuid). `mongodb-org/7.0` is missing a
`noble` release; use `8.0` (verified at 8.0.23). `libxrender1` is dlopened by
some openbabel format readers; without it `from openbabel import pybel`
aborts at import time even though MOFA only uses a handful of formats.

### LAMMPS source build (`pair_style mace`)

The conda-forge LAMMPS from §2 enables `pair_style mliap` only.
`CloudVMConfig.lammps_pkg` defaults to `'ml-mace'`, which needs
`pair_style mace` from the [ACEsuit/lammps](https://github.com/ACEsuit/lammps)
fork so LAMMPS can load the libtorch-format MACE model at
`input-files/mace/mace-mp0_medium-lammps.pt`. The mliap fallback isn't
viable right now: the `mace-mp-0a → mliap` converter in `mace-torch==0.3.13`
fails on a state-dict shape mismatch. Source-build the mace branch once:

```bash
# CPU libtorch with cxx11-abi — NOT the conda-shipped CUDA libtorch
curl -L -o /tmp/libtorch.zip \
    https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
unzip -q /tmp/libtorch.zip -d /home/cc/    # extracts to /home/cc/libtorch

# ACEsuit/lammps mace branch (last verified at 4d222cb)
git clone -b mace https://github.com/ACEsuit/lammps.git /home/cc/lammps-mace
mkdir -p /home/cc/lammps-mace/build && cd /home/cc/lammps-mace/build

cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH=/home/cc/libtorch \
    -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include \
    -DPKG_ML-MACE=on -DPKG_PYTHON=off
cmake --build . -j$(nproc)

./lmp -h | grep -i 'mace'      # expect "mace" in the pair_style list

# generate the libtorch-format MACE model pair_style mace consumes
~/conda-envs/mofa-stream/bin/mace_create_lammps_model \
    --format libtorch --dtype float32 \
    ~/mof-generation-at-scale/input-files/mace/mace-mp0_medium.model
# produces ./mace-mp0_medium-lammps.pt; move/symlink into input-files/mace/

export MOFA_LAMMPS_BIN=/home/cc/lammps-mace/build/lmp
```

Gotchas: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required because parts of
the lammps build tree still use cmake 2.x policy syntax;
`-DMKL_INCLUDE_DIR` lets the FFT pkg find conda's `mkl_dfti.h`; the libtorch
flavor must be CPU cxx11-abi (the CUDA-libtorch shipped via conda's
`pytorch-cuda` requires a CUDA toolchain Chameleon CPU images don't have).

Set `MOFA_LAMMPS_BIN` before invoking [bin/run-cloud-vm.sh](../bin/run-cloud-vm.sh)
(or persist it in `~/.bashrc`). To skip the source-build and stick with
the conda LAMMPS, switch `CloudVMConfig.lammps_pkg` back to `'ml-iap'` and
supply an mliap-format model — but the mliap path is not currently
validated on Chameleon.

[bin/run-cloud-vm.sh](../bin/run-cloud-vm.sh) drives a trimmed-budget
end-to-end run for a single backend:

```bash
bin/run-cloud-vm.sh mofka     # auto-launches bedrock via bin/start-bedrock.py
bin/run-cloud-vm.sh octopus   # needs cached Globus tokens (§5)
bin/run-cloud-vm.sh files     # local-file backend, no daemon
```

The compute config is [configs/cloud-vm.py](../configs/cloud-vm.py) →
`CloudVMConfig` (CPU-only). It resolves `lmp` and the CP2K command from
`$PATH`; override with `MOFA_LAMMPS_BIN` / `MOFA_CP2K_BIN`.

**CP2K needs a `cp2k_shell.ssmp` wrapper** — `cp2k.ssmp --shell` does *not*
work directly. ASE's CP2K calculator asserts the launch command contains the
substring `cp2k_shell` (`assert 'cp2k_shell' in command` in
`ase/calculators/cp2k.py`), so the bare `cp2k.ssmp --shell` raises
`AssertionError` before CP2K runs. And conda-forge does not set `CP2K_DATA_DIR`, so even once the shell
launches CP2K aborts (`basis set <DZVP-MOLOPT-SR-GTH> ... not found in
BASIS_MOLOPT`). Note `cp2k.ssmp` only exists in conda-forge's *serial* build:
cp2k 2026.1 is MPI-only (`cp2k.psmp`/`cp2k.popt`), so the §2 env yml pins
`conda-forge::cp2k=*=*nompi*` — the newest serial build that still ships
`cp2k.ssmp` (cp2k 2024.2).
[bin/install-cp2k-shell-wrapper.sh](../bin/install-cp2k-shell-wrapper.sh)
installs a wrapper into the active env's `bin/` that fixes both — its name
satisfies ASE's assert and it exports `CP2K_DATA_DIR`:

```bash
conda activate ~/conda-envs/mofa-stream
bin/install-cp2k-shell-wrapper.sh        # creates $CONDA_PREFIX/bin/cp2k_shell.ssmp
```

[bin/run-cloud-vm.sh](../bin/run-cloud-vm.sh) defaults `MOFA_CP2K_BIN` to
`cp2k_shell.ssmp` and refuses to start if the wrapper isn't on `$PATH`.

The wrapper also sets `OMP_NUM_THREADS=8` (override with `MOFA_CP2K_OMP`):
the global `OMP_NUM_THREADS=1` cap from §0 otherwise leaves CP2K
single-threaded, and one MOF SCF then takes tens of minutes. CP2K runs in its
own parsl worker subprocess, so its OpenMP threads don't steal librdkafka's
broker threads in the main process; nested BLAS stays single-threaded
(`MKL`/`OPENBLAS` caps are inherited) to avoid OMP×BLAS oversubscription. Even
threaded, CP2K can dominate wall time: a generated MOF whose SCF doesn't
converge grinds to `max_scf=128` (`mofa/simulation/dft/cp2k.py`). That's a
DFT-convergence cost, independent of the stream backend.

### 9a. Parameter tiers (CPU-only cloud VM)

Pick a tier based on what you're trying to prove. Wall-time figures come from
end-to-end runs on a 48-core Ubuntu 24.04 cloud VM with no GPU.

| Tier | `--simulation-budget` | `--num-samples` / `--gen-batch-size` | `--molecule-sizes` | `--md-timesteps` | `--dft-opt-steps` | `--retrain-freq` / `--num-epochs` | Approx wall (mofka / octopus) |
|---|---:|---:|---|---:|---:|---:|---|
| **smoke** (defaults baked into `bin/run-cloud-vm.sh`) | 1 | 4 / 4 | `8 12` | 50 | 1 | 1 / 1 | ~8 min / ~15 min + CP2K SCF (see below) |
| **md** (the 3–6 hour tier) | 4 | 8 / 8 | `8 10 12` | 200 | 8 | 2 / 2 | ~3–6 hr / ~3–6 hr (driven by first MD on a hard MOF) |

**Smoke** proves the orchestration boots and the queue ferries one round of
every task class (generation, ligand-validate, assembly, LAMMPS+MACE, CP2K).
Use it before any longer run to confirm the chain still works.

- **mofka**: validated PASS (2026-05-27) — `MOFAThinker completed`, process
  exits rc=0 in ~8 min. (Whether the CP2K step actually fires is
  nondeterministic at `--simulation-budget 1`: the LAMMPS budget can be spent
  before a MOF is queued for CP2K. CP2K itself is verified working via the §9
  wrapper — see the known-issue rows in §8.)
- **octopus**: runs the full pipeline (generation → MD → CP2K) after the
  parsl idle-worker fix (§8). Verified 2026-05-27: 2/2 LAMMPS+MACE relaxations
  completed (the step that used to stall). It is much slower than mofka — the
  Kafka round-trip latency makes the generation→assembly phase take ~7 min, and
  CP2K then dominates. Reaching `MOFAThinker completed` is gated on CP2K SCF
  convergence, which is MOF-dependent (a generated MOF whose SCF oscillates
  grinds to `max_scf=128`); that cost is not octopus-specific. For a quick
  full-chain sanity check prefer **mofka**; use octopus when you specifically
  need to exercise the AWS-MSK path end-to-end.

**md** exercises serious MD wall time: 4 successful LAMMPS results, each at
4× the smoke `--md-timesteps`, so LAMMPS dominates total wall. The 3–6 h
range is wide because CPU MD per MOF varies enormously with system size
(observed: one MOF finished in 30 s, another 40+ min at md=200). Generation
on the larger `--num-samples` adds another ~10–20 min upfront.

Don't go bigger than the **md** tier on a single cloud VM. A `--simulation-budget 8 --num-samples 16 --md-timesteps 1000` run hit 14 h then **deadlocked in
the shutdown drain phase** — `submit_*` agents completed but `store_*` agents
got stuck in `consumer.pull()` after `queues.send_kill_signal()`, which
appears to be a Colmena-shutdown / message-buffering interaction at high
in-flight task counts. Investigation TBD; for now don't trigger it.

### 9b. Invoking with custom params

`bin/run-cloud-vm.sh` hardcodes the smoke tier. For the **md** tier, call
`run_parallel_workflow.py` directly (still under the same env exports the
launcher sets — `PATH`, `CONDA_PREFIX`, `MOFA_LAMMPS_BIN`,
`MOFA_CP2K_BIN=cp2k_shell.ssmp`, BLAS caps, and for mofka also `LD_PRELOAD` /
`LD_LIBRARY_PATH`):

```bash
# mofka — first start bedrock and capture the group file
GROUP_FILE=$(python bin/start-bedrock.py /tmp/mofka-run)

# then launch with md-tier params
python run_parallel_workflow.py \
    --node-path input-files/zn-paddle-pillar/node.json \
    --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
    --generator-config-path models/geom-300k/config-tf32-a100.yaml \
    --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
    --retrain-freq 2 --num-epochs 2 \
    --num-samples 8 --gen-batch-size 8 --molecule-sizes 8 10 12 \
    --simulation-budget 4 --md-timesteps 200 --dft-opt-steps 8 \
    --redis-host 127.0.0.1 \
    --compute-config configs/cloud-vm.py \
    --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
    --stream-engine mofka --mofka-group-file "$GROUP_FILE"
```

Swap `--stream-engine mofka --mofka-group-file "$GROUP_FILE"` for
`--stream-engine octopus` (no group file) to run against AWS MSK.

#### Thinker/server split (`--launch-option`)

The command above runs the whole workflow in one process (`--launch-option
both`, the default — unchanged behaviour). To run it as the legacy octopus2
**thinker** (steering + MongoDB) and **server** (Parsl task executor) as two
separate processes — e.g. on two hosts, or in separate containers — pass
`--launch-option` plus a *shared* `--queue-prefix` so both halves rendezvous on
the same stream topics, and point the server's `--mongo-host` at the thinker:

```bash
# thinker: steering + a mongod bound to 0.0.0.0
python run_parallel_workflow.py ... --launch-option thinker --queue-prefix mofa_run1
# server: Parsl task server only; reaches the thinker's mongod
python run_parallel_workflow.py ... --launch-option server --queue-prefix mofa_run1 --mongo-host <thinker-host>
```

Without `--queue-prefix` each process picks a random prefix, so a split pair
would never meet. The [mofa-mini-app](https://github.com/diaspora-project/mofa-mini-app)
containerizes exactly this split. (This replaces the mini-app's former
build-time monkey-patch of `run_parallel_workflow.py`.)

### 9c. Cleanup

The **smoke** mofka run was observed to exit cleanly on its own (rc=0) a
couple of seconds after `MOFAThinker completed` — `run_parallel_workflow.py`
has no `os._exit`; it runs a `finally` that calls `queues.send_kill_signal()`
and `store.close()`, then the interpreter shuts down. (`mofa/diaspora.py`'s
standalone test *does* `os._exit` because its bare `Driver`'s C++ destructors
hang, but the full workflow's ordered teardown avoids that for the smoke
tier.) Larger runs can still hang in the shutdown drain — the
`--simulation-budget 8` run noted above deadlocked there — so keep the
force-kill commands as a fallback. Once `MOFAThinker completed` is logged (or,
on **md**, the heartbeat counters stop moving and submit agents are done) and
the process has not exited within a minute or two:

```bash
pkill -9 -f "run_parallel_workflow.py"
pkill -9 -f "bedrock-config-single"     # mofka only
pkill -9 -f "mongod.*parallel-cloud"
pkill -9 -f "parsl: HTEX"               # often orphaned
```

(If you run these from a shell whose own command line contains the pattern —
e.g. a one-liner that also greps for `run_parallel_workflow` — `pkill -f` will
match and kill that shell too; use a bracketed pattern like
`'[r]un_parallel_workflow.py'` to avoid the self-match.)

After an octopus run, scope topic cleanup to this run's prefix only:

```bash
~/conda-envs/mofa-stream/bin/python tests/smoke_octopus.py \
    --prefix <mofa_xxxxxx-from-run.log> --no-rotate-keys
```

The workflow logs its prefix at start-up (search the log for `mofa_` + 6 hex).
`--no-rotate-keys` preserves cached AWS credentials so subsequent runs don't
need a fresh Globus auth.
