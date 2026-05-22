# Polaris Diaspora Stream env

> branch: `diaspora-debug` · date: 2026-05-22 · python: 3.12

Build a conda env for MOFA's Polaris dependencies plus the
[mochi-hpc](https://github.com/mochi-hpc) **mofka** and **octopus** backends
used by `mofa.diaspora.DiasporaQueues`. Everything below runs **on a Polaris
login node** — Polaris ships glibc 2.38, so the prebuilt mochi binaries run
natively (no container).

The mochi packages are not on conda-forge. They ship as a per-release tarball
at [mochi-hpc/mochi-conda-packages/releases](https://github.com/mochi-hpc/mochi-conda-packages/releases).
The env yml ([environment-polaris-stream.yml](environment-polaris-stream.yml))
does not pin `mofka`, `diaspora-stream-api`, or `diaspora-stream-octopus`, so
whatever the staged channel contains is what you get. The snippet in §1 pins
the channel to **`TAG=2026-03-30`** — the last release whose yokan and
diaspora-stream-octopus agree on a libuuid pin (see §8 for the conflict in
≥ 2026-04-27).

---

## 0. Login-node prerequisites

Each user is in a cgroup with `pids.max = 128`. Anything that spins up a
thread pool (mamba's solver, OpenBLAS, MKL, rdkafka) eventually fails with
`RuntimeError: can't start new thread` or segfaults. Two consequences:

- **Use plain `conda`, not `mamba`** (mamba's solver uses a thread pool).
- **Cap thread pools** in every command from this guide:

    ```bash
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export CONDA_FETCH_THREADS=1 CONDA_EXTRACT_THREADS=1
    ```

Load conda:

```bash
module use /soft/modulefiles/ && module load conda
```

---

## 1. Stage the mochi-hpc conda channel

Conda does not expand `$HOME` or env vars in channel URLs, so the channel
must live at the absolute path hard-coded in
[environment-polaris-stream.yml:12](environment-polaris-stream.yml#L12)
(`/home/haochenpan/mochi-conda-packages`). If your username isn't
`haochenpan`, either symlink that path or `sed`-replace the URL in the yml
before §2.

```bash
mkdir -p /home/$USER/mochi-conda-packages && cd /home/$USER/mochi-conda-packages

# Pinned: 2026-04-27+ releases break the env (libuuid conflict, see §8).
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
    --file envs/environment-polaris-stream.yml \
    --prefix ~/conda-envs/mofa-stream
conda activate ~/conda-envs/mofa-stream
pip install ConfigSpace   # only needed for `mofkactl config generate`
```

After a new mochi-conda-packages release, re-stage with §1 and pull just the
mochi packages forward:

```bash
conda update --override-channels \
    -c file:///home/$USER/mochi-conda-packages -c conda-forge \
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

Mofka needs a bedrock daemon. The minimal config emitted by `mofkactl` needs
two tweaks before bedrock will accept it.

```bash
mkdir -p /tmp/mofka-test && cd /tmp/mofka-test

# 4a. Generate the bedrock config (single-server, shared-memory transport).
python -m mochi.mofka.mofkactl config generate \
    -a "na+sm" \
    --master-db-path-prefixes /tmp/mofka-test \
    --metadata-db-path-prefixes /tmp/mofka-test \
    --data-storage-path-prefixes /tmp/mofka-test \
    > bedrock-config.json

# 4b. Unwrap the JSON array (mofkactl emits a list; bedrock CLI wants a single
# object) and switch the flock provider from "mpi" to "self" (single-process).
python -c "
import json
cfgs = json.load(open('bedrock-config.json'))
cfg = cfgs[0]
for p in cfg['providers']:
    if p['type'] == 'flock':
        p['config']['bootstrap'] = 'self'
json.dump(cfg, open('bedrock-config-single.json','w'), indent=2)
"

# 4c. Launch. LD_PRELOAD is required: libyokan-server.so dlopens lmdb/leveldb
# but doesn't list them in DT_NEEDED.
P=~/conda-envs/mofa-stream/lib
LD_PRELOAD="$P/liblmdb.so:$P/libleveldb.so" LD_LIBRARY_PATH=$P \
    ~/conda-envs/mofa-stream/bin/bedrock na+sm -c bedrock-config-single.json \
    > bedrock.log 2>&1 &

sleep 3
tail -1 bedrock.log                       # expect: "Bedrock daemon now running at na+sm://<pid>-0"
ls /tmp/mofka-test/mofka.flock.json       # group file consumed by the client
```

`na+sm` is shared-memory transport (single-host only). For multi-host use
e.g. `ofi+tcp;ofi_rxm`. Always invoke `mofkactl` as
`python -m mochi.mofka.mofkactl …` — the bash wrapper resolves to the wrong
system python.

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
… mofa.diaspora INFO assembly: <hash> -> {'input': <hash>, 'output': <2*hash>}
… mofa.diaspora INFO RESULT: PASS (5 round-trips on <mode>)
```

Process exits 0. `mofa/diaspora.py` uses `os._exit` because the diaspora
Driver's C++ destructors hang on regular interpreter shutdown. The first
octopus invocation against a fresh Kafka topic may take >60 s to propagate;
re-run if it times out.

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
| `bedrock` exits with `cannot use key() for non-object iterators` | `mofkactl config generate` emits a JSON array | Unwrap with the snippet in §4b |
| `ImportError: No module named 'ConfigSpace'` from `mofkactl config generate` | optional dep | `pip install ConfigSpace` |
| `SyntaxError: future feature annotations is not defined` when running `mofkactl` directly | bash wrapper resolves to system python 3.6 | `python -m mochi.mofka.mofkactl …` |
| `No ABT-IO provider provided or found in server` from `add_default_partition` (mofka 0.9+) | `add_default_partition` now requires an abt-io provider and a `partition_config` path | `mofa.diaspora.DiasporaQueues` calls `add_memory_partition` instead — works against the minimal bedrock config and is appropriate for ephemeral test data |
| `mochi-yokan requires libuuid >=2.42` conflicts with `diaspora-stream-octopus requires util-linux >=2.41.3,<2.42` during `conda env create` (mochi-conda-packages ≥ 2026-04-27) | upstream rebuilt yokan against newer libuuid but did not rebuild octopus — the two no longer co-install | §1 pins `TAG=2026-03-30`; revisit when a release bumps `util-linux` in `diaspora-stream-octopus` |
| `Yokan error: Invalid mode` / bedrock logs `yk_put_direct_ult:132: mode not supported by database` on `create_topic` (mochi-conda-packages 2026-03-17 build) | mofka 0.8.2 in that build calls `put_direct` with a mode the bundled yokan 0.9.1 backends reject | use `TAG=2026-03-30` — same versions but rebuilt; the runtime regression is gone |
