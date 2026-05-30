# Mofka thread-affinity probes (evidence for ADR-0002)

These are the standalone scripts that produced the probe matrix in
[../0002-mofka-dispatcher-thread.md](../0002-mofka-dispatcher-thread.md). They
exist so the matrix is reproducible rather than citing results that only ever
lived in a terminal. Each probe isolates one threading configuration of the
mochi-mofka / Argobots runtime and reports `PASS`, a silent drop, an abort, or a
hang.

Originally run **2026-05-28** against a loopback `ofi+tcp` bedrock
(`mofka 0.8.2` / `diaspora-stream-api 0.5.2`); the recorded verdicts are the
ones in the ADR matrix.

## Probe → matrix-row map

| Script | Test | Configuration | Recorded verdict |
|---|---|---|---|
| `affinity_probe.py` | T1 | all calls on the one constructing (main) thread | **PASS** (matrix row 1) |
| `affinity_probe.py` | T2 | producer/consumer built on main, pushed/pulled from a worker thread | **silent drop** (row 2) |
| `affinity_probe.py` | T3 | two plain threads, each with its OWN `Driver` (proxystore-style per-role) | off-anchor failure (row 3) |
| `affinity_probe2.py` | T4 | internal engine given a JSON `margo` progress thread + `start_progress_thread()`, then off-thread calls | still **silent drop**; `start_progress_thread` not bound on the diaspora `Driver` (Alternatives) |
| `affinity_probe3.py` | T5 | two independent per-thread engines, **no** idle engine-holder on main | thread A **PASS**, thread B aborts constructing the 2nd engine → ES is per-process; 2-engine case is the row-4 boundary |
| `affinity_probe4.py` | T6 | worker builds its own engine while main is `idle` / `hold` / `drop` | `idle`→worker ok; `hold` (idle engine-holder)→**HANG** (row 5) |
| `affinity_probe5.py` | T7 | 2 per-thread engines **with** per-engine progress threads, main holds an idle engine and blocks in `join` | **HANG/drop** even with progress threads — at this 2-thread scale |
| `affinity_probe6.py` | T8 | **6** self-contained per-thread engines (the thinker's thread count), concurrent construction | **HANG** (row 6) |
| `affinity_probe7.py` | T9 | same 6 engines but all construction serialized behind one lock | **HANG** → the deadlock is N-engine *steady state*, not a construction race (row 6) |
| `affinity_probe_parsl.py` | V1/V2 | the **parsl** companion to `affinity_probe.py`: margo anchored on main (control push PASSes), then a parsl `ThreadPoolExecutor` app touches mofka on a worker thread two ways | V1 *construct* off-anchor → `ABT_ERR_INV_XSTREAM` (5/5 here, but order-dependent — see the docstring); V2 *use* a main-built consumer off-anchor → **silent drop** (≈4/6) or whole-process **hang** (≈2/6). Rows 1 + 2/3 via parsl rather than `threading.Thread` |

Net: the only configuration that survives the thinker's multi-thread access is
"every mofka call on one thread" — the dispatcher. See the ADR for the argument.

## Running them

Point `MOFKA_GROUP_FILE` at a running bedrock's flock file (defaults to
`/tmp/mofka-offthread/mofka.flock.json`), and pull in the lmdb/leveldb shims the
mofka client dlopens:

```bash
P=~/conda-envs/mofa-stream/lib
export MOFKA_GROUP_FILE=/path/to/mofka.flock.json
export LD_PRELOAD="$P/liblmdb.so:$P/libleveldb.so" LD_LIBRARY_PATH=$P
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python docs/adr/probes/affinity_probe.py        # T1/T2/T3
python docs/adr/probes/affinity_probe6.py 6     # T8 at K=6
python docs/adr/probes/affinity_probe7.py 6     # T9 at K=6

timeout 120 python docs/adr/probes/affinity_probe_parsl.py   # mofka+parsl (needs bedrock)
```

`affinity_probe_parsl.py` runs under an outer `timeout` because its VARIANT 2
(off-anchor *use*) can deadlock the whole process — a blocked mofka C call holds
the GIL, so an in-process timeout can't recover it; read a missing VARIANT-2
line as the hang case.

The hang probes self-terminate via per-thread `join(timeout=…)` and report how
many threads are still alive, so they will not block forever.

### Reproduction caveat — multi-homed hosts

These must run against a bedrock the client can actually reach over loopback. On
a **multi-homed host** (e.g. one with docker bridge interfaces), libfabric's
`tcp` provider may advertise/respond on a non-loopback NIC even when bedrock is
bound to `127.0.0.1`; the daemon then aborts with `HG_TIMEOUT` and every probe
hangs — a transport/NIC artifact, not the affinity behaviour under test. Pin the
interface to loopback (e.g. `FI_TCP_IFACE` / `FI_SOCKETS_IFACE`) or run bedrock
and the probe in one clean network namespace. See
[../0001-mofka-bedrock-transport.md](../0001-mofka-bedrock-transport.md).
