# ADR-0002: Mofka calls funnel through a single dispatcher thread

## Status

Accepted (2026-05-27). Re-confirmed and expanded 2026-05-28 with a probe matrix
(below) after investigating whether the dispatcher could be dropped to mirror an
earlier proxystore-based integration. The investigation reinforced the decision.

Independently re-verified 2026-05-29: every matrix row maps to a committed probe
in [probes/](probes/) (and the row-6 wording was corrected — see the note below
the matrix); the two claims about the earlier design hold (the current API
removed engine injection, and that design's smoke test only ever ran one mofka
thread per process); and the full workflow round-trips end-to-end on **both**
mofka and octopus with the dispatcher, including a thinker/server split across
separate containers whose only channel is the stream backend.

## Context — why the current stack needs a dispatcher

`MOFAThinker` is Colmena's `BaseThinker`, which runs each decorated method as
its own thread. Mofka objects are therefore touched from **many** threads in
**both** Colmena processes:

- **Thinker process** — several `@task_submitter` threads (generation, lammps,
  cp2k) all push to the one `{prefix}_requests` queue, and each
  `@result_processor(topic=…)` is its own thread looping on
  `get_result(topic=…)` against a per-topic consumer.
- **Doer process** (the Parsl-forked `ParslTaskServer`) — the
  `listen_and_launch` thread pulls from `{prefix}_requests`, while Parsl's
  task-completion callbacks push results from other thread(s).

The natural fix — give each producer/consumer its own engine, as the earlier
proxystore design did (see *Alternatives*) — is not available here, for two
reasons.

**The API no longer lets you own an engine.** The unified mofka API
(`libmofka 0.8.2` / `diaspora-stream-api 0.5.2`) removed engine injection.
`MofkaDriver` takes only a JSON config and builds its engine internally. That
JSON *can* configure the engine — it accepts a `margo` block (protocol, pools,
even a progress thread) — but it cannot accept a pre-constructed `pymargo`
engine, and exposes no engine accessor.

**And more engines wouldn't help anyway, because an engine object is not an
isolated runtime.** Argobots — the scheduler under thallium — initializes once
per process and anchors to the first thread that starts it. Every engine you
create, however you configure it via JSON, attaches to that one process-global
runtime. mofka's calls run Argobots operations (e.g. `ABT_xstream_self`) that
only succeed on a thread the runtime recognizes as an execution stream — in
practice only the anchor thread. More engines ≠ more independent execution
streams.

We measured the threading behaviour directly on 2026-05-28 against the deployed
bedrock (`ofi+tcp`), one mofka engine per "role" except where noted. The scripts
are committed in [probes/](probes/) (one probe per row; see the README there for
the row map):

| Configuration | Result |
|---|---|
| all calls on the one constructing thread | **PASS** |
| object built on thread A, called from thread B | **silent drop** (no error) |
| object constructed on a thread that isn't the margo anchor | aborts `ABT_ERR_INV_XSTREAM` |
| 2 independent engines, separate threads, no idle engine-holder | **PASS** |
| an engine-holding thread that then blocks (e.g. the thinker's main thread) | **HANG** (whole process) |
| 6 independent per-thread engines (the thinker's thread count) | **HANG**, even when all construction is serialized behind one lock |

The last row's two backing probes settle *why* it deadlocks: concurrent
construction hangs (`probes/affinity_probe6.py`), and serializing all
construction behind one lock still hangs (`probes/affinity_probe7.py`) — so the
deadlock is N-engine steady state, not a construction race. The per-engine
*progress-thread* variant was only exercised at the 2-thread scale
(`probes/affinity_probe5.py`, and the single-engine `affinity_probe2.py`), where
it likewise failed to lift the affinity; it was not run at the 6-thread count, so
this row no longer claims it (see *Alternatives* for the progress-thread case).

The earlier rule of thumb — "objects are bound to the constructing thread" — is
too clean. A small number (≤2) of fully independent, always-active engines with
no idle engine-holder *can* coexist. But every configuration that matches the
thinker (≥6 agent threads, the main thread alive, a shared requests producer)
either silently drops, aborts, or — most often at realistic scale —
**deadlocks**. The silent-drop case is the most dangerous: no error is raised,
the workflow just never receives results.

So the only configuration that survives the thinker's multi-thread access is the
trivial one: **every mofka call on a single thread.** That thread is the
dispatcher. `fork()` keeps only the calling thread, so the dispatcher (and any
engine) never survives into a Parsl child; each process must build its own.

## Decision

All mofka work funnels through one dedicated `_MofkaDispatcher` thread per
process — the single working configuration (row 1 above):

- The dispatcher thread constructs the `Driver`, pre-wires every topic, and owns
  every producer/consumer for the life of the process. It is the one thread that
  initializes margo (so it is the Argobots anchor) and the only thread that calls
  mofka. It need not be the main thread; it must be the only one.
- Callers (the agent threads, the task server) submit `(op, qname, payload)`
  work items through a `queue.Queue` and block on a per-call future; the
  dispatcher executes them in order on its own thread.
- `os.register_at_fork(after_in_child=…)` discards the inherited dispatcher in
  Parsl-forked children; the next call lazily rebuilds it, so each process ends
  up with its own Argobots state.
- The send path skips `producer.flush()` (mofka's `flush().wait()` blocks the
  full timeout even with nothing batched); `push().wait()` returning means the
  broker acked.
- `_get_message(timeout=None)` polls in 1 s slices so a blocked `pull` can't
  starve other callers and Colmena's kill-signal contract is honoured.

Files and octopus bypass the dispatcher — they are pthread-safe.

## Distributed deployment — why the dispatcher is benign

The dispatcher is **per-process and entirely process-local**: a thread plus a
`queue.Queue` that funnels one process's own threads onto one thread. It holds no
cross-process state, no locks visible to other hosts, no shared files, and makes
no co-location assumption — it builds its `Driver` from the group file and talks
to whatever broker address that file names, local or remote. So spreading MOFA
across many Docker containers or VMs yields **one independent dispatcher per
process; they never interact and cannot conflict.** The lazy + per-process +
`register_at_fork` design is exactly what lets each process — wherever it lands —
bootstrap its own Argobots runtime independently. Distribution validates the
design rather than straining it.

What a distributed deployment actually needs is unrelated to the dispatcher and
lives in the transport (ADR-0001):

- **The broker must bind a routable address.** Today bedrock binds loopback
  (`bin/start-bedrock.py`: `ofi+tcp://127.0.0.1`), reachable only within one
  host. Multi-host requires a routable IP (or `0.0.0.0`) so the address baked
  into `mofka.flock.json` is reachable, plus the broker port opened between
  hosts. `na+sm` is single-host shared memory and stays rejected.
- **The group file must reach every streaming process** (mounted volume, copied
  in, or shared FS) — it encodes the broker address; the dispatcher only reads
  it.
- **One broker, many clients.** bedrock is single-process / single-host, so every
  distributed process opens its own connection to that one broker. That fan-in is
  a broker-scaling question (connection count, partition layout), not a
  dispatcher one.

For genuinely elastic multi-host runs, **octopus** (Kafka / AWS MSK) is the more
natural backend — already multi-host, no local broker to expose. The **files**
backend needs a shared filesystem. Neither changes the dispatcher.

## Alternatives considered

- **Per-object engine, no dispatcher (the proxystore / "Valerie's" design).**
  Each producer/consumer owns its own `pymargo.core.server` engine, created
  lazily on whichever thread first uses it — one engine per role, one Argobots ES
  per role, satisfied by construction. This is the clean design and is why that
  attempt needed no dispatcher — but it relied on the older mofka API that let
  you *inject* a server-mode engine into `MofkaDriver(group_file, engine)`. That
  constructor is gone here (the driver takes only a JSON config), and the probe
  matrix shows the pattern doesn't survive the thinker's thread count: 6
  independent per-thread engines deadlock even with serialized construction.
  Note also that the old design's smoke test only ever ran one mofka thread per
  process (its consumer was a separate `ProcessPoolExecutor`), so it never
  exercised the multi-thread case that breaks here.
- **Configure the internal engine via the JSON `margo` block / progress thread.**
  The JSON does accept a `margo` config, so the engine *is* configurable. But
  giving it a progress thread does not lift the affinity — off-thread calls
  still silently drop, and `start_progress_thread()` isn't even bound on the
  diaspora `Driver`.
- **Shared `Driver`, construction-order.** Build once on the main thread before
  the agent threads spawn, share lock-free. Disproven: off-thread `pull`
  silently drops and off-anchor construction aborts.
- **Per-thread `Driver`s.** A separate `Driver` in each agent thread. Aborts
  (`ABT_ERR_INV_XSTREAM`) or deadlocks at the thinker's thread count, and breaks
  mofka's per-topic producer/consumer uniqueness; doesn't survive the fork.
- **Lock around every mofka call.** Serialises the agent threads but doesn't fix
  affinity — the calls still run on the wrong thread (abort, or silent drop).
- **`spawn` start method.** Fresh interpreters, no inherited state; heavier,
  pickle-dependent, and unsupported as a global switch by Colmena/Parsl.

## Consequences

- **Performance: negligible.** The dispatcher serializes every mofka send/recv
  through one thread — a `queue.Queue` hop (~50–100 µs) per call, no cross-thread
  parallelism for mofka I/O, and one extra OS thread per process. This doesn't
  matter because mofka is the **control plane** (task-dispatch messages and
  result handles), not the data plane: heavy payloads ride ProxyStore/Redis and
  wall-clock is dominated by generation, MD, DFT, and GCMC — seconds to minutes
  per task, against sub-millisecond queue hops. The message rate is far below
  what one thread can serialize; the dispatcher mostly blocks on its work queue.
- **A dispatcher death is unrecoverable** — subsequent `submit()` calls raise.
  Accepted: silent data loss would be worse, and rebuilding the `Driver` outside
  the dispatcher reopens the affinity bug.
- **Octopus is NOT fork-safe** — librdkafka's background broker threads die on
  `fork`, so the octopus `Driver`, though built eagerly in `__init__`, is rebuilt
  in the forked child by the same `register_at_fork` hook. It is thread-safe for
  sharing once built, so it needs no dispatcher.
- **Mofka topic creation is racy across forked processes** — the child
  dispatcher may `create_topic` for a topic the parent already made; we swallow
  "already exists" and re-check `topic_exists`.
- **The constraint is version-bound, not fundamental.** If an upstream mofka
  restores injectable server-mode engines and the per-engine pattern proves
  robust at the thinker's thread count, the dispatcher could be revisited.
- **`ofi+tcp` (ADR-0001)** remains the bedrock transport for non-HPC hosts.
