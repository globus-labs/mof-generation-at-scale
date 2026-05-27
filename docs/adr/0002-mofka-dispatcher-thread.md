# ADR-0002: Mofka calls funnel through a single dispatcher thread

## Status

Accepted (2026-05-27).

## Context

`MOFAThinker` is Colmena's `BaseThinker`, which launches ~11 agent threads
(generator, training, MD selector, DFT selector, RASPA selector, result
collector, etc.). Parsl's `ParslTaskServer` additionally forks worker
processes. Both touch `DiasporaQueues`.

mochi-mofka is built on Argobots execution streams under thallium. An
Argobots ES has thread affinity: any call into `Driver`, `Topic`, `Producer`,
or `Consumer` from a thread other than the one that constructed the Driver
aborts with `ABT_ERR_INV_XSTREAM`, and `fork()` inheritance leaves the child
with corrupted ES state that deadlocks silently inside thallium on the next
call.

The original `mofa/diaspora.py` (on `diaspora-debug`) worked for the
single-threaded smoke test (`python mofa/diaspora.py mofka`) because every
call came from the main thread of one process. The full workflow does not
have that property.

Alternatives considered:

- **Per-thread Drivers.** Open a separate `Driver` in each Colmena agent
  thread. Workable but blows past mofka's per-topic producer/consumer
  uniqueness assumptions; also doesn't survive Parsl's fork.
- **Octopus-only.** librdkafka is thread-safe and fork-safe under
  `fork_after = false`. We considered restricting to octopus in Q2 of the
  design grilling, but `envs/chameleon-stream.md` documents both backends as
  supported and we want to honour that.
- **Lock around every mofka call.** A single `threading.Lock` lets the
  agent threads serialise, but doesn't fix the affinity issue: the lock
  holder still isn't the constructing thread, so the Driver still aborts.

## Decision

Lift debug2's `_MofkaDispatcher` design verbatim:

- A dedicated daemon thread (`MofkaDispatcher`) constructs the `Driver`,
  pre-wires all topics on first use, and owns every `producer`/`consumer`
  for the life of the process.
- All callers (agent threads, Parsl task server, etc.) submit work via a
  `queue.Queue` of `(op, qname, payload, future_queue)` tuples; the
  dispatcher executes them in order on its own thread and returns results
  through the per-call future.
- `os.register_at_fork(after_in_child=…)` discards the inherited dispatcher
  in Parsl-forked children; the next call lazily reconstructs it. Each
  process ends up with its own Argobots ES, no inheritance.
- The mofka send path skips `producer.flush()` (mofka 0.8.2's
  `flush().wait()` always blocks the full timeout even with nothing
  batched); `push().wait()` returning means the broker acked.

## Consequences

- **One extra OS thread per process** carrying the workflow. Negligible
  cost; the dispatcher mostly blocks on its work queue.
- **Send/recv latency now includes a queue.Queue hop.** Measured at
  ~50–100 µs on local benchmarks; well below mofka's per-RPC cost.
- **Files and octopus backends bypass the dispatcher** — they don't need
  it. The dispatcher is mofka-specific and is built lazily only when
  `stream_engine == "mofka"`.
- **A dispatcher death is unrecoverable.** If the thread crashes
  (`_init_error` set during construction, or an unhandled exception inside
  `_loop`), subsequent `submit()` calls raise. We accept this — silent
  data loss would be worse, and re-constructing the Driver outside the
  dispatcher reopens the affinity bug.
- **Mofka topic creation is racy across forked processes.** The dispatcher
  in the child may try to `create_topic` for a topic the parent already
  created; we swallow `"already exists"` errors and re-check
  `topic_exists`. Documented in `_MofkaDispatcher._loop`.
