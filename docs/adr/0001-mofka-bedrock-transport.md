# ADR-0001: Use `ofi+tcp` (not `na+sm`) for bedrock on non-HPC hosts

## Status

Accepted (2026-05-27).

## Context

`mofa.diaspora.DiasporaQueues(stream_engine="mofka")` opens a thallium/margo
connection to a local `bedrock` daemon. Bedrock supports multiple transports:

- `na+sm` — shared-memory via `process_vm_writev`. Fastest single-host option;
  this is the Polaris-login-node default and what `envs/chameleon-stream.md` §4
  originally documented.
- `ofi+tcp` — libfabric over TCP. Slower than `na+sm` (loopback TCP, not shm).

Two failure modes pushed us off `na+sm`:

1. **Yama on cloud VMs.** `process_vm_writev` is blocked by Linux Yama
   (`kernel.yama.ptrace_scope >= 1`). Stock Ubuntu cloud images set this; Polaris
   login nodes happen to relax it. Switching to `na+sm` on cloud VMs requires
   `sysctl -w kernel.yama.ptrace_scope=0` — root, system-wide, persists past
   the workflow.
2. **Parsl-forked task server.** `na+sm` keys its `/dev/shm` segment by margo
   address. When the Parsl task server forks worker processes, the child
   inherits a half-initialised mochi state that contends with the parent on
   the same shm segment; the Driver hangs without error. Documented in the
   diaspora-debug2 commit `Wire run_parallel_workflow.py to the diaspora-stream
   mofka backend`.

`ofi+tcp` avoids both: TCP needs no `process_vm_writev`, and the parent/child
don't share a margo-shm segment.

## Decision

Bedrock listens on `ofi+tcp://127.0.0.1:<port>` for all non-Polaris targets.
`bin/start-bedrock.py` regenerates the `mofkactl` config with `-a "ofi+tcp"`,
patches flock `bootstrap` → `self`, patches yokan `database` →
`{"type": "map"}` (cloud-VM libstdc++ ABI mismatch), and launches bedrock with
the required `LD_PRELOAD` shims. The `127.0.0.1` bind is single-host only by
design — see ADR sequel if multi-node mofka becomes in-scope.

Polaris users may still prefer `na+sm` for throughput; the doc lists both as
valid choices but recommends `ofi+tcp` once the workflow involves a forked
task server.

## Consequences

- **Slower mofka throughput** vs `na+sm`. Loopback TCP serialises through the
  kernel; on a benchmark host this was ~30% lower throughput for the smoke
  workload. Acceptable for evaluation runs; revisit if benchmarking the queue
  itself.
- **Bound to loopback.** Multi-node mofka requires re-binding to a routable
  interface — out of scope for this ADR but the transport choice is already
  compatible with multi-host scaling (unlike `na+sm`, which never can be).
- **Multi-homed hosts can still mis-route `ofi+tcp`.** Even bound to
  `127.0.0.1`, libfabric's `tcp` provider may advertise/respond on a
  non-loopback NIC when the host has extra interfaces (e.g. docker bridges); the
  daemon then aborts with `HG_TIMEOUT` and clients hang — a transport artifact,
  not a code bug. Observed 2026-05-29 on a VM with docker bridges (a standalone
  bedrock tried to respond to a `172.x` bridge address and aborted, while the
  byte-identical mofka stack worked fine inside containers). Fix: pin libfabric
  to loopback (e.g. `FI_TCP_IFACE` / `FI_SOCKETS_IFACE`) or run bedrock and its
  clients in one network namespace — which is what the containerised
  thinker/server deployment does (shared netns → only `lo` + `eth0`).
- **Yama sysctl no longer required.** The cloud-VM provisioning runbook is
  shorter; no root step.
- **The §8 row recommending `sysctl ptrace_scope=0`** in
  [envs/chameleon-stream.md](../../envs/chameleon-stream.md) is now misleading and
  should be removed when this ADR lands.
