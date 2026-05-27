# Project context

Glossary for the MOFA workflow and its streaming queue backends. This file is
a glossary, not a spec or scratchpad — implementation details live in code and
ADRs.

## MOFA application

The full workflow launched by [run_parallel_workflow.py](run_parallel_workflow.py):
DiffLinker generator → ligand assembly → LAMMPS MD → CP2K DFT → RASPA GCMC,
orchestrated by a Colmena `MOFAThinker` (≈11 agent threads) on top of a Parsl
task server (forks worker pools). State lives in MongoDB; large payloads
go through a ProxyStore-on-Redis store.

## Streaming queue backends — `DiasporaQueues.stream_engine`

`DiasporaQueues` is MOFA's `ColmenaQueues` subclass. The `stream_engine`
parameter picks the transport:

- **`files`** — on-disk per-topic logs, no daemon. Used for local development.
- **`mofka`** — [mochi-hpc/mofka](https://github.com/mochi-hpc/mofka) over a
  local `bedrock` daemon (margo / thallium / Argobots). Not thread-safe in the
  obvious Python sense (see *Mofka dispatcher*).
- **`octopus`** — Diaspora Stream's `octopus` driver, which is librdkafka
  pointed at AWS MSK Kafka, authenticated with Globus + AWS-MSK-IAM. Thread-safe.

## Mofka dispatcher

A dedicated Python thread that owns the mofka `Driver`, topics, producers,
and consumers. Required because mochi-mofka uses Argobots execution streams
under thallium; calling Driver methods from any other thread aborts with
`ABT_ERR_INV_XSTREAM` or deadlocks silently. Implemented in
`mofa/diaspora.py` via a `queue.Queue` of work items plus
`os.register_at_fork(after_in_child=…)` so the dispatcher is rebuilt cleanly
in Parsl-forked task-server children.

## Bedrock daemon

A mochi services container that hosts a mofka provider plus yokan storage.
Single process, single host. Configured by `mofkactl config generate` then
hand-patched (flock provider → `self`, JSON array unwrap → object).
Transport choice matters: `na+sm` (shared-memory) is faster but blocked by
Linux Yama on stock cloud images and contends with Parsl-forked children on
the `/dev/shm` margo segment; `ofi+tcp` avoids both issues.

## Cloud VM host

The Chameleon Cloud Ubuntu 24.04 VM this environment is built and validated
on. Treated as an "HPC login node" with no scheduler:
`run_parallel_workflow.py` runs as a plain process and Parsl uses
`LocalProvider` (single block, no qsub). Both `mofka` and `octopus`
backends must work here. The deployment guide is
[envs/chameleon-stream.md](envs/chameleon-stream.md) and the env yml is
[envs/environment-chameleon-stream.yml](envs/environment-chameleon-stream.yml).

Distinct from a Polaris login node in four ways:

1. No 128-PID cgroup cap. The §0 thread caps in
   [envs/chameleon-stream.md](envs/chameleon-stream.md) are still
   nice-to-have (BLAS still fights librdkafka for thread budget) but no
   longer load-bearing.
2. No NVIDIA GPU — every task runs on CPU. Generation and MD are slow but
   functional. Selected by `CloudVMConfig` rather than `LocalConfig`.
3. Yama `kernel.yama.ptrace_scope >= 1` by default + cloud libstdc++ ABI
   mismatch with mochi-conda-packages → bedrock must use `ofi+tcp` (not
   `na+sm`) and yokan backends must be the in-memory `{"type":"map"}` form.
4. Conda-forge LAMMPS only ships `pair_style mliap`. `CloudVMConfig`
   defaults to `lammps_pkg='ml-mace'` (consuming `mace-mp0_medium-lammps.pt`
   via `pair_style mace`), which requires a source-build from
   ACEsuit/lammps — recipe in [envs/chameleon-stream.md](envs/chameleon-stream.md)
   §9 "LAMMPS source build".
