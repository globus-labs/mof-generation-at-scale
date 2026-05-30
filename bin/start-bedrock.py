#!/usr/bin/env python
"""Launch a local bedrock daemon for mofka on a non-HPC host.

One artifact replaces the §4 copy-paste sequence in envs/chameleon-stream.md.
Does the regen → unwrap → patch flock/yokan → launch dance, including the
cloud-VM-specific workarounds.

Invariants encoded here (see [docs/adr/0001-mofka-bedrock-transport.md]):

  * transport is ``ofi+tcp://127.0.0.1`` — single-host but Parsl-fork-safe.
    ``na+sm`` is rejected; the doc has the rationale.
  * flock provider's ``bootstrap`` is switched from ``mpi`` to ``self`` so
    bedrock works as a single-process daemon.
  * yokan provider databases are patched to ``{"type":"map"}`` (in-memory)
    when BEDROCK_YOKAN_MAP=1 (default on cloud VMs — the prebuilt lmdb/
    rocksdb backends reject mofka's put_direct mode on non-Polaris
    libstdc++ ABIs). Set to 0 once mochi-conda-packages fixes that upstream.
  * LD_PRELOAD pulls in lmdb + leveldb that libyokan-server.so dlopens.

Usage:

    python bin/start-bedrock.py /tmp/mofka-test

Writes bedrock-config-single.json + bedrock.log into the directory and
prints the group file path on success (suitable for piping into
``MOFKA_GROUP_FILE=$(python bin/start-bedrock.py /tmp/mofka-test)``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


TRANSPORT = "ofi+tcp"
TRANSPORT_LISTEN = "ofi+tcp://127.0.0.1"


def _conda_prefix() -> Path:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        sys.exit("CONDA_PREFIX not set — activate the mofa-stream env first")
    return Path(prefix)


def _generate_and_patch(workdir: Path, yokan_map: bool) -> Path:
    """Generate bedrock-config-single.json in workdir; return its path."""
    raw = workdir / "bedrock-config.json"
    out = workdir / "bedrock-config-single.json"

    # mofkactl emits a JSON array; bedrock CLI wants a single object.
    subprocess.run(
        [
            sys.executable, "-m", "mochi.mofka.mofkactl", "config", "generate",
            "-a", TRANSPORT,
            "--master-db-path-prefixes", str(workdir),
            "--metadata-db-path-prefixes", str(workdir),
            "--data-storage-path-prefixes", str(workdir),
        ],
        stdout=raw.open("w"),
        check=True,
    )

    cfgs = json.loads(raw.read_text())
    cfg = cfgs[0]

    for p in cfg["providers"]:
        if p["type"] == "flock":
            # Single-process daemon → no MPI peer to bootstrap from.
            p["config"]["bootstrap"] = "self"
        if yokan_map and p["type"] == "yokan":
            # In-memory map: the lmdb/rocksdb/leveldb backends in
            # mochi-conda-packages 2026-03-30 reject mofka's put_direct mode
            # on non-Polaris libstdc++ ABIs (§8 of envs/chameleon-stream.md).
            # Map is fine because DiasporaQueues data is ephemeral by design.
            # Newer mofka schemas use `databases` (array); 0.8.x uses
            # `database` (single dict) — handle both.
            yk = p.get("config", {})
            if "database" in yk:
                yk["database"] = {"type": "map", "config": {}}
            for db in yk.get("databases", []):
                db["type"] = "map"
                db["config"] = {}

    out.write_text(json.dumps(cfg, indent=2))
    return out


def _launch(workdir: Path, cfg_path: Path) -> subprocess.Popen:
    prefix = _conda_prefix()
    lib = prefix / "lib"

    env = os.environ.copy()
    env["LD_PRELOAD"] = ":".join([
        str(lib / "liblmdb.so"),
        str(lib / "libleveldb.so"),
    ] + ([env["LD_PRELOAD"]] if env.get("LD_PRELOAD") else []))
    env["LD_LIBRARY_PATH"] = str(lib) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    log = workdir / "bedrock.log"
    proc = subprocess.Popen(
        [str(prefix / "bin" / "bedrock"), TRANSPORT_LISTEN, "-c", str(cfg_path)],
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(workdir),
    )
    return proc


def _wait_for_ready(workdir: Path, proc: subprocess.Popen, timeout_s: float = 20.0) -> Path:
    """Block until bedrock writes the group file; return its path or exit."""
    group_file = workdir / "mofka.flock.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(
                f"bedrock exited early (rc={proc.returncode}); "
                f"see {workdir/'bedrock.log'}"
            )
        if group_file.exists():
            return group_file
        time.sleep(0.2)
    sys.exit(f"bedrock did not produce {group_file} within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a local bedrock daemon for mofka.")
    parser.add_argument("workdir", type=Path, help="Working directory for bedrock state.")
    args = parser.parse_args()

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    yokan_map = os.environ.get("BEDROCK_YOKAN_MAP", "1") == "1"
    cfg = _generate_and_patch(workdir, yokan_map=yokan_map)
    proc = _launch(workdir, cfg)
    group_file = _wait_for_ready(workdir, proc)

    # The pid is the only handle the caller has to tear it down.
    print(str(group_file))
    print(f"# bedrock pid={proc.pid} log={workdir/'bedrock.log'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
