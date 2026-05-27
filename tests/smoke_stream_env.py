"""Smoke-check the Diaspora Stream conda env on Polaris.

Not a pytest test (no test_ prefix, no assertions) — a sys.exit health check
for the eight imports / dlopens the stream workflow depends on. Run from
inside the activated env:

    python tests/smoke_stream_env.py

Exits 0 only if all 8 checks pass. On Polaris (glibc 2.38) all 8 should pass;
on Midway3 (glibc 2.28) the two diaspora_stream.api checks fail with
GLIBC_2.32 not found unless run inside a container.
"""

import ctypes
import os
import sys
import traceback
from pathlib import Path

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {name}" + (f" — {detail}" if detail else ""))


def check_import(name: str, attr: str | None = None) -> None:
    try:
        mod = __import__(name, fromlist=[attr] if attr else [])
        if attr is not None:
            getattr(mod, attr)
        record(name + (f".{attr}" if attr else ""), True, mod.__file__ or "<builtin>")
    except Exception as e:
        record(name + (f".{attr}" if attr else ""), False, f"{type(e).__name__}: {e}")


def check_liboctopus() -> None:
    # sys.prefix is the prefix of the python actually running this script,
    # which is what we want to test — distinct from $CONDA_PREFIX (which may
    # point at whichever env was active when this process was launched).
    prefix = sys.prefix
    candidates = list(Path(prefix, "lib").glob("liboctopus.so*"))
    if not candidates:
        record("liboctopus.so", False, f"no liboctopus.so* under {prefix}/lib")
        return
    target = candidates[0]
    try:
        ctypes.CDLL(str(target))
        record("liboctopus.so", True, str(target))
    except OSError as e:
        record("liboctopus.so", False, f"dlopen failed: {e}")


print(f"Python: {sys.version.split()[0]} ({sys.executable})")
print(f"sys.prefix: {sys.prefix}")
print()

check_import("diaspora_stream.api", "Driver")
check_import("mochi.mofka")
check_liboctopus()
check_import("diaspora_event_sdk", "Client")
check_import("colmena.queue.base")
check_import("torch")
check_import("ase")
check_import("mofa.diaspora")

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print()
print(f"Summary: {passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
