#!/usr/bin/env python
"""Summarize DiasporaQueues benchmark traces.

Reads the per-process JSON-line trace files written by
``mofa.diaspora.DiasporaQueues`` (see ``log_benchmark``) and prints
per-operation latency / throughput statistics.

Usage:
    bin/analyze-bench.py <trace_dir_or_glob> [<more> ...]

Each argument may be a directory (searched recursively for ``*trace*.log``),
a glob, or a file. Durations are reported in microseconds.
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def _iter_files(arg):
    if os.path.isdir(arg):
        yield from glob.glob(os.path.join(arg, "**", "*trace*.log"), recursive=True)
    else:
        matches = glob.glob(arg, recursive=True)
        if matches:
            yield from matches
        elif os.path.exists(arg):
            yield arg


def load(args):
    records = []
    files = []
    for a in args:
        for f in _iter_files(a):
            files.append(f)
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # a SIGKILL mid-write can truncate the final line
                        pass
    return records, sorted(set(files))


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt_us(ns):
    return f"{ns / 1e3:,.1f}"


def summarize(records, group_queue=False):
    buckets = defaultdict(list)
    sizes = defaultdict(list)
    for r in records:
        key = r["task_name"]
        if group_queue and r.get("queue"):
            key = f"{key} [{r['queue']}]"
        buckets[key].append(r["duration_ns"])
        if r.get("msg_size"):
            sizes[key].append(r["msg_size"])

    print("\t".join(["operation", "count", "mean", "p50", "p90",
                      "p99", "max", "avg_bytes"]))
    for key in sorted(buckets, key=lambda k: -sum(buckets[k])):
        ds = buckets[key]
        sz = sizes.get(key, [])
        print("\t".join([
            key,
            str(len(ds)),
            fmt_us(statistics.mean(ds)),
            fmt_us(pct(ds, 0.50)),
            fmt_us(pct(ds, 0.90)),
            fmt_us(pct(ds, 0.99)),
            fmt_us(max(ds)),
            f"{(statistics.mean(sz) if sz else 0):,.0f}",
        ]))
    print("\n(all latencies in microseconds; avg_bytes = mean payload size)")


def throughput(records):
    """End-to-end message throughput from wall_ns span, per send/recv op."""
    print("\nthroughput (wall-clock span of the trace):")
    print("\t".join(["op", "msgs", "span_min", "msg_per_s", "MB_total"]))
    for op in ("SEND_MSG", "GET_MSG"):
        rs = [r for r in records if r["task_name"] == op]
        if not rs:
            continue
        walls = [r["wall_ns"] for r in rs if "wall_ns" in r]
        if len(walls) < 2:
            continue
        span_s = (max(walls) - min(walls)) / 1e9
        rate = len(rs) / span_s if span_s > 0 else 0.0
        tot_bytes = sum(r.get("msg_size", 0) for r in rs)
        print("\t".join([
            op,
            str(len(rs)),
            f"{span_s/60:.1f}",
            f"{rate:.2f}",
            f"{tot_bytes/1e6:.2f}",
        ]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="trace dirs / globs / files")
    ap.add_argument("--by-queue", action="store_true", help="break down per queue too")
    args = ap.parse_args()

    records, files = load(args.paths)
    if not records:
        print("no benchmark records found", file=sys.stderr)
        sys.exit(1)

    pids = sorted({r.get("pid") for r in records})
    engines = sorted({r.get("engine") for r in records if r.get("engine")})
    print(f"trace files : {len(files)}")
    for f in files:
        print(f"  - {f}")
    print(f"records     : {len(records):,}")
    print(f"engine(s)   : {', '.join(map(str, engines))}")
    print(f"process(es) : {len(pids)} pids {pids}\n")

    summarize(records, group_queue=args.by_queue)
    throughput(records)


if __name__ == "__main__":
    main()
