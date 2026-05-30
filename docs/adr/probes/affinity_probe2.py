"""T4: does configuring the engine (margo progress thread) let off-thread
mofka calls work? If yes, the dispatcher could be replaced by engine config.

Builds a mofka Driver whose internal engine has a dedicated progress thread,
calls start_progress_thread(), then constructs producer/consumer on the MAIN
thread and pushes/pulls from a WORKER thread (the ADR silent-drop case).
"""
import json, sys, threading, time, queue
from diaspora_stream.api import Driver
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 5


def build_driver(use_progress):
    opts = {"group_file": GROUP}
    if use_progress:
        # configure the internally-created engine to use a progress thread
        opts["margo"] = {"use_progress_thread": True}
    return Driver(backend="mofka", options=opts)


def ensure_topic(name):
    d = MofkaDriver(group_file=GROUP)
    if not d.topic_exists(name):
        d.create_topic(name=name)
        d.add_memory_partition(topic_name=name, server_rank=0)


def t4(use_progress, call_start_progress):
    topic = f"probe_t4_{int(use_progress)}{int(call_start_progress)}"
    ensure_topic(topic)
    d = build_driver(use_progress)
    if call_start_progress:
        # MofkaDriver exposes start_progress_thread(); the diaspora Driver may not
        try:
            d.start_progress_thread()
            sp = "start_progress_thread:ok"
        except Exception as e:
            sp = f"start_progress_thread:ERR({e!r})"
    else:
        sp = "start_progress_thread:skipped"

    th = d.open_topic(topic)
    prod = th.producer("p")   # built on MAIN
    cons = th.consumer("c")   # built on MAIN
    out = queue.Queue()

    def worker():
        try:
            for i in range(N):
                prod.push(json.dumps({"i": i})).wait(timeout_ms=4000)
            got = []
            end = time.time() + 12
            pending = cons.pull()
            while len(got) < N and time.time() < end:
                ev = pending.wait(timeout_ms=500)
                if ev is not None and ev.event_id is not None:
                    got.append(ev.metadata); ev.acknowledge(); pending = cons.pull()
            out.put(("ok", len(got)))
        except BaseException as e:
            out.put(("err", repr(e)))

    t = threading.Thread(target=worker); t.start(); t.join(timeout=35)
    if t.is_alive():
        return f"T4 use_progress={use_progress} {sp}: HANG"
    kind, val = out.get()
    if kind == "err":
        return f"T4 use_progress={use_progress} {sp}: RAISED {val}"
    verdict = "PASS (off-thread works!)" if val == N else "FAIL (silent drop)"
    return f"T4 use_progress={use_progress} {sp}: got {val}/{N} -> {verdict}"


if __name__ == "__main__":
    up = sys.argv[1] == "1"
    csp = sys.argv[2] == "1" if len(sys.argv) > 2 else False
    print(t4(up, csp), flush=True)
