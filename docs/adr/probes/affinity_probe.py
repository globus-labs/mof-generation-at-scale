"""Probe: does a per-thread / per-Driver design sidestep the mofka affinity bug
that ADR-0002 cites to justify the dispatcher?

Run against the live bedrock at /tmp/mofka-offthread (ofi+tcp://127.0.0.1).

Tests:
  T1 baseline   : Driver+producer+consumer built on main, all calls on main.
  T2 off-thread : Driver+producer+consumer built on main, push & pull from a
                  worker thread (the ADR "silent drop" case for a SHARED driver).
  T3 per-thread : worker-A builds its OWN Driver and pushes; worker-B builds its
                  OWN Driver and pulls. Both are plain threading.Thread (NOT the
                  main thread, NOT explicit Argobots ES). This is the proxystore
                  "one engine per role" pattern adapted to the current API.
"""
import json, sys, threading, time, queue
from diaspora_stream.api import Driver
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 5


def make_topic(name):
    d = Driver(backend="mofka", options={"group_file": GROUP})
    if not d.topic_exists(name):
        d.create_topic(name=name)
        admin = MofkaDriver(group_file=GROUP)
        admin.add_memory_partition(topic_name=name, server_rank=0)
    return d


def drain(consumer, want, deadline_s):
    got = []
    end = time.time() + deadline_s
    pending = consumer.pull()
    while len(got) < want and time.time() < end:
        ev = pending.wait(timeout_ms=500)
        if ev is not None and ev.event_id is not None:
            got.append(ev.metadata)
            ev.acknowledge()
            pending = consumer.pull()
    return got


def t1_baseline():
    topic = "probe_t1"
    d = make_topic(topic)
    th = d.open_topic(topic)
    prod = th.producer("p")
    cons = th.consumer("c")
    for i in range(N):
        prod.push(json.dumps({"i": i})).wait(timeout_ms=5000)
    got = drain(cons, N, 15)
    return f"T1 baseline (all on main): pushed {N}, got {len(got)} -> {'PASS' if len(got)==N else 'FAIL'}"


def t2_offthread_shared():
    topic = "probe_t2"
    d = make_topic(topic)
    th = d.open_topic(topic)
    prod = th.producer("p")          # built on MAIN
    cons = th.consumer("c")          # built on MAIN
    out = queue.Queue()

    def worker():
        try:
            for i in range(N):
                prod.push(json.dumps({"i": i})).wait(timeout_ms=5000)  # off-thread push
            got = drain(cons, N, 12)   # off-thread pull
            out.put(("ok", len(got)))
        except BaseException as e:
            out.put(("err", repr(e)))

    t = threading.Thread(target=worker); t.start(); t.join(timeout=40)
    if t.is_alive():
        return "T2 off-thread (shared Driver, calls on worker): HANG/timeout"
    kind, val = out.get()
    if kind == "err":
        return f"T2 off-thread (shared Driver, calls on worker): RAISED {val}"
    return f"T2 off-thread (shared Driver, calls on worker): got {val}/{N} -> {'PASS' if val==N else 'FAIL (silent drop)' }"


def t3_per_thread_drivers():
    topic = "probe_t3"
    make_topic(topic)               # ensure topic exists (main)
    out = queue.Queue()

    def producer_thread():
        try:
            d = Driver(backend="mofka", options={"group_file": GROUP})  # OWN driver on this plain thread
            th = d.open_topic(topic)
            prod = th.producer("p")
            for i in range(N):
                prod.push(json.dumps({"i": i})).wait(timeout_ms=5000)
            out.put(("prod_ok", None))
        except BaseException as e:
            out.put(("prod_err", repr(e)))

    def consumer_thread():
        try:
            d = Driver(backend="mofka", options={"group_file": GROUP})  # OWN driver on this plain thread
            th = d.open_topic(topic)
            cons = th.consumer("c")
            got = drain(cons, N, 20)
            out.put(("cons", len(got)))
        except BaseException as e:
            out.put(("cons_err", repr(e)))

    tc = threading.Thread(target=consumer_thread); tc.start()
    time.sleep(1.0)
    tp = threading.Thread(target=producer_thread); tp.start()
    tp.join(timeout=40); tc.join(timeout=40)

    msgs = {}
    while not out.empty():
        k, v = out.get(); msgs[k] = v
    alive = {"prod_alive": tp.is_alive(), "cons_alive": tc.is_alive()}
    return f"T3 per-thread Drivers (each plain thread owns its Driver): msgs={msgs} {alive}"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"t1": t1_baseline, "t2": t2_offthread_shared, "t3": t3_per_thread_drivers}
    if which == "all":
        for k in ("t1", "t2", "t3"):
            print(fns[k](), flush=True)
    else:
        print(fns[which](), flush=True)
