"""T5: is an Argobots ES per-engine or per-process?

Two PLAIN threads, each builds its OWN MofkaDriver (its own engine via JSON).
No driver on the main thread. Thread A creates topic+partition+producer and
pushes; thread B builds a SECOND driver and a consumer and pulls.

- If ES is per-ENGINE: both threads work independently -> the old per-object
  design is replicable, dispatcher avoidable.
- If ES is per-PROCESS: the first thread to init margo claims the only ES;
  the second thread aborts when constructing mofka objects -> dispatcher needed.

Each step in thread B is labelled so we see exactly where it fails.
"""
import json, threading, time, queue
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 5
TOPIC = f"probe_t5_{int(time.time())}"
out = queue.Queue()
topic_ready = threading.Event()


def thread_a():
    step = "create_driver"
    try:
        d = MofkaDriver(group_file=GROUP)          # FIRST margo init -> A claims primary ES?
        step = "create_topic"
        if not d.topic_exists(TOPIC):
            d.create_topic(name=TOPIC)
            d.add_memory_partition(topic_name=TOPIC, server_rank=0)
        step = "open_topic"
        th = d.open_topic(TOPIC)
        step = "producer"
        prod = th.producer("p")
        topic_ready.set()
        step = "push"
        for i in range(N):
            prod.push(json.dumps({"i": i})).wait(timeout_ms=4000)
        out.put(("A", "ok", None))
    except BaseException as e:
        topic_ready.set()
        out.put(("A", f"FAIL@{step}", repr(e)))


def thread_b():
    topic_ready.wait(timeout=20)
    time.sleep(1.0)
    step = "create_driver"
    try:
        d = MofkaDriver(group_file=GROUP)          # SECOND engine, on thread B
        step = "open_topic"
        th = d.open_topic(TOPIC)
        step = "consumer"
        cons = th.consumer("c")                    # <-- ADR says this aborts off-ES
        step = "pull"
        got = []
        end = time.time() + 12
        pending = cons.pull()
        while len(got) < N and time.time() < end:
            ev = pending.wait(timeout_ms=500)
            if ev is not None and ev.event_id is not None:
                got.append(ev.metadata); ev.acknowledge(); pending = cons.pull()
        out.put(("B", "ok", f"got {len(got)}/{N}"))
    except BaseException as e:
        out.put(("B", f"FAIL@{step}", repr(e)))


ta = threading.Thread(target=thread_a)
tb = threading.Thread(target=thread_b)
tb.start(); ta.start()
ta.join(timeout=40); tb.join(timeout=40)

msgs = []
while not out.empty():
    msgs.append(out.get())
for who, status, detail in sorted(msgs):
    print(f"thread {who}: {status}" + (f"  | {detail}" if detail else ""), flush=True)
print(f"(A_alive={ta.is_alive()} B_alive={tb.is_alive()})", flush=True)
