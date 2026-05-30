"""T8: does the per-thread-own-engine pattern scale to the real thinker?

Mirrors the dispatcher-less ("resemble old code") design: each agent thread
lazily builds its OWN engine and its OWN producer/consumer, used only on that
thread. K such threads run concurrently and long-lived.

  main_engine=0 : main holds NO mofka engine (faithful thinker; agent threads
                  touch mofka, main only spawns+joins). Each thread creates its
                  own topic. -> does per-thread engine scale past 2 (cf. T5)?
  main_engine=1 : main builds an engine, pre-creates topics, HOLDS it, blocks
                  in join (e.g. if connect() ran on main). -> expect hang/abort.

Each thread is fully self-contained: own driver -> own topic -> producer push N
-> consumer pull N, all on its own thread.
"""
import json, sys, threading, time, queue, gc
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 4
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
main_engine = (len(sys.argv) > 2 and sys.argv[2] == "1")
STAMP = int(time.time())
out = queue.Queue()
start_gate = threading.Event()


def ensure(d, name):
    if not d.topic_exists(name):
        d.create_topic(name=name)
        d.add_memory_partition(topic_name=name, server_rank=0)


held = None
if main_engine:
    held = MofkaDriver(group_file=GROUP)
    for k in range(K):
        ensure(held, f"probe_t8_{STAMP}_{k}")


def worker(k):
    topic = f"probe_t8_{STAMP}_{k}"
    step = "create_driver"
    start_gate.wait(timeout=10)
    try:
        d = MofkaDriver(group_file=GROUP)
        if not main_engine:
            step = "ensure_topic"; ensure(d, topic)
        step = "open_topic"; th = d.open_topic(topic)
        step = "producer";   prod = th.producer("p")
        step = "consumer";   cons = th.consumer("c")
        step = "push"
        for i in range(N):
            prod.push(json.dumps({"k": k, "i": i})).wait(timeout_ms=4000)
        step = "pull"
        got = []; end = time.time() + 12; pending = cons.pull()
        while len(got) < N and time.time() < end:
            ev = pending.wait(timeout_ms=400)
            if ev is not None and ev.event_id is not None:
                got.append(ev.metadata); ev.acknowledge(); pending = cons.pull()
        out.put((k, "ok" if len(got) == N else f"FAIL@pull({len(got)}/{N})", None))
    except BaseException as e:
        out.put((k, f"FAIL@{step}", repr(e)[:90]))


threads = [threading.Thread(target=worker, args=(k,)) for k in range(K)]
for t in threads:
    t.start()
start_gate.set()                       # release all workers together
for t in threads:
    t.join(timeout=40)

res = {}
while not out.empty():
    k, status, detail = out.get(); res[k] = (status, detail)
alive = sum(t.is_alive() for t in threads)
oks = sum(1 for k in res if res[k][0] == "ok")
print(f"main_engine={int(main_engine)} K={K}: {oks}/{K} ok, {alive} threads still alive", flush=True)
for k in range(K):
    s, d = res.get(k, ("HANG/no-report", None))
    print(f"  worker {k}: {s}" + (f"  | {d}" if d else ""), flush=True)
