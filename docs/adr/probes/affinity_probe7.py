"""T9: is the T8 deadlock from concurrent CONSTRUCTION or N-engine STEADY STATE?

Same as T8 (no main engine, K self-contained per-thread engines) but engine +
producer + consumer construction is serialized behind a global lock. Operations
(push/pull) still run concurrently on each thread.

- T9 PASS  -> T8 hang was concurrent construction; a dispatcher-less design with
              a construction lock might be conceivable.
- T9 HANG  -> N independent engines deadlock in steady state; the single-thread
              dispatcher is the only viable shape.
"""
import json, sys, threading, time, queue
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 4
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STAMP = int(time.time())
out = queue.Queue()
build_lock = threading.Lock()
go = threading.Event()


def worker(k):
    topic = f"probe_t9_{STAMP}_{k}"
    step = "construct"
    try:
        with build_lock:                       # serialize ALL construction
            d = MofkaDriver(group_file=GROUP)
            if not d.topic_exists(topic):
                d.create_topic(name=topic)
                d.add_memory_partition(topic_name=topic, server_rank=0)
            th = d.open_topic(topic)
            prod = th.producer("p")
            cons = th.consumer("c")
        go.wait(timeout=15)                    # all threads operate concurrently
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
time.sleep(2.0)                                 # let serialized construction finish
go.set()
for t in threads:
    t.join(timeout=40)

res = {}
while not out.empty():
    k, status, detail = out.get(); res[k] = (status, detail)
alive = sum(t.is_alive() for t in threads)
oks = sum(1 for k in res if res[k][0] == "ok")
print(f"serialized-construct K={K}: {oks}/{K} ok, {alive} still alive", flush=True)
for k in range(K):
    s, d = res.get(k, ("HANG/no-report", None))
    print(f"  worker {k}: {s}" + (f"  | {d}" if d else ""), flush=True)
