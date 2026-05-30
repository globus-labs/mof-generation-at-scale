"""T6: what exactly determines whether a worker thread's OWN engine works?

Variable: what the MAIN thread does with margo before the worker runs.
  mode=idle  : main never touches mofka; worker builds its own driver+producer.
  mode=hold  : main builds a driver and KEEPS it alive; worker builds its own.
  mode=drop  : main builds a driver then deletes it; worker builds its own.

Worker always: own MofkaDriver -> open_topic -> producer -> push N (same thread).
Reports the exact step where the worker fails, if any.
"""
import json, sys, threading, time, queue, gc
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 3
mode = sys.argv[1]
TOPIC = f"probe_t6_{mode}_{int(time.time())}"
out = queue.Queue()

# make sure the topic exists using a throwaway driver on MAIN regardless of mode
boot = MofkaDriver(group_file=GROUP)
if not boot.topic_exists(TOPIC):
    boot.create_topic(name=TOPIC)
    boot.add_memory_partition(topic_name=TOPIC, server_rank=0)

held = None
if mode == "hold":
    held = MofkaDriver(group_file=GROUP)          # main keeps an engine alive
elif mode == "drop":
    tmp = MofkaDriver(group_file=GROUP); del tmp; gc.collect()
elif mode == "idle":
    pass
# NOTE: 'boot' itself is a main-thread engine; keep a separate flag to also drop it
if mode == "idle":
    del boot; gc.collect()


def worker():
    step = "create_driver"
    try:
        d = MofkaDriver(group_file=GROUP)
        step = "open_topic"; th = d.open_topic(TOPIC)
        step = "producer";   prod = th.producer("p")
        step = "push"
        for i in range(N):
            prod.push(json.dumps({"i": i})).wait(timeout_ms=4000)
        out.put(("ok", None))
    except BaseException as e:
        out.put((f"FAIL@{step}", repr(e)))


t = threading.Thread(target=worker); t.start(); t.join(timeout=30)
status, detail = (out.get() if not out.empty() else ("HANG", None))
print(f"mode={mode}: worker {status}" + (f"  | {detail}" if detail else ""), flush=True)
