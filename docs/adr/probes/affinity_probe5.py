"""T7: can a per-thread-engine design replace the dispatcher if each engine
runs its OWN progress thread?

Scenario mirrors Colmena: MAIN sets up the topic, holds an idle engine, then
blocks (join) doing nothing. A producer thread and a consumer thread EACH own a
MofkaDriver. With progress=1 each calls start_progress_thread() on its engine.

progress=0 -> control (expect hang/drop, like T6-hold)
progress=1 -> each engine self-progresses (does it now work despite idle main?)
"""
import json, sys, threading, time, queue
from mochi.mofka.client import MofkaDriver

import os
GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")
N = 5
progress = sys.argv[1] == "1"
main_holds = (sys.argv[2] if len(sys.argv) > 2 else "hold") == "hold"
TOPIC = f"probe_t7_{int(progress)}_{int(time.time())}"
out = queue.Queue()

boot = MofkaDriver(group_file=GROUP)
if not boot.topic_exists(TOPIC):
    boot.create_topic(name=TOPIC)
    boot.add_memory_partition(topic_name=TOPIC, server_rank=0)
# MAIN either keeps an idle engine alive (like Colmena's main thread) or drops it.
import gc
if main_holds:
    held = boot
else:
    del boot; gc.collect(); held = None


def maybe_progress(d, who):
    if progress:
        try:
            d.start_progress_thread(); return f"{who}:progress_on"
        except Exception as e:
            return f"{who}:progress_ERR({e!r})"
    return f"{who}:no_progress"


def producer_thread():
    step = "create_driver"
    try:
        d = MofkaDriver(group_file=GROUP); sp = maybe_progress(d, "prod")
        step = "open_topic"; th = d.open_topic(TOPIC)
        step = "producer";   prod = th.producer("p")
        step = "push"
        for i in range(N):
            prod.push(json.dumps({"i": i})).wait(timeout_ms=4000)
        out.put(("prod", "ok", sp))
    except BaseException as e:
        out.put(("prod", f"FAIL@{step}", repr(e)))


def consumer_thread():
    step = "create_driver"
    try:
        d = MofkaDriver(group_file=GROUP); sp = maybe_progress(d, "cons")
        step = "open_topic"; th = d.open_topic(TOPIC)
        step = "consumer";   cons = th.consumer("c")
        step = "pull"
        got = []; end = time.time() + 14; pending = cons.pull()
        while len(got) < N and time.time() < end:
            ev = pending.wait(timeout_ms=500)
            if ev is not None and ev.event_id is not None:
                got.append(ev.metadata); ev.acknowledge(); pending = cons.pull()
        out.put(("cons", "ok" if len(got) == N else "FAIL@pull", f"{sp} got {len(got)}/{N}"))
    except BaseException as e:
        out.put(("cons", f"FAIL@{step}", repr(e)))


tc = threading.Thread(target=consumer_thread); tc.start(); time.sleep(1.0)
tp = threading.Thread(target=producer_thread); tp.start()
tp.join(timeout=30); tc.join(timeout=30)   # MAIN blocks here, idle, holding `held`

msgs = []
while not out.empty():
    msgs.append(out.get())
for who, status, detail in sorted(msgs):
    print(f"  {who}: {status}" + (f"  | {detail}" if detail else ""), flush=True)
print(f"  (prod_alive={tp.is_alive()} cons_alive={tc.is_alive()})", flush=True)
