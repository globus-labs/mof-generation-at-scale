"""mofka + parsl probe for ADR-0002: mofka objects are bound to the thread that
built the Driver (the Argobots/margo "anchor"), so the parsl-forked doer — which
touches mofka from parsl's own threads — can't use them.

This anchors margo on MAIN (a control push proves the broker is reachable), then
has a parsl ThreadPoolExecutor app touch mofka on a worker thread two ways:

  VARIANT 1  construct off-anchor: worker builds topic.consumer("c") itself
             -> aborts ABT_ERR_INV_XSTREAM (order-dependent; see the inline note)
  VARIANT 2  use off-anchor: MAIN builds the consumer, worker only pull()s it
             -> SILENT DROP (no error, 0 events) or whole-process HANG

So the only thing that works is one mofka thread — the dispatcher. (You can't
give the worker its own engine either: the sole constructor is
MofkaDriver(arg0: json), so a pre-built pymargo engine can't be injected.)

Needs a LIVE bedrock (a stale mofka.flock.json points at a dead port and just
hangs). VARIANT 2 can deadlock — a blocked mofka C call holds the GIL — so run
under an outer guard and read a missing line as the hang. FI_TCP_IFACE=lo pins
libfabric to loopback on a multi-homed host (docker bridges) — without it the
client may answer on a bridge NIC and the Driver build hangs (see ../README.md):

    PY=~/conda-envs/mofa-stream/bin/python
    P=~/conda-envs/mofa-stream/lib

    export MOFKA_GROUP_FILE=$(CONDA_PREFIX=~/conda-envs/mofa-stream FI_TCP_IFACE=lo \
        $PY bin/start-bedrock.py /tmp/mofka-probe)
    FI_TCP_IFACE=lo LD_PRELOAD="$P/liblmdb.so:$P/libleveldb.so" LD_LIBRARY_PATH=$P \
        timeout 120 $PY docs/adr/probes/affinity_probe_parsl.py

    pkill -f "mofka-probe/bedrock-config"   # stop the bedrock when done
"""
import json
import os

GROUP = os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-offthread/mofka.flock.json")


def _step(msg):
    """Flushed progress marker so a hang shows the last step reached."""
    print(f"... {msg}", flush=True)


def _make_topic(name):
    """Build a Driver on the CALLING thread (this becomes the margo anchor),
    create the topic + a memory partition if needed, and return the Driver."""
    from diaspora_stream.api import Driver
    from mochi.mofka.client import MofkaDriver

    d = Driver(backend="mofka", options={"group_file": GROUP})
    if not d.topic_exists(name):
        d.create_topic(name=name)
        MofkaDriver(group_file=GROUP).add_memory_partition(topic_name=name, server_rank=0)
    return d


def _drain(consumer, want, deadline_s=8):
    """Pull up to `want` events; return how many arrived. Same loop on the
    anchor (works) and inside the off-anchor worker (silently returns 0)."""
    import time

    got = 0
    pending = consumer.pull()
    end = time.time() + deadline_s
    while got < want and time.time() < end:
        ev = pending.wait(timeout_ms=500)
        if ev is not None and ev.event_id is not None:
            got += 1
            ev.acknowledge()
            pending = consumer.pull()
    return got


def run():
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors.threads import ThreadPoolExecutor

    if not os.path.exists(GROUP):
        raise SystemExit(f"MOFKA_GROUP_FILE={GROUP} does not exist — start a "
                         f"bedrock (bin/start-bedrock.py) and point at its flock file")

    _step("loading parsl ThreadPoolExecutor")
    parsl.load(Config(executors=[ThreadPoolExecutor(max_threads=2, label="t")]))

    # Everything mofka is built on MAIN — MAIN is the Argobots anchor.
    _step(f"building mofka Driver from {GROUP} "
          f"(HANGS HERE if no bedrock is live at the address inside that file)")
    d = _make_topic("probe_parsl")
    _step("opening topic + building producer on MAIN (the anchor)")
    th = d.open_topic("probe_parsl")
    prod = th.producer("p")
    _step("pushing 3 events on the anchor (main) thread")
    for i in range(3):
        prod.push(json.dumps({"i": i})).wait(timeout_ms=5000)
    print("CONTROL: pushed 3 on the anchor thread -> OK "
          "(broker reachable, affinity satisfied)", flush=True)

    # ---- VARIANT 1: worker CONSTRUCTS the consumer (off-anchor) ------------
    @python_app
    def construct_off_anchor(topic_handle):
        # Builds a consumer on a parsl WORKER THREAD. ORDER-DEPENDENT: whichever
        # thread first inits margo is the anchor. Here MAIN got there first, so
        # this aborts ABT_ERR_INV_XSTREAM; in a forked HTEx worker (fresh
        # process) the same call may instead succeed or deadlock — don't rely
        # on it raising.
        consumer = topic_handle.consumer("c")        # <-- off-anchor construction
        ev = consumer.pull().wait(timeout_ms=2000)
        return None if ev is None else ev.event_id

    _step("VARIANT 1: parsl worker constructing a consumer off-anchor")
    try:
        r = construct_off_anchor(th).result(timeout=30)
        print(f"VARIANT 1 construct off-anchor: SURVIVED -> {r} "
              f"(margo anchored on the worker this run)", flush=True)
    except Exception as e:  # noqa: BLE001
        print("VARIANT 1 construct off-anchor: FAILED -> "
              f"{type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)

    # ---- VARIANT 2: MAIN builds the consumer, worker only USES it ----------
    cons = th.consumer("c2")                                    # built on MAIN

    @python_app
    def use_off_anchor(consumer, want):
        # Pulls on a parsl WORKER THREAD using a consumer built on the anchor.
        # MEASURED: returns nothing and raises nothing -> SILENT DROP (got 0),
        # though the events are in the broker; ~1/3 of runs instead HANG the
        # whole process (blocked C call holds the GIL). The dangerous case.
        return _drain(consumer, want)

    _step("VARIANT 2: parsl worker pulling off-anchor "
          "(may silently drop, or HANG here — watch the outer timeout)")
    try:
        got = use_off_anchor(cons, 3).result(timeout=30)
        verdict = "SILENT DROP — no error, events never arrived" if got == 0 \
            else ("PASS" if got == 3 else f"PARTIAL {got}/3")
        print(f"VARIANT 2 use off-anchor: worker got {got}/3 -> {verdict}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"VARIANT 2 use off-anchor: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]}", flush=True)

    # Proof the events were there all along: pull on MAIN (the anchor).
    print(f"CONTROL: same consumer drained on MAIN afterwards -> {_drain(cons, 3)}/3 "
          f"(confirms VARIANT 2 was a silent drop, not an empty broker)", flush=True)

    try:
        parsl.dfk().cleanup()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    run()
    # os._exit: the mofka Driver's C++ destructors hang on a clean interpreter exit.
    os._exit(0)
