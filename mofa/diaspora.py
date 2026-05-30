"""Diaspora Stream implementation of ColmenaQueues.

Supports three backends via ``stream_engine``:

- ``files`` — on-disk per-topic logs, no daemon. Default; pthread-safe.
- ``octopus`` — AWS MSK Kafka via Globus tokens at ``~/.diaspora/storage.db``;
  pthread-safe through librdkafka.
- ``mofka`` — mochi-mofka via a local bedrock daemon (see
  ``envs/chameleon-stream.md`` §2). Not pthread-safe — see below.

Why mofka mode is non-trivial. mochi-mofka uses Argobots under thallium, and
mofka objects are bound to the Python thread that constructed the ``Driver``.
Calls from any other thread fail in operation-specific ways (measured against
mofka 0.8.2 on 2026-05-27): constructing a producer/consumer aborts with
``ABT_ERR_INV_XSTREAM``; ``Driver`` metadata calls (``topic_exists`` /
``open_topic``) deadlock; ``consumer.pull()`` silently returns nothing — no
error, the worst case; only an already-constructed ``producer.push().wait()``
happens to work. The owning thread need not be the main thread, just a single
consistent one. The implementation funnels all mofka work onto one such thread
with four cooperating pieces:

1. A dedicated ``_MofkaDispatcher`` thread owns the Driver, topics, producers,
   and consumers. Colmena's ~11 MOFAThinker agent threads plus the parsl-forked
   task server funnel every push/pull through a ``queue.Queue``.
2. ``os.register_at_fork(after_in_child=…)`` drops the inherited dispatcher in
   the child so each process builds its own Argobots state post-fork.
   Dispatcher construction is also deferred to first use for the same reason.
3. ``_get_message(timeout=None)`` polls forever (1 s granularity). Colmena's
   ``BaseTaskServer.listen_and_launch`` interprets ``TimeoutException`` from
   ``get_task(None)`` as a kill signal, so a hard budget on ``None`` would shut
   the task server down after one task.
4. The mofka send path skips ``producer.flush()`` (mofka 0.8.2's
   ``flush().wait()`` always blocks the full timeout even with nothing
   batched). ``push().wait()`` returning means the broker acked. The
   files/octopus paths keep ``flush()`` because they need it for persistence.

Bedrock must use a TCP-class transport (``ofi+tcp://<ip>``) — with ``na+sm``,
the parsl-forked child contends with the parent on a ``/dev/shm`` segment
keyed by margo address and the rebuilt Driver hangs. The launcher scripts
([../run-mofa-stream-common.sh](../run-mofa-stream-common.sh)) handle this.

Run as an end-to-end Colmena round-trip. For mofka, first start bedrock
(see ``envs/chameleon-stream.md`` §2) and export ``MOFKA_GROUP_FILE``:

    ./start-bedrock.sh /tmp/mofka-test
    export MOFKA_GROUP_FILE=/tmp/mofka-test/mofka.flock.json

    # mofka — LD_PRELOAD pulls in lmdb/leveldb that libyokan-server.so dlopens.
    P=~/conda-envs/mofa-stream/lib
    LD_PRELOAD="$P/liblmdb.so:$P/libleveldb.so" LD_LIBRARY_PATH=$P \\
        ~/conda-envs/mofa-stream/bin/python -u mofa/diaspora.py mofka \\
            --group-file "$MOFKA_GROUP_FILE"

    # octopus — no LD_PRELOAD needed.
    ~/conda-envs/mofa-stream/bin/python -u mofa/diaspora.py octopus

On many-core shared hosts, cap BLAS workers first — otherwise numpy starves
librdkafka's broker threads ("Unable to create broker thread"):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <command above>
"""

import atexit
import json
import logging
import os
import queue as _queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union, Literal, Any

from colmena.exceptions import TimeoutException, KillSignalException
from colmena.models import SerializationMethod
from colmena.queue.base import ColmenaQueues
from diaspora_stream.api import Driver

logger = logging.getLogger(__name__)


# Sentinel for graceful dispatcher shutdown.
_DISPATCHER_STOP = object()


class _MofkaDispatcher:
    """Thread that owns the mofka Driver, topics, producers, and consumers,
    and serves push/pull requests from any caller thread.

    See the module docstring for why mofka calls must all funnel through one
    thread (Argobots/thallium ES affinity). Each "recv" reuses a per-consumer
    pending FutureEvent across short-timeout waits — pull() once, wait() many
    — matching the single-process poll loop.
    """

    def __init__(self, queue_names, group_file, partition_rank, log_benchmark=None):
        self._queue_names = list(queue_names)
        self._group_file = group_file
        self._partition_rank = partition_rank
        # Optional DiasporaQueues.log_benchmark callback so the dispatcher
        # thread can record the broker-side push/pull/ack latencies that the
        # caller-visible SEND_MSG/GET_MSG numbers fold the inter-thread handoff
        # into. A no-op default keeps the dispatcher usable on its own.
        self._log_benchmark = log_benchmark or (lambda *a, **k: None)
        self._req_queue: _queue.Queue = _queue.Queue()
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._loop, name="MofkaDispatcher", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            raise self._init_error

    def _loop(self):
        topics: Dict[str, Dict[str, Any]] = {}
        try:
            from mochi.mofka.client import MofkaDriver  # noqa: F401 lazy

            driver = Driver(
                backend="mofka", options={"group_file": self._group_file}
            )
            for qn in self._queue_names:
                if not driver.topic_exists(qn):
                    try:
                        logger.info("dispatcher: create_topic %s", qn)
                        driver.create_topic(name=qn)
                        admin = MofkaDriver(group_file=self._group_file)
                        admin.add_memory_partition(
                            topic_name=qn, server_rank=self._partition_rank
                        )
                    except Exception as e:
                        # Lost a race with a sibling dispatcher (e.g. the
                        # parsl-forked task-server process) that already
                        # ran create_topic. Re-check; raise only if the
                        # topic truly isn't there.
                        if "already exists" not in str(e).lower():
                            raise
                        if not driver.topic_exists(qn):
                            raise
                topics[qn] = {
                    "topic": driver.open_topic(qn),
                    "producer": None,
                    "consumer": None,
                    "pending_pull": None,
                }
            logger.info("dispatcher: %d topic(s) pre-wired", len(topics))
        except BaseException as e:  # noqa: BLE001 — caller re-raises
            self._init_error = e
            self._ready.set()
            return

        self._ready.set()

        while True:
            req = self._req_queue.get()
            if req is _DISPATCHER_STOP:
                return
            op, qname, payload, fut = req
            try:
                t = topics[qname]

                if op == "send":
                    if t["producer"] is None:
                        t["producer"] = t["topic"].producer(
                            f"producer-{qname}"
                        )
                    # push().wait() = broker ack; flush() is skipped (see
                    # module docstring: mofka 0.8.2 flush().wait() blocks
                    # the full timeout even when nothing is batched).
                    push_start = time.perf_counter_ns()
                    t["producer"].push(payload).wait(timeout_ms=10000)
                    push_end = time.perf_counter_ns()
                    self._log_benchmark("MOFKA_PUSH", push_start, push_end,
                                        queue=qname, msg=payload)
                    fut.put((True, None))
                elif op == "recv":
                    timeout_ms = payload
                    if t["consumer"] is None:
                        t["consumer"] = t["topic"].consumer(
                            f"consumer-{qname}"
                        )
                    if t["pending_pull"] is None:
                        t["pending_pull"] = t["consumer"].pull()
                    wait_start = time.perf_counter_ns()
                    ev = t["pending_pull"].wait(timeout_ms=timeout_ms)
                    wait_end = time.perf_counter_ns()
                    md = None
                    if ev is not None and ev.event_id is not None:
                        md = ev.metadata
                        ev.acknowledge()
                        ack_end = time.perf_counter_ns()
                        t["pending_pull"] = None
                        self._log_benchmark("MOFKA_EVENT_WAIT", wait_start,
                                            wait_end, queue=qname, msg=md)
                        self._log_benchmark("MOFKA_EVENT_ACK", wait_end,
                                            ack_end, queue=qname, msg=md)
                    fut.put((True, md))
                else:
                    fut.put((False, ValueError(f"unknown op {op!r}")))
            except BaseException as e:  # noqa: BLE001
                fut.put((False, e))

    def submit(self, op: str, qname: str, payload: Any) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("Mofka dispatcher thread is not running")
        fut: _queue.Queue = _queue.Queue()
        self._req_queue.put((op, qname, payload, fut))
        ok, result = fut.get()
        if not ok:
            raise result
        return result

    def stop(self):
        if self._thread.is_alive():
            self._req_queue.put(_DISPATCHER_STOP)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


class DiasporaQueues(ColmenaQueues):
    def __init__(
        self,
        topics: Collection[str],
        prefix: str = "mofa_test2",
        serialization_method: Union[
            str, SerializationMethod
        ] = SerializationMethod.PICKLE,
        keep_inputs: bool = True,
        proxystore_name: Optional[Union[str, Dict[str, str]]] = None,
        proxystore_threshold: Optional[Union[int, Dict[str, int]]] = None,
        stream_engine: Literal["files", "mofka", "octopus"] = "files",
        stream_conf: Optional[Dict[str, str]] = None,
        benchmark_file: Optional[str] = None,
    ):
        self.stream_engine = stream_engine
        stream_conf = stream_conf or {}

        # Benchmark trace. Always on: every send/recv/connect/topic op is timed
        # and written as one JSON object per line. ``benchmark_file`` is a *base*
        # path — the live PID is appended (see _setup_benchmark_logger) so that
        # the forked Parsl task server and, in the thinker/server split, the two
        # processes (even when they live in separate containers) each own a
        # distinct trace file instead of interleaving writes into one.
        if benchmark_file is None:
            benchmark_file = f"diaspora-trace-{uuid.uuid4().hex[:8]}.log"
        self._benchmark_file_base = str(benchmark_file)
        self._benchmark_handler: Optional[logging.Handler] = None
        self.benchmark_logger: Optional[logging.Logger] = None
        self._setup_benchmark_logger()
        atexit.register(self._flush_benchmark)

        if self.stream_engine == "octopus":
            from diaspora_event_sdk import Client as GlobusClient

            c = GlobusClient()
            key_result = c.create_key()
            region = stream_conf.get("region", "us-east-1")
            authorization = c.app.get_authorizer(c.DIASPORA_RESOURCE_SERVER).get_authorization_header()

            # AWS_* are read by librdkafka's aws_msk_iam SASL plugin via boto3's
            # default credential chain — they're not referenced by name in this file.
            os.environ["AWS_SECRET_ACCESS_KEY"] = key_result["secret_key"]
            os.environ["AWS_ACCESS_KEY_ID"] = key_result["access_key"]
            os.environ["AWS_REGION"] = region
            os.environ["OCTOPUS_SUBJECT"] = c.subject_openid
            os.environ["OCTOPUS_AUTHORIZATION"] = authorization

            self.driver_config = {
                "kafka": {"bootstrap.servers": key_result["endpoint"].split(",")},
                "aws_msk_iam": {"region": region},
                "octopus": {
                    "subject_env": "OCTOPUS_SUBJECT",
                    "authorization_env": "OCTOPUS_AUTHORIZATION",
                },
                "namespace": c.namespace,
            }
        elif self.stream_engine == "mofka":
            if "group_file" not in stream_conf:
                raise ValueError("mofka backend requires 'group_file' in stream_conf")
            # mofka builds its Driver inside the dispatcher from _mofka_group_file;
            # it never reads self.driver_config (only the files/octopus connect()
            # path does), so we don't set one here.
            self._mofka_group_file = stream_conf["group_file"]
            self._mofka_partition_rank = int(stream_conf.get("partition_rank", 0))
        else:
            self.driver_config = {
                "root_path": stream_conf.get("root_path", "stream")
            }


        super().__init__(
            topics,
            serialization_method,
            keep_inputs,
            proxystore_name,
            proxystore_threshold,
        )
        # self.topics is handled in super
        self.prefix = prefix

        self._dispatcher: Optional[_MofkaDispatcher] = None
        self.driver = None
        self.opened_topics: Dict[str, Dict[str, Any]] = {}
        self._driver_lock = threading.Lock()
        # Guards lazy dispatcher startup so two worker threads can race
        # into the first queue op without spawning duplicate dispatchers.
        self._connect_lock = threading.Lock()

        # Defer mofka connect() so Argobots state is built post-fork by the
        # parsl child (see module docstring). Non-mofka backends are
        # pthread-safe and can connect eagerly.
        if self.stream_engine != "mofka":
            self.connect()

        # Drop the inherited Driver/dispatcher after fork so each process
        # rebuilds its own Argobots state (see module docstring).
        os.register_at_fork(after_in_child=self._on_fork)

    # ------------------------------------------------------------------ benchmark

    def _setup_benchmark_logger(self):
        """(Re)build the benchmark logger so this process writes its own trace.

        File name is ``<base stem>-pid<PID><suffix>`` so the forked Parsl task
        server and, in the thinker/server split, the two halves (even in separate
        containers) never write into the same file. Called from __init__ and
        again from _on_fork / __setstate__, so a forked or unpickled child drops
        the handler it inherited (which points at the *parent's* file) and opens
        its own.

        A plain line-flushed ``FileHandler`` is used rather than a buffering
        ``MemoryHandler`` on purpose: (a) nothing is lost if a hung run is
        SIGKILLed, and (b) — the distributed-setting trap — a buffering handler
        inherited across a multiprocessing fork would, on close() in the child,
        flush the *parent's* still-buffered records into the *parent's* file and
        duplicate them. With no buffer there is nothing to mis-flush. The cost
        (one write per record) is off the measured interval and negligible next
        to the ms-scale ops being timed. ``delay=True`` means a process that
        never touches the queue (e.g. a Parsl worker fork) leaves no empty file."""
        base = Path(self._benchmark_file_base)
        path = base.with_name(f"{base.stem}-pid{os.getpid()}{base.suffix}")
        bench_logger = logging.getLogger("diaspora_queue")
        # Drop handlers inherited across fork/unpickle — they target the parent's
        # file. A FileHandler holds no buffer, so close() writes nothing.
        for h in list(bench_logger.handlers):
            bench_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        try:
            base.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        file_handler = logging.FileHandler(path, delay=True)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        bench_logger.addHandler(file_handler)
        bench_logger.setLevel(logging.DEBUG)
        bench_logger.propagate = False  # keep JSON out of the app's stdout/run.log
        self.benchmark_logger = bench_logger
        self._benchmark_handler = file_handler
        self._benchmark_path = path

    @staticmethod
    def _msg_size(msg) -> int:
        """Best-effort payload byte count. The workflow's queue messages are
        dicts wrapping a serialized colmena Result string, so report that
        inner length when present; otherwise fall back to sys.getsizeof."""
        try:
            if isinstance(msg, (bytes, bytearray, str)):
                return len(msg)
            if isinstance(msg, dict) and isinstance(msg.get("message"), (str, bytes, bytearray)):
                return len(msg["message"])
        except Exception:
            pass
        return sys.getsizeof(msg)

    def log_benchmark(self, task_name: str, start_ns: int, end_ns: int,
                      queue: Optional[str] = None, msg: Any = None) -> None:
        """Append one JSON record per timed operation. perf_counter_ns drives
        the durations (monotonic, per-process); wall_ns/pid let traces from the
        thinker and server processes be merged onto one timeline afterwards."""
        if self.benchmark_logger is None:
            return
        try:
            self.benchmark_logger.info(json.dumps({
                "task_name": task_name,
                "engine": self.stream_engine,
                "queue": queue,
                "msg_size": self._msg_size(msg),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
                "pid": os.getpid(),
                "wall_ns": time.time_ns(),
            }))
        except Exception:
            pass

    def _flush_benchmark(self) -> None:
        h = getattr(self, "_benchmark_handler", None)
        if h is not None:
            try:
                h.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------ helpers

    def _expected_queue_names(self):
        """The full set of {prefix}_… queue names ColmenaQueues will route
        through this object. Pre-creating them up front lets the mofka
        dispatcher build every topic/producer/consumer before the colmena
        agent threads start poking at us."""
        return [f"{self.prefix}_requests"] + [
            f"{self.prefix}_{t}_result" for t in self.topics
        ]

    # ------------------------------------------------------------------ fork

    def _reset_runtime_state(self):
        """Drop all lazily-built connection state so the next connect()
        rebuilds it. Shared by _on_fork (post-fork child) and __setstate__
        (post-unpickle) so the two reset paths can't drift."""
        self._dispatcher = None
        self.driver = None
        self.opened_topics = {}
        self._driver_lock = threading.Lock()
        self._connect_lock = threading.Lock()

    def _on_fork(self):
        # fork() copies only the calling thread, so the dispatcher pthread is
        # gone in the child. Reset all driver state and let lazy connect()
        # rebuild on first use.
        self._reset_runtime_state()
        # Re-point the benchmark trace at a fresh per-PID file so the child
        # (e.g. the forked Parsl task server) writes its own honest trace
        # instead of inheriting — and corrupting — the parent's buffered file.
        self._setup_benchmark_logger()

    # ------------------------------------------------------------------ pickle

    _UNPICKLABLE = ('driver', '_dispatcher', 'opened_topics',
                    '_driver_lock', '_connect_lock',
                    'benchmark_logger', '_benchmark_handler')

    def __getstate__(self):
        state = super().__getstate__()
        state['_was_connected'] = self.is_connected
        for k in self._UNPICKLABLE:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        was_connected = state.pop('_was_connected', False)
        super().__setstate__(state)
        self._reset_runtime_state()
        # Rebuild the benchmark logger for this (unpickled, possibly remote)
        # process — its handlers don't survive pickling. _benchmark_file_base
        # is carried in state, so the trace name stays consistent.
        self._benchmark_handler = None
        self.benchmark_logger = None
        self._setup_benchmark_logger()
        atexit.register(self._flush_benchmark)
        if was_connected:
            self.connect()

    # ------------------------------------------------------------------ connect

    def connect(self):
        """For mofka, start the dispatcher thread (owns the Driver and
        pre-creates every expected topic). For files/octopus, open a Driver
        on the calling thread; topic creation is lazy."""
        start = time.perf_counter_ns()
        if self.stream_engine == "mofka":
            with self._connect_lock:
                if self._dispatcher is not None and self._dispatcher.alive:
                    self.log_benchmark("CONNECT", start, time.perf_counter_ns())
                    return
                self._dispatcher = _MofkaDispatcher(
                    queue_names=self._expected_queue_names(),
                    group_file=self._mofka_group_file,
                    partition_rank=self._mofka_partition_rank,
                    log_benchmark=self.log_benchmark,
                )
            self.log_benchmark("CONNECT", start, time.perf_counter_ns())
            return

        if not self.driver:
            self.driver = Driver(backend=self.stream_engine, options=self.driver_config)
        self.log_benchmark("CONNECT", start, time.perf_counter_ns())

    def disconnect(self):
        """Disconnect from the server.

        Useful if sending the connection object to another process.
        """
        start = time.perf_counter_ns()
        if self._dispatcher is not None:
            self._dispatcher.stop()
            self._dispatcher = None
        self.driver = None
        self.opened_topics = {}
        self.log_benchmark("DISCONNECT", start, time.perf_counter_ns())
        self._flush_benchmark()

    # ------------------------------------------------------------------ non-mofka topic helpers

    def get_or_create_queue(self, queue, requester: Literal["producer", "consumer"]):
        """Used by the file/octopus paths. Mofka does not call this — its
        producers/consumers are owned by the dispatcher thread."""
        start = time.perf_counter_ns()
        with self._driver_lock:
            if queue in self.opened_topics:
                if requester in self.opened_topics[queue]:
                    self.log_benchmark("GET_TOPIC", start, time.perf_counter_ns(), queue=queue)
                    return self.opened_topics[queue][requester]

                if requester == "producer":
                    self.opened_topics[queue][requester] = self.opened_topics[queue]["topic"].producer(f"producer-{queue}")
                else:
                    self.opened_topics[queue][requester] = self.opened_topics[queue]["topic"].consumer(f"consumer-{queue}")
                self.log_benchmark("GET_TOPIC", start, time.perf_counter_ns(), queue=queue)
                return self.opened_topics[queue][requester]

            if not self.driver.topic_exists(queue):
                self.driver.create_topic(name=queue)
                if self.stream_engine == "octopus":
                    # octopus' create_topic is eventually consistent: open_topic
                    # called immediately after create returns "Topic not found".
                    deadline = time.time() + 30
                    while not self.driver.topic_exists(queue):
                        if time.time() > deadline:
                            raise TimeoutError(f"octopus topic {queue!r} did not propagate in 30s")
                        time.sleep(0.5)

            topic = self.driver.open_topic(queue)
            self.opened_topics[queue] = {"topic": topic}
            if requester == "producer":
                rq = topic.producer(f"producer-{queue}")
            else:
                rq = topic.consumer(f"consumer-{queue}")
            self.opened_topics[queue][requester] = rq
            self.log_benchmark("CREATE_TOPIC", start, time.perf_counter_ns(), queue=queue)
            return rq

    # ------------------------------------------------------------------ send/recv

    def _send_message(self, message, queue):
        start = time.perf_counter_ns()
        if not self.is_connected:
            self.connect()

        if self.stream_engine == "mofka":
            # The actual broker push happens on the dispatcher thread (which
            # logs MOFKA_PUSH); SEND_MSG here is the full caller-visible cost,
            # inter-thread handoff included.
            self._dispatcher.submit("send", queue, message)
            self.log_benchmark("SEND_MSG", start, time.perf_counter_ns(), queue=queue, msg=message)
            return

        producer = self.get_or_create_queue(queue, "producer")
        push_start = time.perf_counter_ns()
        producer.push(message).wait(timeout_ms=10000)
        # files/octopus need flush() for persistence; the mofka path skips it
        # (see module docstring).
        flush_start = time.perf_counter_ns()
        producer.flush().wait(timeout_ms=10000)
        end = time.perf_counter_ns()
        self.log_benchmark("PUSH", push_start, flush_start, queue=queue, msg=message)
        self.log_benchmark("FLUSH", flush_start, end, queue=queue, msg=message)
        self.log_benchmark("SEND_MSG", start, end, queue=queue, msg=message)

    def _get_message(self, queue, timeout: float = None):
        # perf_counter_ns drives the GET_MSG duration; a separate monotonic
        # clock drives the timeout budget. (The bench-branch original reused one
        # `start` for both, mixing perf_counter_ns with time.time() and
        # corrupting every duration — fixed here.)
        start = time.perf_counter_ns()
        if not self.is_connected:
            self.connect()
        # timeout=None means "block forever" — colmena's BaseTaskServer treats
        # TimeoutException from get_task(None) as a kill signal (see module
        # docstring). Finite timeout is a hard deadline.
        budget = None if timeout is None else max(1.0, float(timeout))
        per_poll_ms = 1000

        if self.stream_engine == "mofka":
            poll = lambda: self._dispatcher.submit("recv", queue, per_poll_ms)
        else:
            consumer = self.get_or_create_queue(queue, "consumer")
            future = consumer.pull()

            def poll():
                wait_start = time.perf_counter_ns()
                event = future.wait(timeout_ms=per_poll_ms)
                wait_end = time.perf_counter_ns()
                if event is None:
                    # Idle poll — folded into GET_MSG, not logged on its own to
                    # keep the trace centered on real message movement.
                    return None
                event.acknowledge()
                ack_end = time.perf_counter_ns()
                self.log_benchmark("EVENT_WAIT", wait_start, wait_end, queue=queue, msg=event.metadata)
                self.log_benchmark("EVENT_ACK", wait_end, ack_end, queue=queue, msg=event.metadata)
                return event.metadata

        budget_start = time.monotonic()
        while True:
            md = poll()
            if md is not None:
                self.log_benchmark("GET_MSG", start, time.perf_counter_ns(), queue=queue, msg=md)
                return md
            if budget is not None and time.monotonic() - budget_start > budget:
                raise TimeoutException(
                    f'Consumer {queue} timed out waiting for message.'
                )

    def _send_request(self, message: str, topic: str):
        queue = f"{self.prefix}_requests"
        event = {"message": message, "topic": topic}
        self._send_message(event, queue)

    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        queue = f'{self.prefix}_requests'
        md = self._get_message(queue, timeout)
        # send_kill_signal() sends the literal string "null"; honour the
        # ColmenaQueues contract and raise so the task server stops cleanly
        # instead of trying to validate "null" as a Result (which crashes it
        # with a pydantic ValidationError at shutdown).
        if md["message"] == "null":
            raise KillSignalException()
        return md["topic"], md["message"]

    def _send_result(self, message: str, topic: str):
        queue = f'{self.prefix}_{topic}_result'
        self._send_message({"message": message}, queue)

    def _get_result(self, topic: str, timeout: int = None) -> str:
        queue = f'{self.prefix}_{topic}_result'
        md = self._get_message(queue, timeout)
        return md["message"]

    @property
    def is_connected(self):
        if self.stream_engine == "mofka":
            return self._dispatcher is not None and self._dispatcher.alive
        return self.driver is not None


if __name__ == "__main__":
    import argparse
    import random
    import string

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("kafka").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="DiasporaQueues end-to-end test for mofka and octopus.")
    parser.add_argument("mode", choices=["mofka", "octopus"])
    parser.add_argument("--group-file", help="path to mofka.flock.json (mofka only)")
    args = parser.parse_args()
    if args.mode == "mofka" and not args.group_file:
        parser.error("--group-file is required for mofka mode")

    TOPICS = ["generation", "lammps", "cp2k", "training", "assembly"]
    # Prefix kept <=14 chars so {prefix}_generation_result fits the web
    # service's 32-char validate_name cap.
    prefix = "dq_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
    conf = ({"group_file": args.group_file} if args.mode == "mofka"
            else {"region": "us-east-1"})
    q = DiasporaQueues(topics=TOPICS, prefix=prefix, stream_engine=args.mode, stream_conf=conf)
    logger.info("queues ready: engine=%s prefix=%s", args.mode, prefix)

    rc = 0
    try:
        for topic in TOPICS:
            x = hash(topic) & 0xFF
            task_id = q.send_inputs(x, method=f"run_{topic}", topic=topic)
            got_topic, task = q.get_task(timeout=10)
            assert (got_topic, task.method) == (topic, f"run_{topic}"), (got_topic, task.method)
            task.deserialize()
            task.set_result({"input": x, "output": x * 2})
            task.serialize()
            q.send_result(task)
            res = q.get_result(topic=topic, timeout=10)
            assert res is not None and res.task_id == task_id, (res, task_id)
            logger.info("%s: %s -> %s", topic, x, res.value)
        logger.info("RESULT: PASS (%d round-trips on %s)", len(TOPICS), args.mode)
    except Exception:
        logger.exception("FAIL")
        rc = 1

    # Delete the prefix-scoped topics so they don't accumulate against the per-user
    # topic budget — octopus' kafka client starts failing with "Unable to create
    # broker thread" once too many topics exist. Mofka topics are scoped to the
    # local bedrock daemon and vanish on restart, so no cleanup is needed there.
    if args.mode == "octopus":
        from diaspora_event_sdk import Client
        admin = Client()
        created = [f"{prefix}_requests"] + [f"{prefix}_{t}_result" for t in TOPICS]
        for name in created:
            try:
                admin.delete_topic(name)
                logger.info("cleanup: deleted %s", name)
            except Exception as e:
                logger.warning("cleanup: delete_topic(%s) failed: %s", name, e)

    # os._exit skips Python finalizers; the Driver's C++ destructors hang on
    # regular interpreter exit.
    os._exit(rc)
