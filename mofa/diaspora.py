"""Diaspora Stream implementation of ColmenaQueues.

Run as an end-to-end Colmena round-trip (send_inputs / get_task / send_result
/ get_result on each of generation, lammps, cp2k, training, assembly). Mofka
needs a local bedrock daemon (see envs/polaris-stream.md section 4); octopus
uses cached Globus tokens at ~/.diaspora/storage.db (GlobusClient().create_key()
mints fresh AWS creds and bootstrap servers at runtime — no AWS_* or bootstrap
env vars need to be pre-set).

    # mofka — LD_PRELOAD pulls in lmdb/leveldb that libyokan-server.so dlopens.
    P=~/conda-envs/mofa-stream/lib
    LD_PRELOAD="$P/liblmdb.so:$P/libleveldb.so" LD_LIBRARY_PATH=$P \
        ~/conda-envs/mofa-stream/bin/python -u mofa/diaspora.py mofka \
            --group-file /tmp/mofka-test/mofka.flock.json

    # octopus — no LD_PRELOAD needed.
    ~/conda-envs/mofa-stream/bin/python -u mofa/diaspora.py octopus

On many-core shared hosts (e.g. a 256-core node), cap BLAS workers before
running — otherwise numpy auto-spawns one thread per core on import and
starves librdkafka's broker threads ("Unable to create broker thread"):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <command above>
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Collection, Dict, Optional, Tuple, Union, Literal, Any

from colmena.exceptions import KillSignalException, TimeoutException
from colmena.models import SerializationMethod
from colmena.queue.base import ColmenaQueues
from diaspora_stream.api import Driver

logger = logging.getLogger(__name__)


def value_serializer(v):
    return json.dumps(v).encode("utf-8")


def value_deserializer(x):
    return json.loads(x.decode("utf-8"))


def _error_if_unconnected(f):
    def wrapper(queue: 'DiasporaQueues', *args, **kwargs) -> Any:
        if not queue.is_connected:
            raise ConnectionError('Not connected. Did you call `.connect()`?')
        return f(queue, *args, **kwargs)

    return wrapper

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
        stream_engine: Literal["file", "mofka", "kafka", "octopus"] = "file",
        stream_conf: Dict[str, str] = { "region": "us-east-1", "auto_offset_reset": "earliest", "root_path": "stream"}
    ):
        self.stream_engine = stream_engine

        if self.stream_engine == "octopus":
            from diaspora_event_sdk import Client as GlobusClient

            c = GlobusClient()
            key_result = c.create_key()
            region = stream_conf["region"]
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
            self.driver_config = {"group_file": stream_conf["group_file"]}
            self._mofka_group_file = stream_conf["group_file"]
            self._mofka_partition_rank = int(stream_conf.get("partition_rank", 0))
        else:
            self.driver_config = {
                "root_path": stream_conf["root_path"]
            }


        super().__init__(
            topics,
            serialization_method,
            keep_inputs,
            proxystore_name,
            proxystore_threshold,
        )
        # self.topics in handled in super
        self.prefix = prefix
        

        self.driver = None
        self.opened_topics = {}
        self.connect()

    def __setstate__(self, state):
        super().__setstate__(state)

        # If you find the Driver placeholder, attempt to reconnect
        if self.driver == 'connected':
            self.driver = None
            self.connect()
            
                    

    def __getstate__(self):
        state = super().__getstate__()

        # If connected, remove the unpicklable Driver and put a placeholder instead
        if self.is_connected:
            state['driver'] = 'connected'
            for queue in self.opened_topics.keys():
                for k in self.opened_topics[queue].keys():
                    self.open_topic[queue][k] = 'connected'
                    
        return state

    def connect(self):
        """Connect to the Diaspora Stream driver."""
        if not self.driver:
            self.driver = Driver(backend=self.stream_engine, options=self.driver_config)

            for queue in self.opened_topics.keys():
                if 'topic' in self.opened_topics[queue]:
                    if self.opened_topics[queue]['topic'] == "connected":
                        self.opened_topics[queue]['topic'] = self.driver.open_topic(queue)
                        
                    if 'producer' in self.opened_topics[queue] and self.opened_topics[queue]['producer'] == "connected":
                        self.opened_topics[queue]['producer'] = self.opened_topics[queue]['topic'].producer(f'producer-{queue}')
                    if 'consumer' in self.opened_topics[queue] and self.opened_topics[queue]['consumer'] == "connected":
                        self.opened_topics[queue]['consumer'] = self.opened_topics[queue]['topic'].consumer(f'consumer-{queue}')
    
    def disconnect(self):
        """Disconnect from the server.

        Useful if sending the connection object to another process.
        """
        self.driver = None
        self.opened_topics = {}

    def _mofka_add_partition(self, topic_name: str) -> None:
        # Mofka topics need a partition before producer.push() will succeed,
        # and diaspora_stream.api.Driver doesn't expose partition mgmt, so use
        # the mofka admin client. We use a memory partition: it works against
        # the minimal bedrock config emitted by `mofkactl config generate`
        # (which has no abt-io provider) and matches the ephemeral data model
        # of these queues (results are pickled blobs, not durable records).
        from mochi.mofka.client import MofkaDriver
        admin = MofkaDriver(group_file=self._mofka_group_file)
        admin.add_memory_partition(
            topic_name=topic_name,
            server_rank=self._mofka_partition_rank,
        )

    def get_or_create_queue(self, queue, requester: Literal["producer", "consumer"]):
        
        if queue in self.opened_topics:
            if requester in  self.opened_topics[queue]:
                return self.opened_topics[queue][requester]
                #self.opened_topics = {}

            if requester == "producer":
                self.opened_topics[queue][requester] = self.opened_topics[queue]["topic"].producer(f"producer-{queue}")
            else:
                self.opened_topics[queue][requester] = self.opened_topics[queue]["topic"].consumer(f"consumer-{queue}")
            rq = self.opened_topics[queue][requester]
            #self.opened_topics = {}
            return rq

        if not self.driver.topic_exists(queue):
            self.driver.create_topic(name=queue)
            if self.stream_engine == "mofka":
                self._mofka_add_partition(queue)
            elif self.stream_engine == "octopus":
                # octopus' create_topic is eventually consistent: open_topic
                # called immediately after create returns "Topic not found".
                deadline = time.time() + 30
                while not self.driver.topic_exists(queue):
                    if time.time() > deadline:
                        raise TimeoutError(f"octopus topic {queue!r} did not propagate in 30s")
                    time.sleep(0.5)

        topic = self.driver.open_topic(queue)
        self.opened_topics[queue] = { "topic": topic }
        # print(f"***{self.opened_topics}***")

        if requester == "producer":
            # self.opened_topics[queue][requester]
            rq = topic.producer(f"producer-{queue}")
        else:
            # self.opened_topics[queue][requester] = 
            rq = topic.consumer(f"consumer-{queue}")
        
        # rq = self.opened_topics[queue][requester]
        self.opened_topics[queue][requester] = rq
        return rq
        

    def _send_message(self, message, queue):
        producer = self.get_or_create_queue(queue, "producer")
        future = producer.push(message).wait(timeout_ms=10000)
        producer.flush().wait(timeout_ms=10000)

    def _get_message(
        self,
        queue,
        timeout: float = None,
    ):
        if timeout is None:
            timeout = 1
        timeout *= 1000  # to ms

        consumer = self.get_or_create_queue(queue, "consumer")
        future = consumer.pull()
        event = None

        start = time.time()
        while event is None:
            event = future.wait(timeout_ms=timeout)
            if time.time() - start > 60:
                break

        if event is None:
            raise TimeoutException(f'Consumer {queue} timed out waiting for message.')

        event.acknowledge()
        return event
        

    @_error_if_unconnected
    def _send_request(self, message: str, topic: str):
        queue = f"{self.prefix}_requests"
        event = {"message": message, "topic": topic}
        # print(f"**REQ {event}**")
        self._send_message(event, queue)
        
        
    @_error_if_unconnected
    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        queue = f'{self.prefix}_requests'
        event = self._get_message(queue, timeout).metadata
        # print(f"**EVENT REQ {event}**")
        topic = event["topic"]
        request = event["message"] #json.loads(event["message"])

        return topic, request


    @_error_if_unconnected
    def _send_result(self, message: str, topic: str):
        queue = f'{self.prefix}_{topic}_result'
        event = { "message": message }
        # print(f"**RES {event}**")
        self._send_message(event, queue)

    @_error_if_unconnected
    def _get_result(self, topic: str, timeout: int = None) -> str:
        queue = f'{self.prefix}_{topic}_result'
        event = self._get_message(queue, timeout).metadata
        # print(f"**EVENT result {event}**")
        return event["message"]

    @property
    def is_connected(self):
        return self.driver is not None


if __name__ == "__main__":
    import argparse
    import os
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
            else {"region": "us-east-1", "auto_offset_reset": "earliest"})
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