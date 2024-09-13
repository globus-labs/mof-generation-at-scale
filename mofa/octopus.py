"""Queues that use Octopus"""

from typing import Collection, Optional, Union, Dict, Tuple
import logging

import os
import json

import pickle

from colmena.exceptions import TimeoutException, KillSignalException
from colmena.models import SerializationMethod

from colmena.queue.base import ColmenaQueues
from diaspora_event_sdk import KafkaProducer
from diaspora_event_sdk import KafkaConsumer
from kafka.errors import KafkaError

from time import time


logger = logging.getLogger(__name__)


def value_serializer(v):
    return json.dumps(v).encode("utf-8")


def value_deserializer(x):
    return json.loads(x.decode("utf-8"))


class OctopusQueues(ColmenaQueues):
    def __init__(
        self,
        topics: Collection[str],
        prefix: str = "mofa_test1",
        auto_offset_reset: str = "earliest",
        discard_events_before: int = int(time() * 1000),
        serialization_method: Union[
            str, SerializationMethod
        ] = SerializationMethod.PICKLE,
        keep_inputs: bool = True,
        proxystore_name: Optional[Union[str, Dict[str, str]]] = None,
        proxystore_threshold: Optional[Union[int, Dict[str, int]]] = None,
    ):
        assert os.environ["OCTOPUS_AWS_ACCESS_KEY_ID"]
        assert os.environ["OCTOPUS_AWS_SECRET_ACCESS_KEY"]
        assert os.environ["OCTOPUS_BOOTSTRAP_SERVERS"]
        super().__init__(
            topics,
            serialization_method,
            keep_inputs,
            proxystore_name,
            proxystore_threshold,
        )
        # self.topics in handled in super
        self.prefix = prefix
        self.auto_offset_reset = auto_offset_reset
        self.discard_events_before = discard_events_before

        self.request_producer = None
        self.request_consumer = None
        self.result_consumers = {}

    def connect_request_producer(self):
        """Connect the request producer."""
        if not isinstance(self.request_producer, KafkaProducer):
            self.request_producer = KafkaProducer(
                value_serializer=value_serializer,
            )

    def connect_request_consumer(self):
        """Connect the request consumer."""
        if not isinstance(self.request_consumer, KafkaConsumer):
            request_topic = f"{self.prefix}_requests"
            self.request_consumer = KafkaConsumer(
                request_topic,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=value_deserializer,
                max_poll_records=1,
            )

    def connect_result_consumer(self, topic):
        """Connect a result consumer for a specific topic."""
        if (topic not in self.result_consumers) or not isinstance(
            self.result_consumers[topic], KafkaConsumer
        ):
            result_topic = f"{self.prefix}_{topic}_result"
            self.result_consumers[topic] = KafkaConsumer(
                result_topic,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=value_deserializer,
                max_poll_records=1,
            )

    def disconnect_request_producer(self):
        """Disconnect the request producer."""
        if self.request_producer:
            self.request_producer.close()

    def disconnect_request_consumer(self):
        """Disconnect the request consumer."""
        if self.request_consumer:
            self.request_consumer.close()
            self.request_consumer = None

    def disconnect_result_consumer(self, topic):
        """Disconnect the result consumer for a specific topic."""
        consumer = self.result_consumers.pop(topic, None)
        if consumer:
            consumer.close()

    def __getstate__(self):
        state = super().__getstate__()

        if self.request_producer:
            state["request_producer"] = "connected"

        if self.request_consumer:
            state["request_consumer"] = "connected"

        for topic in list(self.result_consumers.keys()):
            state["result_consumers"][topic] = "connected"

        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        if self.request_producer:
            self.connect_request_producer()

        if self.request_consumer:
            self.connect_request_consumer()

        for topic in list(self.result_consumers.keys()):
            self.connect_result_consumer(topic)

    def _publish_event(self, message, octopus_topic):
        try:
            future = self.request_producer.send(octopus_topic, message)
            res = future.get(timeout=5)
            self.request_producer.flush()
            return res
        except KafkaError as e:
            print(f"Error producing message: {e}")

    def _send_request(self, message: str, topic: str):
        self.connect_request_producer()
        queue = f"{self.prefix}_requests"
        event = {"message": message, "topic": topic}
        self._publish_event(event, queue)

    def _get_message(
        self,
        consumer,
        timeout: float = None,
    ):
        if timeout is None:
            timeout = 0
        timeout *= 1000  # to ms
        assert consumer, "should be initialized"

        try:
            while True:  # blocks indefinitely
                events = consumer.poll(timeout)
                for tp, es in events.items():
                    ts, event = es[0].timestamp, es[0].value
                    if ts < self.discard_events_before:
                        continue

                    return event

        except KafkaError as e:
            print(f"Error consuming message: {e}")
            raise TimeoutException()

    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        self.connect_request_consumer()

        event = self._get_message(self.request_consumer, timeout)
        if event["message"].endswith("null"):
            raise KillSignalException()

        topic, message = event["topic"], event["message"]
        return topic, message

    def _send_result(self, message: str, topic: str):
        self.connect_request_producer()
        queue = f"{self.prefix}_{topic}_result"
        self._publish_event(message, queue)

    def _get_result(self, topic: str, timeout: int = None) -> str:
        self.connect_result_consumer(topic)
        consumer = self.result_consumers.get(topic)
        if not consumer:
            raise ConnectionError(
                f"No consumer connected for topic '{topic}'. Did you call 'connect_result_consumer('{topic}')'?"
            )

        event = self._get_message(consumer, timeout)
        return event


if __name__ == "__main__":
    queues = OctopusQueues(
        topics=["generation", "lammps", "cp2k", "training", "assembly"]
    )
    print(queues)
    print(queues.topics)

    queues.connect_request_producer()
    queues.connect_request_consumer()
    for topic in queues.topics:
        queues.connect_result_consumer(topic)

    queues_dumped = pickle.dumps(queues)
    print(queues_dumped)

    queues_loaded = pickle.loads(queues_dumped)
    print(queues_loaded.request_producer)
    print(queues_loaded.request_consumer)
    print(queues_loaded.result_consumers)
