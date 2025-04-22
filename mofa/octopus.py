"""Octopus implementation of ColmenaQueues

OctopusQueues leverages MSK (Amazon Managed Kafka) producers and consumers to manage request and result queues.

Prerequisites:
    - Environment variables for AWS (OCTOPUS_AWS_ACCESS_KEY_ID, OCTOPUS_AWS_SECRET_ACCESS_KEY)
      and Kafka (OCTOPUS_BOOTSTRAP_SERVERS) must be set.
    - `diaspora_event_sdk` and `kafka-python` libraries must be installed
      pip install "diaspora-event-sdk[kafka-python]"
"""

import json
import logging
import os
import pickle
from datetime import datetime
from time import time
from typing import Collection, Dict, Optional, Tuple, Union

from colmena.exceptions import KillSignalException, TimeoutException
from colmena.models import SerializationMethod
from colmena.queue.base import ColmenaQueues
from diaspora_event_sdk import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


def value_serializer(v):
    return json.dumps(v).encode("utf-8")


def value_deserializer(x):
    return json.loads(x.decode("utf-8"))


class OctopusQueues(ColmenaQueues):
    def __init__(
        self,
        topics: Collection[str],
        prefix: str = "mofa_test2",
        auto_offset_reset: str = "earliest",
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
        assert consumer, "consumer should be initialized"

        try:
            for message in consumer:
                yield message.value

        except KafkaError as e:
            print(f"Error consuming message: {e}")
            raise TimeoutException()

    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        self.connect_request_consumer()
        try:
            message = next(self._get_message(self.request_consumer, timeout))
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f"Octopus::request event:: {current_time}, event={message}")

            if message["message"].endswith("null"):
                raise KillSignalException()

            return message["topic"], message["message"]
        except StopIteration:
            raise TimeoutException()

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

        try:
            message = next(self._get_message(consumer, timeout))
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f"Octopus::result event:: {current_time}, event={message}")
            return message
        except StopIteration:
            raise TimeoutException()


if __name__ == "__main__":
    # Configure logging only when running as main script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set higher log levels for third-party modules
    logging.getLogger("kafka").setLevel(logging.ERROR)
    logging.getLogger("kafka.client").setLevel(logging.ERROR)
    logging.getLogger("kafka.conn").setLevel(logging.ERROR)
    logging.getLogger("kafka.coordinator").setLevel(logging.ERROR)
    logging.getLogger("kafka.consumer").setLevel(logging.ERROR)

    # Initialize OctopusQueues with defined topics.
    topics = ["generation", "lammps", "cp2k", "training", "assembly"]
    queues = OctopusQueues(topics=topics)
    logger.info("Initialized OctopusQueues with topics: %s\n", queues.topics)

    # Establish all necessary Kafka connections.
    queues.connect_request_producer()
    queues.connect_request_consumer()
    for topic in queues.topics:
        queues.connect_result_consumer(topic)

    # # Test serialization using pickle.
    queues_dumped = pickle.dumps(queues)
    logger.info("Serialized queues: %s", queues_dumped)
    queues_loaded = pickle.loads(queues_dumped)
    logger.info("Deserialized request producer: %s", queues_loaded.request_producer)
    logger.info("Deserialized request consumer: %s", queues_loaded.request_consumer)
    logger.info("Deserialized result consumers: %s\n", queues_loaded.result_consumers)

    # Example tests for sending and receiving messages.
    queues._send_request("123456", "generation")
    logger.info("Request sent. Waiting for request...\n")
    topic, request_message = queues._get_request(timeout=1)
    logger.info("Received request: Topic='%s', Message='%s'\n", topic, request_message)
    queues._send_result("abcbbc", "generation")
    logger.info("Result sent. Waiting for result...\n")
    result_message = queues._get_result("generation", timeout=1)
    logger.info("Received result for topic 'generation': %s\n", result_message)

    # Uncomment the following connectivity tests as needed:
    # from diaspora_event_sdk.sdk.kafka_client import MSKTokenProvider
    # tp = MSKTokenProvider()
    # print(tp.token())

    # producer = KafkaProducer(value_serializer=value_serializer)
    # future = producer.send(
    #     topic="__connection_test",
    #     value={"message": "Synchronous message from Diaspora SDK"},
    # )
    # record_metadata = future.get(timeout=10)
    # print(record_metadata)

    # from diaspora_event_sdk import block_until_ready
    # block_until_ready()
