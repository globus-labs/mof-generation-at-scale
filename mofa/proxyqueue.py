# the implementation here follows Valerie's:
# https://github.com/ValHayot/mofka-docker/blob/proxystore/mocto/octopus.py

import logging
import os
from datetime import datetime
from time import time
from typing import Collection, Dict, Optional, Tuple, Union

from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from colmena.exceptions import KillSignalException, TimeoutException
from colmena.models import SerializationMethod
from colmena.queue.base import ColmenaQueues
from confluent_kafka import Consumer, Producer
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store, register_store
from proxystore.stream import StreamConsumer, StreamProducer
from proxystore.stream.shims.kafka import KafkaPublisher, KafkaSubscriber

logger = logging.getLogger(__name__)

assert os.environ["OCTOPUS_AWS_ACCESS_KEY_ID"]
assert os.environ["OCTOPUS_AWS_SECRET_ACCESS_KEY"]
assert os.environ["OCTOPUS_BOOTSTRAP_SERVERS"]

assert os.environ["PROXYSTORE_GLOBUS_CLIENT_ID"]
assert os.environ["PROXYSTORE_GLOBUS_CLIENT_SECRET"]

# assert os.environ["PROXYSTORE_ENDPOINT"]

# print(os.environ["PROXYSTORE_ENDPOINT"])


class ProxyQueues(ColmenaQueues):
    def __init__(
        self,
        store,
        topics: Collection[str],
        prefix: str = "mofa_test2",
        auto_offset_reset: str = "earliest",
        discard_events_before: int = int(time() * 1000),
        serialization_method: Union[
            str, SerializationMethod
        ] = SerializationMethod.PICKLE,
        keep_inputs: bool = True,
        proxystore_name: Optional[Union[str, Dict[str, str]]] = None,
        proxystore_threshold: Optional[Union[int, Dict[str, int]]] = None,
    ):
        super().__init__(
            topics,
            serialization_method,
            keep_inputs,
            proxystore_name,
            proxystore_threshold,
        )
        # self.topics in handled in super
        self.store = store
        self.prefix = prefix
        self.auto_offset_reset = auto_offset_reset
        self.discard_events_before = discard_events_before

        self.request_producer = None
        self.request_consumer = None
        self.result_consumers = {}

    def octopus_conf(self, group_id: str | None, auto_offset_reset: str):
        REGION = "us-east-1"
        assert os.environ["OCTOPUS_AWS_ACCESS_KEY_ID"]
        assert os.environ["OCTOPUS_AWS_SECRET_ACCESS_KEY"]
        assert os.environ["OCTOPUS_BOOTSTRAP_SERVERS"]

        def oauth_cb(oauth_config):
            auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token(REGION)
            return auth_token, expiry_ms / 1000

        conf = {
            "bootstrap.servers": os.environ["OCTOPUS_BOOTSTRAP_SERVERS"],
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "OAUTHBEARER",
            "oauth_cb": oauth_cb,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
        }

        return conf

    def connect_request_producer(self):
        """Connect the request producer."""
        if not isinstance(self.request_producer, StreamProducer):
            conf = self.octopus_conf("my-group", self.auto_offset_reset)
            producer = Producer(conf)
            publisher = KafkaPublisher(client=producer)

            proxy_topics = [f"{self.prefix}_requests"]
            for topic in self.topics:
                proxy_topic = f"{self.prefix}_{topic}_result"
                proxy_topics.append(proxy_topic)
            # print("proxy_topics", proxy_topics)

            oprod = StreamProducer(
                publisher=publisher, stores={k: self.store for k in proxy_topics}
            )
            self.request_producer = oprod

    def connect_request_consumer(self):
        """Connect the request consumer."""
        if not isinstance(self.request_consumer, StreamConsumer):
            conf = self.octopus_conf("my-group", self.auto_offset_reset)
            consumer = Consumer(conf)
            request_topic = f"{self.prefix}_requests"
            consumer.subscribe([request_topic])
            subscriber = KafkaSubscriber(client=consumer)
            oconsumer = StreamConsumer(subscriber=subscriber)
            self.request_consumer = oconsumer

    def connect_result_consumer(self, topic):
        """Connect a result consumer for a specific topic."""
        if (topic not in self.result_consumers) or not isinstance(
            self.result_consumers[topic], StreamConsumer
        ):
            conf = self.octopus_conf("my-group", self.auto_offset_reset)
            consumer = Consumer(conf)
            result_topic = f"{self.prefix}_{topic}_result"
            consumer.subscribe([result_topic])
            subscriber = KafkaSubscriber(client=consumer)
            oconsumer = StreamConsumer(subscriber=subscriber)
            self.result_consumers[topic] = oconsumer

    def disconnect_request_producer(self):
        """Disconnect the request producer."""
        if self.request_producer:
            self.request_producer.close()
            self.request_producer = None
            print("close!!!")

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
            self.request_producer.send(octopus_topic, message, evict=True)
            self.request_producer.flush_topic(octopus_topic)
        except Exception as e:
            print(
                f"Error producing message: {e}, {self.request_producer}, {type(self.request_producer)}"
            )

    def _send_request(self, message: str, topic: str):
        self.connect_request_producer()
        queue = f"{self.prefix}_requests"
        event = {"message": message, "topic": topic}
        print("_send_request", 123)
        self._publish_event(event, queue)
        print("_send_request", 456)

    def _get_message(
        self,
        consumer,
        timeout: float = None,
    ):
        if timeout is None:
            timeout = 0
        # timeout *= 1000  # to ms
        assert consumer, "consumer should be initialized"

        try:
            while True:
                event = consumer.next_object()
                if event:  # gives None if there is a timeout
                    print("event here:", event)
                    return event
                else:
                    print("event is None")

        except Exception as e:
            print(f"Error consuming message: {e}, {timeout}")
            raise TimeoutException()

    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        self.connect_request_consumer()
        print("_get_request", 123)
        event = self._get_message(self.request_consumer, timeout)
        print("_get_request", 456)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(f"ProxyQueues::request event:: {current_time}, event={event}")
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
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(f"ProxyQueues::result event:: {current_time}, event={event}")
        return event


if __name__ == "__main__":
    endpoints = [
        "41a566dd-d9b0-4c57-99ab-d16a8f1a0e54",
    ]
    endpoint_connector = EndpointConnector(endpoints)

    store = Store("my-store", connector=endpoint_connector)
    register_store(store)

    queues = ProxyQueues(
        store=store,
        topics=["generation", "lammps", "cp2k", "training", "assembly"],
        proxystore_name="my-store",
    )
    print(queues)
    print(queues.topics)

    # queues.connect_request_producer()
    # queues.connect_request_consumer()
    # for topic in queues.topics:
    #     queues.connect_result_consumer(topic)

    # queues_dumped = pickle.dumps(queues)
    # print(queues_dumped)

    # queues_loaded = pickle.loads(queues_dumped)
    # print(queues_loaded.request_producer)
    # print(queues_loaded.request_consumer)
    # print(queues_loaded.result_consumers)

    # queues._send_request("123", "generation")
    # queues._get_message(queues.request_consumer)
    # print("here")
