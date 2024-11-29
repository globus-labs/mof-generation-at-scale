# the implementation here follows Valerie's:
# https://github.com/ValHayot/mofka-docker/blob/proxystore/mocto/octopus.py

import logging
import os
import pickle
from datetime import datetime
from time import sleep, time
from typing import Collection, Dict, Optional, Tuple, Union

from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from colmena.exceptions import KillSignalException, TimeoutException
from colmena.models import SerializationMethod
from colmena.queue.base import ColmenaQueues
from confluent_kafka import Consumer, Producer, TopicPartition
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
assert os.environ["PROXYSTORE_ENDPOINT"]


def oauth_cb(oauth_config):
    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token("us-east-1")
    print("oauth_cb", auth_token)
    return auth_token, expiry_ms / 1000


def confluent_producer_conf():
    return {
        "bootstrap.servers": os.environ["OCTOPUS_BOOTSTRAP_SERVERS"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "OAUTHBEARER",
        "oauth_cb": oauth_cb,
    }


def confluent_consumer_conf(group_id: str, auto_offset_reset: str):
    return {
        # "debug": "all",
        "bootstrap.servers": os.environ["OCTOPUS_BOOTSTRAP_SERVERS"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "OAUTHBEARER",
        "oauth_cb": oauth_cb,
        "group.id": group_id,
        "auto.offset.reset": auto_offset_reset,
    }


class ProxyQueues(ColmenaQueues):
    def __init__(
        self,
        topics: Collection[str],
        prefix: str = "mofa_test2",
        group_id: str = "my-mofa-group",
        auto_offset_reset: str = "earliest",
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
        self.prefix = prefix
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset

        self.proxy_topics = [f"{self.prefix}_requests"]
        for topic in self.topics:
            self.proxy_topics.append(f"{self.prefix}_{topic}_result")

        self.endpoint = os.environ["PROXYSTORE_ENDPOINT"]
        self.endpoint_connector = None
        self.store = None

        self.request_producer = None
        self.request_consumer = None
        self.result_consumers = {}

    def connect_request_producer(self):
        if not self.endpoint_connector:
            self.endpoint_connector = EndpointConnector([self.endpoint])
            self.store = Store("my-store", connector=self.endpoint_connector)

        if not isinstance(self.request_producer, StreamProducer):
            producer = Producer(confluent_producer_conf())
            publisher = KafkaPublisher(client=producer)
            self.request_producer = StreamProducer(
                publisher=publisher,
                stores={k: self.store for k in self.proxy_topics},
            )

    def connect_request_consumer(self):
        if not isinstance(self.request_consumer, StreamConsumer):
            consumer = Consumer(
                confluent_consumer_conf(self.group_id, self.auto_offset_reset)
            )
            request_topic = f"{self.prefix}_requests"
            # consumer.subscribe([request_topic])
            topic_partition = TopicPartition(request_topic, partition=0)
            consumer.assign([topic_partition])
            subscriber = KafkaSubscriber(client=consumer)
            self.request_consumer = StreamConsumer(
                subscriber=subscriber,
            )

    def connect_result_consumer(self, topic):
        if (topic not in self.result_consumers) or not isinstance(
            self.result_consumers[topic], StreamConsumer
        ):
            consumer = Consumer(
                confluent_consumer_conf(self.group_id, self.auto_offset_reset)
            )
            result_topic = f"{self.prefix}_{topic}_result"
            consumer.subscribe([result_topic])
            subscriber = KafkaSubscriber(client=consumer)
            oconsumer = StreamConsumer(
                subscriber=subscriber,
            )
            self.result_consumers[topic] = oconsumer

    def __getstate__(self):
        state = super().__getstate__()

        if self.request_producer:
            state["request_producer"] = "connected"

        if self.request_consumer:
            state["request_consumer"] = "connected"

        for topic in list(self.result_consumers.keys()):
            state["result_consumers"][topic] = "connected"

        state["endpoint_connector"] = None
        state["store"] = None

        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # only needed producers and consumers will be recreated

    def _publish_event(self, message, proxy_topic):
        try:
            self.request_producer.send(proxy_topic, message, evict=True)
            self.request_producer.flush_topic(proxy_topic)
        except Exception as e:
            print(f"Error producing message: {e}")

    def _send_request(self, message: str, topic: str):
        self.connect_request_producer()
        event = {"message": message, "topic": topic}
        print("_send_request", 123)
        self._publish_event(event, f"{self.prefix}_requests")
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

        event = consumer.next_object()
        return event

    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        print("_get_request", 0)
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
        print("_send_result", 123)
        self.connect_request_producer()
        print("_send_result", 456)
        self._publish_event(message, f"{self.prefix}_{topic}_result")
        print("_send_result", 789)

    def _get_result(self, topic: str, timeout: int = None) -> str:
        print("_get_result", 0)
        self.connect_result_consumer(topic)
        print("_get_result", 123)
        consumer = self.result_consumers.get(topic)
        if not consumer:
            raise ConnectionError(
                f"No consumer connected for topic '{topic}'. Did you call 'connect_result_consumer('{topic}')'?"
            )
        print("_get_result", 124)
        event = self._get_message(consumer, timeout)
        print("_get_result", 456)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(f"ProxyQueues::result event:: {current_time}, event={event}")
        return event


if __name__ == "__main__":
    queues = ProxyQueues(
        topics=["generation", "lammps", "cp2k", "training", "assembly"],
    )
    assert not any(
        queues.proxystore_name.values()
    )  # disable internal use of proxystore
    assert not any(
        queues.proxystore_threshold.values()
    )  # disable internal use of proxystore
    assert len(queues.topics) == 6  # including the 'default' topic

    # queues.connect_request_producer()
    # queues.connect_request_consumer()

    # for i in range(10):
    #     print(queues._send_request("1234", "generation"))
    #     print(queues._send_request("4568", "generation"))
    #     print(queues._send_result("_send_result message111", "generation"))
    #     print(queues._send_result("_send_result message222", "generation"))

    # queues_dumped = pickle.dumps(queues)
    # queues_loaded = pickle.loads(queues_dumped)
    # for i in range(10):
    #     print(queues_loaded._get_request())
    #     print(queues_loaded._get_request())
    #     print(queues_loaded._get_result("generation"))
    #     print(queues_loaded._get_result("generation"))
