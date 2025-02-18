# the implementation here follows Valerie's:
# https://github.com/ValHayot/mofka-docker/blob/proxystore/mocto/octopus.py

import parsl
import logging
import os
import pickle
import sys
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from datetime import datetime
from time import sleep, time
from typing import Collection, Dict, Optional, Tuple, Union, Literal
from uuid import uuid4

from colmena.exceptions import KillSignalException, TimeoutException
from colmena.models import SerializationMethod
from colmena.queue.base import ColmenaQueues
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.store import Store, register_store
from proxystore.stream import StreamConsumer, StreamProducer

logger = logging.getLogger(__name__)

ENGINE: Literal["octopus", "mofka"] = os.environ["STREAM_ENGINE"]

if ENGINE == "octopus":
    from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
    from confluent_kafka import Consumer, Producer, TopicPartition
    from proxystore.stream.shims.kafka import KafkaPublisher, KafkaSubscriber

    assert os.environ["OCTOPUS_AWS_ACCESS_KEY_ID"]
    assert os.environ["OCTOPUS_AWS_SECRET_ACCESS_KEY"]
    assert os.environ["OCTOPUS_BOOTSTRAP_SERVERS"]

    assert os.environ["PROXYSTORE_GLOBUS_CLIENT_ID"]
    assert os.environ["PROXYSTORE_GLOBUS_CLIENT_SECRET"]
    assert os.environ["PROXYSTORE_ENDPOINT"]

    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["OCTOPUS_AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["OCTOPUS_AWS_SECRET_ACCESS_KEY"]
else:
    from proxystore.ex.stream.shims.mofka import MofkaSubscriber
    from proxystore.ex.stream.shims.mofka import MofkaPublisher

    assert (MOFKA_GROUPFILE := os.environ["MOFKA_GROUPFILE"])
    assert os.environ["MOFKA_PROTOCOL"]


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

        if ENGINE == "octopus":
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
        else:
            logger.info("Creating a MofkaPublisher")
            publisher = MofkaPublisher(group_file=MOFKA_GROUPFILE)
            self.request_producer = StreamProducer(publisher=publisher)
            logger.info("MofkaPublisher creation completed")

    def connect_request_consumer(self):
        if not isinstance(self.request_consumer, StreamConsumer):

            request_topic = f"{self.prefix}_requests"
            if ENGINE == "octopus":
                consumer = Consumer(
                    confluent_consumer_conf(self.group_id, self.auto_offset_reset)
                )
                # consumer.subscribe([request_topic])
                topic_partition = TopicPartition(request_topic, partition=0)
                consumer.assign([topic_partition])
                subscriber = KafkaSubscriber(client=consumer)
            else:
                logger.info("Creating a MofkaSubscriber")
                subscriber = MofkaSubscriber(
                    group_file=MOFKA_GROUPFILE,
                    topic_name=request_topic,
                    subscriber_name=str(f"MOFA-request-{uuid4()}"),
                )
                logger.info("MofkaSubscriber creation completed")
            self.request_consumer = StreamConsumer(
                subscriber=subscriber,
            )

    def connect_result_consumer(self, topic):
        if (topic not in self.result_consumers) or not isinstance(
            self.result_consumers[topic], StreamConsumer
        ):
            result_topic = f"{self.prefix}_{topic}_result"

            if ENGINE == "octopus":
                consumer = Consumer(
                    confluent_consumer_conf(self.group_id, self.auto_offset_reset)
                )
                consumer.subscribe([result_topic])
                subscriber = KafkaSubscriber(client=consumer)
            else:
                logger.info("Connecting to result consumer Mofka")
                subscriber = MofkaSubscriber(
                    group_file=MOFKA_GROUPFILE,
                    topic_name=result_topic,
                    subscriber_name=str(f"MOFA-result-{uuid4()}"),
                )

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
        logger.info("Publishing event")
        try:
            self.request_producer.send(proxy_topic, message, evict=False)
            self.request_producer.flush_topic(proxy_topic)
        except Exception as e:
            print(f"Error producing message: {e}")

    def _send_request(self, message: str, topic: str):
        self.connect_request_producer()
        event = {"message": message, "topic": topic}
        self._publish_event(event, f"{self.prefix}_requests")

    def _get_message(
        self,
        consumer,
        timeout: float = None,
    ):
        logger.info("Consuming event")
        if timeout is None:
            timeout = 0
        # timeout *= 1000  # to ms
        assert consumer, "consumer should be initialized"

        event = consumer.next_object()
        return event

    def _get_request(self, timeout: float = None) -> Tuple[str, str]:
        self.connect_request_consumer()
        event = self._get_message(self.request_consumer, timeout)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(f"ProxyQueues::request event:: {current_time}, event={event}")
        if event["message"].endswith("null"):
            raise KillSignalException()

        topic, message = event["topic"], event["message"]
        return topic, message

    def _send_result(self, message: str, topic: str):
        self.connect_request_producer()
        self._publish_event(message, f"{self.prefix}_{topic}_result")

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

    @parsl.python_app
    def consume(serialized_queues):
        queues_loaded = pickle.loads(serialized_queues)

        for i in range(10):
            print(f"{queues_loaded._get_request()=}")
            print(f"{queues_loaded._get_request()=}")
            print(f'{queues_loaded._get_result("generation")=}')
            print(f'{queues_loaded._get_result("generation")=}')
            print(f'{queues_loaded._get_result("generation")=}')

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("proxyqueue.log"),
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    queues = ProxyQueues(
        topics=["generation", "lammps", "cp2k", "training", "assembly"],
    )

    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers_per_node=1,
                cpu_affinity="block",
            )
        ]
    )
    parsl.clear()
    parsl.load(config)

    assert not any(
        queues.proxystore_name.values()
    )  # disable internal use of proxystore
    assert not any(
        queues.proxystore_threshold.values()
    )  # disable internal use of proxystore
    assert len(queues.topics) == 6  # including the 'default' topic

    queues.connect_request_producer()
    # queues.connect_request_consumer()

    for i in range(10):
        print(f"{queues._send_request('1234', 'generation')=}")
        print(f'{queues._send_request("4568", "generation")=}')
        print(f'{queues._send_result("_send_result message111", "generation")=}')
        print(f'{queues._send_result("_send_result message222", "generation")=}')
        print(f'{queues.send_inputs("test_send_inputs", topic="generation")=}')

    # for future impl for ColmenaQueues:
    # it's very importantant to ensure serilization works
    # otherwise it would cause silent errors in MOFA,
    # causing future operation to fail
    queues_dumped = pickle.dumps(queues)

    future = consume(queues_dumped)
    print(future.result())
    queues.close()
