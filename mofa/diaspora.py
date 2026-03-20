"""Diaspora Stream implementation of ColmenaQueues


Prerequisites:
    - Environment variables for Octopus driver: AWS (OCTOPUS_AWS_ACCESS_KEY_ID, OCTOPUS_AWS_SECRET_ACCESS_KEY)
      and Kafka (OCTOPUS_BOOTSTRAP_SERVERS) must be set."
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
            subject = c.subject_openid
            authorization = c.web_client.authorizer.get_authorization_header()
            namespace = f"ns-{subject.replace('-', '')[-12:]}"

            os.environ["AWS_SECRET_ACCESS_KEY"] = key_result["secret_key"]
            os.environ["AWS_ACCESS_KEY_ID"] = key_result["access_key"]
            os.environ["AWS_REGION"] = stream_conf["region"]
            os.environ["OCTOPUS_SUBJECT"] = subject
            os.environ["OCTOPUS_AUTHORIZATION"] = authorization

            self.driver_config = {
                "kafka": {
                    "bootstrap.servers": key_result["endpoint"].split(","),
                    #"auto_offset_reset": stream_conf["auto_offset_reset"]
                },
                "aws_msk_iam": {
                    "region": region
                },
                "octopus": {
                    "subject_env": "OCTOPUS_SUBJECT",
                    "authorization_env": "OCTOPUS_AUTHORIZATION"
                },
                "namespace": namespace
            }
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
    # Configure logging only when running as main script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set higher log levels for third-party modules
    logging.getLogger("kafka").setLevel(logging.ERROR)

    # Initialize OctopusQueues with defined topics.
    topics = ["generation", "lammps", "cp2k", "training", "assembly"]
    queues = DiasporaQueues(topics=topics, stream_engine="octopus")
    logger.info("Initialized OctopusQueues with topics: %s\n", queues.topics)

    # Example tests for sending and receiving messages.
    queues._send_request("123456", "generation")
    logger.info("Request sent. Waiting for request...\n")
    topic, request_message = queues._get_request(timeout=1)
    logger.info("Received request: Topic='%s', Message='%s'\n", topic, request_message)
    queues._send_result("abcbbc", "generation")
    logger.info("Result sent. Waiting for result...\n")
    result_message = queues._get_result("generation", timeout=1)
    logger.info("Received result for topic 'generation': %s\n", result_message)