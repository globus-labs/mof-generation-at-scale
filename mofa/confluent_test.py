from datetime import datetime
import logging
import os

from proxyqueue import confluent_producer_conf, confluent_consumer_conf
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from confluent_kafka import Consumer, Producer, TopicPartition

logger = logging.getLogger(__name__)

assert os.environ["AWS_ACCESS_KEY_ID"]  # handled in proxyqueue
assert os.environ["AWS_SECRET_ACCESS_KEY"]  # handled in proxyqueue
assert os.environ["OCTOPUS_BOOTSTRAP_SERVERS"]


auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token("us-east-1")
logger.warning(f"auth_token:: {auth_token}")


if __name__ == "__main__":
    request_topic = "mofa_test2_requests"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.warning(f"Message to send {current_time}")
    producer = Producer(confluent_producer_conf())
    producer.produce(request_topic, str(current_time))
    producer.flush()

    consumer = Consumer(confluent_consumer_conf("my-group", "earliest"))
    topic_partition = TopicPartition(request_topic, partition=0)
    consumer.assign([topic_partition])

    msg = consumer.poll(timeout=5)  # Adjust timeout as necessary
    logger.warning(f"Received message: {msg.value().decode('utf-8')}")
    consumer.close()
    logger.warning("Remember to recreate the topic after testing")
