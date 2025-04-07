"""Test utilities related to Colmena"""
import pickle as pkl

from colmena.queue import ColmenaQueues, PipeQueues
from proxystore.connectors.file import FileConnector
from proxystore.proxy import Proxy
from proxystore.store import Store, register_store
from pytest import fixture
from colmena.models import Result

from mofa.hpc.colmena import DiffLinkerInference


def data_func(size: int) -> bytes:
    for i in range(1):
        yield b'1' * size


@fixture()
def queues(store) -> ColmenaQueues:
    return PipeQueues(proxystore_name=store.name, proxystore_threshold=100)


@fixture()
def store(tmpdir) -> Store:
    store = Store(name='file', connector=FileConnector(store_dir=tmpdir))
    register_store(store)
    return store


def test_message_relay(queues, store):
    # Create the function and inputs
    func = DiffLinkerInference(
        function=data_func, streaming_queue=queues, store_return_value=True, store=store
    )
    func = pkl.loads(pkl.dumps(func))
    task = Result(inputs=((256,), {}), serialization_method='pickle')
    task.topic = 'default'
    task.serialize()

    # Calling the function should result in a completed task being returned
    result = func(task)
    assert result.success
    assert result.complete
    assert result.value is None

    # And 8 new tasks being placed into the queue
    topic, task_stream = queues.get_task(timeout=1)
    task_stream.deserialize()
    assert isinstance(task_stream.args[0], Proxy)
    assert task_stream.method == 'process_ligands'
    assert topic == 'default'
    assert task_stream.proxystore_name == 'file'
