"""Test utilities related to Colmena"""
from colmena.queue import ColmenaQueues, PipeQueues
from pytest import fixture
from colmena.models import Result

from mofa.hpc.colmena import DiffLinkerInference


def data_func(size: int) -> bytes:
    for i in range(1):
        yield b'1' * size


@fixture()
def queues() -> ColmenaQueues:
    return PipeQueues()


def test_message_relay(queues):
    # Create the function and inputs
    func = DiffLinkerInference(
        function=data_func, streaming_queue=queues, store_return_value=True
    )
    task = Result(inputs=((8,), {}), serialization_method='pickle')
    task.topic = 'default'
    task.serialize()

    # Calling the function should result in a completed task being returned
    result = func(task)
    assert result.success
    assert result.complete
    assert result.value is None

    # And 8 new tasks being placed into the queue
    topic, task_stream = queues.get_task(timeout=1)
    assert task_stream.method == 'process_ligands'
    assert topic == 'default'
