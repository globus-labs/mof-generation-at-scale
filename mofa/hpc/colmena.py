"""Utilities specific to application using Colmena"""
from typing import Any, Callable, Union, Generator, Iterable, Optional

from colmena.models import Result
from colmena.models.methods import PythonGeneratorMethod
from colmena.queue import ColmenaQueues
from proxystore.store import Store, register_store


class DiffLinkerInference(PythonGeneratorMethod):
    """Subclass of the DiffLinker Python generator which submits linker post-processing as a new Task"""

    def __init__(self,
                 function: Callable[..., Union[Generator, Iterable]],
                 name: Optional[str] = None,
                 store_return_value: bool = False,
                 streaming_queue: Optional[ColmenaQueues] = None,
                 store: Optional[Store] = None) -> None:
        super().__init__(function, name, store_return_value, streaming_queue)
        self.store = store

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Make sure the store is registered
        self.store = Store.from_config(state['store'])
        register_store(self.store, exist_ok=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['store'] = self.store.config()
        return state

    def stream_result(self, y: Any, result: Result, start_time: float):
        """Submit a new task given the linkers"""
        self.streaming_queue.send_inputs(
            y,
            method='process_ligands',
            topic=result.topic,
            task_info=result.task_info
        )
