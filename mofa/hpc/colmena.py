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

        # Reconstruct the proxystore only if one was configured. The store is
        # optional — ProxyStore is disabled by default (see run_parallel_workflow.py).
        store_config = state.get('store')
        if store_config is not None:
            self.store = Store.from_config(store_config)
            register_store(self.store, exist_ok=True)
        else:
            self.store = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['store'] = self.store.config() if self.store is not None else None
        return state

    def stream_result(self, y: Any, result: Result, start_time: float):
        """Submit a new task given the linkers"""
        self.streaming_queue.send_inputs(
            y,
            method='process_ligands',
            topic=result.topic,
            task_info=result.task_info
        )
