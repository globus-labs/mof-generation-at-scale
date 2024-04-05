"""Utilities specific to application using Colmena"""
from typing import Any

from colmena.models import Result
from colmena.models.methods import PythonGeneratorMethod


class DiffLinkerInference(PythonGeneratorMethod):
    """Subclass of the DiffLinker Python generator which submits linker post-processing as a new Task"""

    def stream_result(self, y: Any, result: Result, start_time: float):
        """Submit a new task given the linkers"""
        self.streaming_queue.send_inputs(
            y,
            method='process_ligands',
            topic=result.topic,
            task_info=result.task_info
        )
