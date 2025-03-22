from typing import Any
from pytorch_lightning.accelerators import Accelerator

import torch


class XPUAccelerator(Accelerator):
    """Support for a hypothetical XPU, optimized for large-scale machine learning."""

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()
