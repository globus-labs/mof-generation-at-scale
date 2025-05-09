from pytorch_lightning.plugins.environments import ClusterEnvironment
import os


class PBSClusterEnvironment(ClusterEnvironment):
    """Specification of a cluster environment where rank is determined externally"""

    def __init__(self, rank: int, size: int, ranks_per_node: int):
        self.rank = rank
        self.size = size
        self.ranks_per_node = ranks_per_node

    @property
    def creates_processes_externally(self) -> bool:
        """Whether the environment creates the subprocesses or not."""
        return True

    @property
    def main_address(self) -> str:
        """The main address through which all processes connect and communicate."""
        main_address = os.environ.get("MASTER_ADDR", None)
        assert main_address is not None, "MASTER_ADDR environment variable must be set"
        return main_address

    @property
    def main_port(self) -> int:
        """An open and configured port in the main node through which all processes communicate."""
        main_port = os.environ.get("MASTER_PORT", None)
        assert main_port is not None, "MASTER_PORT environment variable must be set"
        return int(main_port)

    @staticmethod
    def detect() -> bool:
        """Detects the environment settings corresponding to this cluster and returns ``True`` if they match."""
        return True  # defaulting to intelligent users

    def world_size(self) -> int:
        """The number of processes across all devices and nodes."""
        return self.size

    def set_world_size(self, size: int) -> None:
        # ???
        pass

    def global_rank(self) -> int:
        """The rank (index) of the currently running process across all nodes and devices."""
        return self.rank

    def set_global_rank(self, rank: int) -> None:
        pass

    def local_rank(self) -> int:
        """The rank (index) of the currently running process inside of the current node."""
        return self.rank % self.ranks_per_node  # might be different for non-Aurora systems

    def node_rank(self) -> int:
        """The rank (index) of the node on which the current process runs."""
        return self.rank // self.ranks_per_node

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        """Validates settings configured in the script against the environment, and raises an exception if there is an
        inconsistency."""
        pass

    def teardown(self) -> None:
        """Clean up any state set after execution finishes."""
        pass
