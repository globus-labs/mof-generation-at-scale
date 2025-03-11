from dataclasses import dataclass
from mofa.hpc.config import SingleJobHPCConfig


@dataclass
class Config(SingleJobHPCConfig):
    """Use the basic """
    pass


hpc_config = Config()
