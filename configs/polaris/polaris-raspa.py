from dataclasses import dataclass
from mofa.hpc.config import SingleJobHPCConfig


@dataclass
class Config(SingleJobHPCConfig):
    """Use the basic Polaris config, which employs RASPA"""
    pass


hpc_config = Config()
