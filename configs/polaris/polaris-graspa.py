"""Version which runs gRASPA instead"""
from dataclasses import dataclass
from mofa.hpc.config import SingleJobHPCConfig


@dataclass
class Config(SingleJobHPCConfig):
    """Polaris, but running gRASPA on compute nodes"""

    raspa_version = 'graspa'
    raspa_cmd = (
        "/eagle/projects/HPCBot/thang/soft/gRASPA/src_clean/nvc_main.x"
    )


hpc_config = Config()
