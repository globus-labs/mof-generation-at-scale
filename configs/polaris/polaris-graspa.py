"""Version which runs gRASPA instead"""
from dataclasses import dataclass
from mofa.hpc.config import SingleJobHPCConfig


@dataclass
class Config(SingleJobHPCConfig):
    """Polaris, but running gRASPA on compute nodes"""

    raspa_version = 'graspa'
    raspa_cmd = "/lus/eagle/projects/MOFA/lward/gRASPA/src_clean/nvc_main.x"
    raspa_executors = ['lammps']


hpc_config = Config()
