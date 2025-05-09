"""Version which runs gRASPA instead"""
from dataclasses import dataclass
from mofa.hpc.config import SingleJobHPCConfig, RASPAVersion


class Config(SingleJobHPCConfig):
    """Polaris, but running gRASPA on compute nodes"""

    raspa_version: RASPAVersion = 'graspa'
    raspa_cmd: str = "/lus/eagle/projects/MOFA/lward/gRASPA/src_clean/nvc_main.x"
    raspa_executors: list[str] = ['lammps']


hpc_config = Config()

