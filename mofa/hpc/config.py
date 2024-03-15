"""Configuring a particular HPC resource"""
from dataclasses import dataclass, field
from subprocess import Popen
from pathlib import Path
import os

from parsl import HighThroughputExecutor

from parsl import Config
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider


class HPCConfig:
    """Base class for HPC configuration"""

    # How tasks run
    torch_device: str = 'cpu'
    """Device used for DiffLinker training"""
    lammps_cmd: tuple[str] = ('lmp_serial',)
    """Command used to launch a non-MPI LAMMPS task"""

    # How tasks are distributed
    sim_executors: str | list[str] = 'all'
    ai_executors: str | list[str] = 'all'

    @property
    def num_workers(self) -> int:
        """Total number of workers"""
        raise NotImplementedError

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        """Launch a monitor process on all resources

        Args:
            log_dir: Folder in which to save logs
            freq: Interval between monitoring
        Returns:
            Process handle
        """
        raise NotImplementedError

    def make_parsl_config(self, run_dir: Path) -> Config:
        """Make a Parsl configuration

        Args:
            run_dir: Directory in which results will be stored
        Returns:
            Configuration that saves Parsl logs into the run directory
        """
        raise NotImplementedError()


@dataclass(kw_only=True)
class LocalConfig(HPCConfig):
    """Configuration used for testing purposes"""

    torch_device = 'cuda'

    @property
    def num_workers(self):
        return 2

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[HighThroughputExecutor(max_workers=2)],
            run_dir=str(run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class PolarisConfig(HPCConfig):
    """Configuration used on Polaris"""

    torch_device = 'cuda'
    lammps_cmd = ('/lus/eagle/projects/ExaMol/mofa/lammps-2Aug2023/build-gpu-nompi-mixed/lmp '
                  '-sf gpu -pk gpu 1').split()
    hosts: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Determine the number of nodes from the PBS_NODEFILE
        node_file = os.environ['PBS_NODEFILE']
        with open(node_file) as fp:
            self.hosts = [x.strip() for x in fp]

    @property
    def num_workers(self):
        return len(self.hosts) * 4

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f'mpiexec -n {len(self.hosts)} --ppn 1 --depth=64 '
                 f'--cpu-bind depth monitor_utilization --frequency {freq} {log_dir.absolute()}'.split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        num_nodes = len(self.hosts)

        # Launch 4 workers per node, one per GPU
        return Config(executors=[
            HighThroughputExecutor(
                max_workers=4,
                cpu_affinity='block-reverse',
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    worker_init="""
module load kokkos
module load nvhpc/23.3
module list
source activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris
which python
hostname""",
                    nodes_per_block=num_nodes
                )
            ),
        ],
            run_dir=str(run_dir)
        )


configs: dict[str, type[HPCConfig]] = {
    'local': LocalConfig,
    'polaris': PolarisConfig
}
