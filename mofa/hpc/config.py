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
    lammps_env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to include when running LAMMPS"""

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
    lammps_env = {}

    @property
    def num_workers(self):
        return 2

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[HighThroughputExecutor(max_workers=1)],
            run_dir=str(run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class LocalConfigXY(HPCConfig):
    """Configuration used for testing purposes"""

    torch_device = 'cuda'
    lammps_cmd = ("/home/xyan11/software/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1").split()
    lammps_env = {}

    @property
    def num_workers(self):
        return 2

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[HighThroughputExecutor(max_workers=1)],
            run_dir=str(run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class PolarisConfig(HPCConfig):
    """Configuration used on Polaris"""

    torch_device = 'cuda'
    lammps_cmd = ('/lus/eagle/projects/ExaMol/mofa/lammps-2Aug2023/build-gpu-nompi-mixed/lmp '
                  '-sf gpu -pk gpu 1').split()
    hosts: list[str] = field(default_factory=list)
    """Lists of hosts on which this computation is running"""
    cpus_per_node: int = 64
    """Number of CPUs to use per node"""

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
            args=f'mpiexec -n {len(self.hosts)} --ppn 1 --depth={self.cpus_per_node} '
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


class SunspotConfig(PolarisConfig):
    """Configuration for running on Sunspot

    Each GPU tasks uses a single tile"""

    torch_device = 'xpu'
    lammps_cmd = ('/home/knight/lammps-git/src/lmp_aurora_gpu-lward '
                  '-pk gpu 1 -sf gpu').split()
    lammps_env = {'OMP_NUM_THREADS': '1'}
    cpus_per_node = 208

    @property
    def num_workers(self) -> int:
        return len(self.hosts) * 12

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        host_file = os.environ['PBS_NODEFILE']
        util_path = '/lus/gila/projects/CSC249ADCD08_CNDA/mof-generation-at-scale/bin/monitor_sunspot'
        return Popen(
            args=f"parallel --onall --sshloginfile {host_file} {util_path} --frequency {freq} ::: {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        num_nodes = len(self.hosts)

        accel_ids = [
            f"{gid}.{tid}"
            for gid in range(6)
            for tid in range(2)
        ]
        return Config(
            executors=[
                HighThroughputExecutor(
                    available_accelerators=accel_ids,  # Ensures one worker per accelerator
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers=12,
                    cores_per_worker=16,
                    provider=LocalProvider(
                        worker_init="""
source activate /lus/gila/projects/CSC249ADCD08_CNDA/mof-generation-at-scale/env
module reset
module use /soft/modulefiles/
module use /home/ftartagl/graphics-compute-runtime/modulefiles
module load oneapi/release/2023.12.15.001
module load intel_compute_runtime/release/775.20
module load gcc/12.2.0
module list
pwd
which python
hostname""",
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        nodes_per_block=num_nodes,
                    ),
                ),
            ],
            run_dir=str(run_dir)
        )


configs: dict[str, type[HPCConfig]] = {
    'local': LocalConfig,
    'polaris': PolarisConfig,
    'sunspot': SunspotConfig
}
