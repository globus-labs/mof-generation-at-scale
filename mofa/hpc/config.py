"""Configuring a particular HPC resource"""
from dataclasses import dataclass, field
from functools import cached_property
from subprocess import Popen
from pathlib import Path
import os
from typing import Literal

from parsl import HighThroughputExecutor
from parsl import Config
from parsl.launchers import MpiExecLauncher, WrappedLauncher
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
    sim_fraction: float = 0.9
    """Maximum fraction of resources set aside for simulation tasks"""
    sim_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for simulation tasks"""
    ai_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for AI tasks"""
    helper_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for processing tasks"""

    @property
    def num_workers(self) -> int:
        """Total number of workers"""
        raise NotImplementedError

    @property
    def num_ai_workers(self) -> int:
        """Number of workers set aside for AI"""
        raise NotImplementedError

    @property
    def num_sim_workers(self) -> int:
        """Number of workers available for simulation"""
        return self.num_workers - self.num_ai_workers

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
    """Configuration used for testing purposes

    Uses a different worker for AI and simulation tasks.
    """

    torch_device = 'cuda'
    lammps_env = {}

    sim_executors = ['sim']
    ai_executors = ['ai']
    helper_executors = ['helper']

    @property
    def num_workers(self):
        return 2

    @property
    def num_ai_workers(self) -> int:
        return 1

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='sim', max_workers=1),
                HighThroughputExecutor(label='helper', max_workers=1),
                HighThroughputExecutor(label='ai', max_workers=1, available_accelerators=1)
            ],
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
    lammps_env = {}

    ai_hosts: list[str] = field(default_factory=list)
    """Hosts which will run AI tasks"""
    sim_hosts: list[str] = field(default_factory=list)
    """Hosts which will run simulation tasks"""

    cpus_per_node: int = 32
    """Number of CPUs to use per node"""
    gpus_per_node: int = 4
    """Number of GPUs per compute node"""

    sim_executors = ['sim']
    ai_executors = ['ai']
    helper_executors = ['helper']

    @cached_property
    def hosts(self):
        """Lists of hosts on which this computation is running"""
        # Determine the number of nodes from the PBS_NODEFILE
        node_file = os.environ['PBS_NODEFILE']
        with open(node_file) as fp:
            hosts = [x.strip() for x in fp]

        # Determine the number of hosts to use for simulation
        num_sim_hosts = min(int(self.sim_fraction * len(hosts)), len(hosts) - 1)
        self.sim_hosts = hosts[-num_sim_hosts:]  # Assign the last hosts, simulation tasks are likely more CPU-intensive and would interfere with Thinker
        self.ai_hosts = hosts[:-num_sim_hosts]
        return hosts

    @property
    def num_workers(self):
        return len(self.hosts) * self.gpus_per_node

    @property
    def num_ai_workers(self):
        return len(self.ai_hosts) * self.gpus_per_node

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f'mpiexec -n {len(self.hosts)} --ppn 1 --depth={self.cpus_per_node} '
                 f'--cpu-bind depth monitor_utilization --frequency {freq} {log_dir.absolute()}'.split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        # Write the nodefiles
        ai_nodefile = run_dir / 'ai.hosts'
        ai_nodefile.write_text('\n'.join(self.ai_hosts))
        sim_nodefile = run_dir / 'sim.hosts'
        sim_nodefile.write_text('\n'.join(self.sim_hosts))

        # Use the same worker_init
        worker_init = """
module load kokkos
module load nvhpc/23.3
module list
source activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris
which python
hostname"""

        # Divide CPUs on "sim" such that a from each NUMA affinity are set aside for helpers
        #  See https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
        cpus_per_worker = self.cpus_per_node // self.gpus_per_node  # Only use one thread per core
        helpers_per_worker = 1  # One core per worker set aside for "helpers"
        sim_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - helpers_per_worker - 1}" for i in range(4)][::-1]  # GPU3 is to cores 0-7
        helper_cores = [str(i) for w in range(4) for i in range((w + 1) * cpus_per_worker - helpers_per_worker, (w + 1) * cpus_per_worker)]

        ai_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - 1}" for i in range(4)][::-1]  # All CPUs to AI tasks

        # Launch 4 workers per node, one per GPU
        return Config(executors=[
            HighThroughputExecutor(
                label='ai',
                max_workers=4,
                cpu_affinity='list:' + ":".join(ai_cores),
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.ai_hosts)} --ppn 1 --hostfile {ai_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='sim',
                max_workers=self.gpus_per_node,
                cpu_affinity='list:' + ":".join(sim_cores),
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.sim_hosts)} --ppn 1 --hostfile {sim_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='helper',
                max_workers=helpers_per_worker * self.gpus_per_node,
                cpu_affinity='list:' + ":".join(helper_cores),
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.sim_hosts)} --ppn 1 --hostfile {sim_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
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
    gpus_per_node = 12

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
    'localXY': LocalConfigXY,
    'polaris': PolarisConfig,
    'sunspot': SunspotConfig
}
