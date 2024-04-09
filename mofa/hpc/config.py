"""Configuring a particular HPC resource"""
from dataclasses import dataclass, field
from functools import cached_property
from subprocess import Popen
from typing import Literal
from pathlib import Path
from math import ceil
import os

from more_itertools import batched

from parsl import HighThroughputExecutor
from parsl import Config
from parsl.launchers import MpiExecLauncher, WrappedLauncher, SimpleLauncher
from parsl.providers import LocalProvider


class HPCConfig:
    """Base class for HPC configuration"""

    # How tasks run
    torch_device: str = 'cpu'
    """Device used for DiffLinker training"""
    lammps_cmd: tuple[str] = ('lmp_serial',)
    """Command used to launch a non-MPI LAMMPS task"""
    cp2k_cmd: str = 'cp2k_shell.psmp'
    """Command used to launch the CP2K shell"""
    lammps_env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to include when running LAMMPS"""

    # How tasks are distributed
    ai_fraction: float = 0.1
    """Maximum fraction of resources set aside for AI tasks"""
    dft_fraction: float = 0.4
    """Maximum fraction of resources not used for AI that will be used for CP2K"""
    lammps_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for simulation tasks"""
    cp2k_executors: Literal['all'] | list[str] = 'all'
    """Which executors to use for CP2K tasks"""
    inference_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for AI tasks"""
    train_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for AI tasks"""
    helper_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for processing tasks"""

    @property
    def num_workers(self) -> int:
        """Total number of workers"""
        return self.num_lammps_workers + self.num_cp2k_workers + self.number_inf_workers

    @property
    def number_inf_workers(self) -> int:
        """Number of workers set aside for AI inference tasks"""
        raise NotImplementedError

    @property
    def num_lammps_workers(self) -> int:
        """Number of workers available for LAMMPS tasks"""
        raise NotImplementedError

    @property
    def num_cp2k_workers(self):
        """Number of workers available for CP2K tasks"""
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
    """Configuration used for testing purposes

    Uses a different worker for AI and simulation tasks.
    """

    torch_device = 'cuda'
    lammps_env = {}

    lammps_executors = ['sim']
    inference_executors = ['ai']
    train_executors = ['ai']
    helper_executors = ['helper']

    @property
    def num_workers(self):
        return self.num_lammps_workers + self.num_cp2k_workers + self.number_inf_workers

    @property
    def number_inf_workers(self) -> int:
        return 1

    @property
    def num_lammps_workers(self) -> int:
        return 1

    @property
    def num_cp2k_workers(self) -> int:
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
class LocalXYConfig(HPCConfig):
    """Configuration Xiaoli uses for testing purposes"""

    torch_device = 'cuda'
    lammps_cmd = "/home/xyan11/software/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1".split()
    lammps_env = {}

    lammps_executors = ['sim']
    inference_executors = ['ai']
    train_executors = ['ai']
    helper_executors = ['helper']

    @property
    def num_workers(self):
        return self.num_lammps_workers + self.num_cp2k_workers + self.number_inf_workers

    @property
    def number_inf_workers(self) -> int:
        return 1

    @property
    def num_lammps_workers(self) -> int:
        return 1

    @property
    def num_cp2k_workers(self) -> int:
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
class PolarisConfig(HPCConfig):
    """Configuration used on Polaris"""

    torch_device = 'cuda'
    lammps_cmd = ('/lus/eagle/projects/ExaMol/mofa/lammps-2Aug2023/build-gpu-nompi-mixed/lmp '
                  '-sf gpu -pk gpu 1').split()
    lammps_env = {}
    run_dir: Path | None = None  # Set when building the configuration

    nodes_per_cp2k: int = 2
    """Number of nodes per CP2K task"""
    lammps_per_gpu: int = 2
    """Number of LAMMPS to run per GPU"""

    ai_hosts: list[str] = field(default_factory=list)
    """Hosts which will run AI tasks"""
    lammps_hosts: list[str] = field(default_factory=list)
    """Hosts which will run LAMMPS tasks"""
    cp2k_hosts: list[str] = field(default_factory=list)
    """Hosts which will run CP2K tasks"""

    cpus_per_node: int = 32  # We choose 32 cores to only use one thread per core
    """Number of CPUs to use per node"""
    gpus_per_node: int = 4
    """Number of GPUs per compute node"""

    lammps_executors = ['lammps']
    inference_executors = ['inf']
    train_executors = ['train']
    cp2k_executors = ['cp2k']
    helper_executors = ['helper']

    @property
    def cp2k_cmd(self):
        # TODO (wardlt): Turn these into factory classes to ensure everything gets set on build
        assert self.run_dir is not None, 'This must be run after the Parsl config is built'
        return (f'mpiexec -n {self.nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                f'--hostfile {self.run_dir}/cp2k-hostfiles/local_hostfile.`printf %03d $PARSL_WORKER_RANK` '
                '/lus/eagle/projects/ExaMol/cp2k-2024.1/set_affinity_gpu_polaris.sh '
                '/lus/eagle/projects/ExaMol/cp2k-2024.1/exe/local_cuda/cp2k_shell.psmp')

    @cached_property
    def hosts(self):
        """Lists of hosts on which this computation is running"""
        # Determine the number of nodes from the PBS_NODEFILE
        node_file = os.environ['PBS_NODEFILE']
        with open(node_file) as fp:
            hosts = [x.strip() for x in fp]

        # Determine the number of nodes to use for AI
        num_ai_hosts = max(1, min(int(self.ai_fraction * len(hosts)), len(hosts) - self.nodes_per_cp2k - 1))
        self.ai_hosts = hosts[:num_ai_hosts]
        if num_ai_hosts < 2:
            raise ValueError('We need at least two AI workers. Increase node count or ai_fraction')

        # Determine the number of hosts to use for simulation
        sim_hosts = hosts[num_ai_hosts:]
        max_cp2k_slots = len(sim_hosts) // self.nodes_per_cp2k
        num_cp2k_slots = max(1, min(int(self.dft_fraction * max_cp2k_slots), max_cp2k_slots))  # [nodes_per_cp2k, len(sim_hosts) - nodes_per_cp2k]
        num_dft_hosts = num_cp2k_slots * self.nodes_per_cp2k
        self.lammps_hosts = sim_hosts[num_dft_hosts:]
        self.cp2k_hosts = sim_hosts[:num_dft_hosts]
        return hosts

    @property
    def number_inf_workers(self):
        return (len(self.ai_hosts) - 1) * self.gpus_per_node

    @property
    def num_lammps_workers(self):
        return len(self.lammps_hosts) * self.gpus_per_node * self.lammps_per_gpu

    @property
    def num_cp2k_workers(self):
        return ceil(len(self.cp2k_hosts) / self.nodes_per_cp2k)

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f'mpiexec -n {len(self.hosts)} --ppn 1 --depth={self.cpus_per_node} '
                 f'--cpu-bind depth monitor_utilization --frequency {freq} {log_dir.absolute()}'.split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        self.run_dir = str(run_dir.absolute())  # Used for CP2K config
        assert len(self.hosts) > 0, 'No hosts detected'

        # Write the nodefiles
        ai_nodefile = run_dir / 'ai.hosts'
        ai_nodefile.write_text('\n'.join(self.ai_hosts[1:]))  # First is used for training
        lammps_nodefile = run_dir / 'lammps.hosts'
        lammps_nodefile.write_text('\n'.join(self.lammps_hosts))
        cp2k_nodefile = run_dir / 'cp2k.hosts'
        cp2k_nodefile.write_text('\n'.join(self.cp2k_hosts))

        # Use the same worker_init for most workers
        worker_init = """
module load kokkos
module load nvhpc/23.3
module list
source /home/lward/miniconda3/bin/activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris
which python
hostname"""

        # Make the nodefiles for the CP2K workers
        nodefile_path = run_dir / 'cp2k-hostfiles'
        nodefile_path.mkdir(parents=True)
        for i, nodes in enumerate(batched(self.cp2k_hosts, self.nodes_per_cp2k)):
            (nodefile_path / f'local_hostfile.{i:03d}').write_text("\n".join(nodes))

        # Divide CPUs on "sim" such that a from each NUMA affinity are set aside for helpers
        #  See https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
        lammps_per_node = self.gpus_per_node * self.lammps_per_gpu
        cpus_per_worker = self.cpus_per_node // lammps_per_node
        helpers_per_worker = 1  # One core per worker set aside for "helpers"
        sim_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - helpers_per_worker - 1}" for i in range(lammps_per_node)][::-1]  # GPU3 ~ c0-7
        helper_cores = [str(i) for w in range(lammps_per_node) for i in range((w + 1) * cpus_per_worker - helpers_per_worker, (w + 1) * cpus_per_worker)]
        lammps_accel = [str(i) for i in range(self.gpus_per_node) for _ in range(self.lammps_per_gpu)]

        cpus_per_worker = self.cpus_per_node // self.gpus_per_node
        ai_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - 1}" for i in range(4)][::-1]  # All CPUs to AI tasks

        # Launch 4 workers per node, one per GPU
        return Config(executors=[
            HighThroughputExecutor(
                label='inf',
                max_workers=4,
                cpu_affinity='list:' + ":".join(ai_cores),
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.ai_hosts) - 1} --ppn 1 --hostfile {ai_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='train',
                max_workers=1,
                available_accelerators=self.gpus_per_node,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n 1 --ppn 1 --host {self.ai_hosts[0]} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='lammps',
                max_workers=len(lammps_accel),
                cpu_affinity='list:' + ":".join(sim_cores),
                available_accelerators=lammps_accel,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.lammps_hosts)} --ppn 1 --hostfile {lammps_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='cp2k',
                max_workers=self.num_cp2k_workers,
                cores_per_worker=1e-6,
                provider=LocalProvider(
                    launcher=SimpleLauncher(),  # Places a single worker on the launch node
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='helper',
                max_workers=len(helper_cores),
                cpu_affinity='list:' + ":".join(helper_cores),
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.lammps_hosts)} --ppn 1 --hostfile {lammps_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
        ],
            run_dir=str(run_dir)
        )


# TODO (wardlt): Update with changes in schema
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
    'localXY': LocalXYConfig,
    'polaris': PolarisConfig,
    'sunspot': SunspotConfig
}
