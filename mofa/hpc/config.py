"""Configuring a particular HPC resource"""
from dataclasses import dataclass, field
from functools import cached_property
from subprocess import Popen
from typing import Literal
from pathlib import Path
from math import ceil

from more_itertools import batched

from parsl import HighThroughputExecutor
from parsl import Config
from parsl.launchers import WrappedLauncher, SimpleLauncher
from parsl.providers import LocalProvider

from mofa.simulation.graspa import gRASPARunner
from mofa.simulation.raspa import RASPARunner


class HPCConfig:
    """Base class for HPC configuration

    Construct a new configuration by subclassing HPCConfig
    or one of the subclasses provided below, then adapt
    the :meth:`launch_monitor_process` and
    :meth:`make_parsl_config` functions are appropriate
    for the new system.

    Also modify any of the configuration options provided in the
    base class as appropriate.

    MOFA uses the configuration class by setting the run_dir,
    potentially altering some of the configuration values at runtime
    (e.g., the fraction of nodes used for DFT), and then
    executing the :meth:`make_parsl_config` function to prepare a Parsl config
    before calling :meth:`launch_monitor_process` function to place
    monitoring daemons on each compute node.
    """

    # Variables which must be set at runtime
    run_dir: Path
    """Where the log files should be written"""

    # How tasks run
    torch_device: str = 'cpu'
    """Device used for DiffLinker training"""
    lammps_cmd: tuple[str] = ('lmp_serial',)
    """Command used to launch a non-MPI LAMMPS task"""
    cp2k_cmd: str = 'cp2k_shell.psmp'
    """Command used to launch the CP2K shell"""
    lammps_env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to include when running LAMMPS"""
    raspa_version: Literal['raspa2', 'raspa3', 'graspa'] = 'raspa2'
    """Version of RASPA used on this system"""
    raspa_cmd: str | None = None
    """Command used to launch RASPA"""

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

    def make_raspa_runner(self) -> RASPARunner | gRASPARunner:
        """Make the RASPA runner appropriate for this workflow"""

        if self.raspa_version == 'raspa2':
            if self.raspa_cmd is None:
                return RASPARunner()
            else:
                return RASPARunner(raspa_command=self.raspa_cmd)
        elif self.raspa_version == 'graspa':
            return gRASPARunner(graspa_command=self.raspa_cmd)
        else:
            raise NotImplementedError(f'No support for {self.raspa_cmd} yet.')

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        """Launch a monitor process on all resources

        Args:
            log_dir: Folder in which to save logs
            freq: Interval between monitoring
        Returns:
            Process handle
        """
        raise NotImplementedError

    def make_parsl_config(self) -> Config:
        """Make a Parsl configuration

        Returns:
            Configuration that saves Parsl logs into the run directory
        """
        raise NotImplementedError


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

    def make_parsl_config(self) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='sim', max_workers_per_node=1),
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='ai', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(self.run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class LocalXYConfig(HPCConfig):
    """Configuration Xiaoli uses for testing purposes"""

    torch_device = 'cuda'
    lammps_cmd = "/home/xyan11/software/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1".split()
    lammps_env = {}
    cp2k_cmd = "OMP_NUM_THREADS=1 mpiexec -np 8 /home/xyan11/software/cp2k-v2024.1/exe/local/cp2k_shell.psmp"
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

    def make_parsl_config(self) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='sim', max_workers_per_node=1),
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='ai', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(self.run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class UICXYConfig(HPCConfig):
    """Configuration Xiaoli uses for uic hpc"""

    torch_device = 'cuda'
    lammps_cmd = "/projects/cme_santc/xyan11/software/source/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1".split()
    lammps_env = {}

    lammps_executors = ['sim']
    ai_executors = ['ai']
    helper_executors = ['helper']

    cp2k_cmd = ("OMP_NUM_THREADS=2 mpirun -np 4 singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd "
                "/projects/cme_santc/xyan11/software/source/cp2k_v2023.1.sif cp2k_shell.psmp")

    @property
    def num_workers(self):
        return self.num_lammps_workers + self.num_cp2k_workers + self.num_ai_workers

    @property
    def num_ai_workers(self) -> int:
        return 2

    @property
    def num_lammps_workers(self) -> int:
        return 3

    @property
    def num_cp2k_workers(self) -> int:
        return 1

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='sim', max_workers_per_node=4, available_accelerators=4),
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='ai', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(self.run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class SingleJobHPCConfig(HPCConfig):
    """A configuration used for running MOFA inside a single HPC job

    Partitions nodes between different tasks and sets aside a series of
    CPUs on some nodes as "helpers" to run post-processing tasks.

    Modify this configuration for a new HPC by changing:
     1. Paths to the executables
     2. Number of cores and GPUs per node
     3. The scheduler used to detect the :attr:`hosts`
     4. The make Parsl config function

    This class provides an implementation for Polaris as an example.
    """

    torch_device = 'cuda'
    lammps_cmd = ('/lus/eagle/projects/ExaMol/mofa/lammps-2Aug2023/build-gpu-nompi-mixed/lmp '
                  '-sf gpu -pk gpu 1').split()
    lammps_env = {}

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
        from parsl.executors.high_throughput.mpi_resource_management import get_nodes_in_batchjob, Scheduler
        hosts = get_nodes_in_batchjob(Scheduler.PBS)

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

    def make_parsl_config(self) -> Config:
        run_dir = self.run_dir.absolute()  # Used for CP2K config
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
module use /soft/modulefiles
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
                max_workers_per_node=4,
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
                max_workers_per_node=1,
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
                max_workers_per_node=len(lammps_accel),
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
                max_workers_per_node=self.num_cp2k_workers,
                cores_per_worker=1e-6,
                provider=LocalProvider(
                    launcher=SimpleLauncher(),  # Places a single worker on the launch node
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='helper',
                max_workers_per_node=len(helper_cores),
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
            run_dir=str(run_dir),
            usage_tracking=3,
        )


configs: dict[str, type[HPCConfig]] = {
    'local': LocalConfig,
    'localXY': LocalXYConfig,
    'UICXY': UICXYConfig,
    'polaris': SingleJobHPCConfig,
}
