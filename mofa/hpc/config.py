"""Configuring a particular HPC resource"""
from functools import cached_property
from subprocess import Popen
from typing import Literal
from pathlib import Path
from math import ceil

from more_itertools import batched
from pydantic import BaseModel, Field, computed_field
from parsl import HighThroughputExecutor
from parsl import Config
from parsl.launchers import WrappedLauncher, SimpleLauncher
from parsl.providers import LocalProvider

from mofa.simulation.dft.base import BaseDFTRunner
from mofa.simulation.raspa.base import BaseRaspaRunner

RASPAVersion = Literal['raspa2', 'raspa3', 'graspa', 'graspa_sycl']
DFTVersion = Literal['cp2k', 'pwdft']


class HPCConfig(BaseModel):
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
    run_dir: Path | None = Field(default=None)
    """Where the log files should be written"""

    # How tasks run
    torch_device: str = Field(default='cpu')
    """Device used for DiffLinker training"""
    lammps_cmd: tuple[str, ...] = Field(default=('lmp_serial',))
    """Command used to launch a non-MPI LAMMPS task"""
    lammps_env: dict[str, str] = Field(default_factory=dict)
    """Extra environment variables to include when running LAMMPS"""
    raspa_version: RASPAVersion = Field(default='raspa2')
    """Version of RASPA used on this system"""
    raspa_cmd: tuple[str, ...] = Field(default=('simulate',))
    """Command used to launch gRASPA-sycl"""
    raspa_delete_finished: bool = True
    """Whether to delete RASPA run files after execution"""
    dft_version: DFTVersion = 'cp2k'

    # Settings related to distributed training
    gpus_per_node: int = Field(default=1)
    """How many GPUs per compute node"""
    num_training_nodes: int = Field(default=1)
    """How many nodes to use for training operations"""

    # How tasks are distributed
    ai_fraction: float = Field(default=0.1)
    """Maximum fraction of resources set aside for AI tasks"""
    dft_fraction: float = Field(default=0.4)
    """Maximum fraction of resources not used for AI that will be used for CP2K"""
    lammps_executors: Literal['all'] | list[str] = Field(default='all')
    """Which executors are available for simulation tasks"""
    dft_executors: Literal['all'] | list[str] = Field(default='all')
    """Which executors to use for CP2K tasks"""
    raspa_executors: Literal['all'] | list[str] = Field(default='all')
    """Which executors to use for RASPA tasks"""
    inference_executors: Literal['all'] | list[str] = Field(default='all')
    """Which executors are available for AI tasks"""
    train_executors: Literal['all'] | list[str] = Field(default='all')
    """Which executors are available for AI tasks"""
    helper_executors: Literal['all'] | list[str] = Field(default='all')
    """Which executors are available for processing tasks"""

    @computed_field()
    @property
    def dft_cmd(self) -> str:
        """Command to launch the DFT codes"""
        return 'cp2k_shell.psmp'

    @property
    def training_nodes(self) -> tuple[str, ...]:
        return ('localhost',)

    @property
    def num_training_ranks(self):
        return self.gpus_per_node * self.num_training_nodes

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

    def make_raspa_runner(self) -> BaseRaspaRunner:
        """Make the RASPA runner appropriate for this workflow"""

        run_dir = self.run_dir / 'raspa-runs'
        if self.raspa_version == 'raspa2':
            from mofa.simulation.raspa.raspa2 import RASPA2Runner
            return RASPA2Runner(raspa_command=self.raspa_cmd, run_dir=run_dir, delete_finished=self.raspa_delete_finished)
        elif self.raspa_version == 'graspa':
            from mofa.simulation.raspa.graspa import gRASPARunner
            return gRASPARunner(raspa_command=self.raspa_cmd, run_dir=run_dir, delete_finished=self.raspa_delete_finished)
        elif self.raspa_version == 'graspa_sycl':
            from mofa.simulation.raspa.graspa_sycl import GRASPASyclRunner
            return GRASPASyclRunner(raspa_command=self.raspa_cmd, run_dir=run_dir, delete_finished=self.raspa_delete_finished)
        else:
            raise NotImplementedError(f'No support for {self.raspa_version} yet.')

    def make_dft_runner(self) -> BaseDFTRunner:
        """Make the runner to use for DFT computations"""

        run_dir = self.run_dir / 'dft-runs'
        if self.dft_version == 'cp2k':
            from mofa.simulation.dft.cp2k import CP2KRunner
            return CP2KRunner(run_dir=run_dir, dft_cmd=self.dft_cmd)
        elif self.dft_version == 'pwdft':
            from mofa.simulation.dft.pwdft import PWDFTRunner
            return PWDFTRunner(run_dir=run_dir, dft_cmd=self.dft_cmd)
        else:
            raise NotImplementedError(f'No support for {self.run_dir} yet.')

    def launch_monitor_process(self, freq: int = 60) -> Popen:
        """Launch a monitor process on all resources

        Args:
            freq: Interval between monitoring (s)
        Returns:
            Process handle
        """
        raise NotImplementedError()

    def make_parsl_config(self) -> Config:
        """Make a Parsl configuration

        Returns:
            Configuration that saves Parsl logs into the run directory
        """
        raise NotImplementedError()


class LocalConfig(HPCConfig):
    """Configuration used for testing purposes. Runs all non-helper tasks on a single worker"""

    torch_device: str = 'cuda'
    lammps_env: dict[str, str] = {}
    lammps_cmd: tuple[str, ...] = ('/home/lward/Software/lammps-mace/build-mace/lmp',)
    raspa_cmd: tuple[str, ...] = ('/home/lward/Software/gRASPA/graspa-sycl/bin/sycl.out',)

    lammps_executors: list[str] = ['gpu']
    inference_executors: list[str] = ['gpu']
    train_executors: list[str] = ['gpu']
    helper_executors: list[str] = ['helper']
    raspa_executors: list[str] = ['gpu']

    @computed_field
    @property
    def dft_cmd(self) -> str:
        return '/home/lward/Software/cp2k-2024.2/exe/local_cuda/cp2k_shell.ssmp'

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

    def launch_monitor_process(self, freq: int = 20) -> Popen:
        log_dir = self.run_dir / 'logs'
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='gpu', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(self.run_dir / 'runinfo')
        )


class LocalXYConfig(LocalConfig):
    """Configuration Xiaoli uses for testing purposes"""

    lammps_cmd: tuple[str, ...] = "/home/xyan11/software/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1".split()

    @computed_field()
    def dft_cmd(self) -> str:
        return "OMP_NUM_THREADS=1 mpiexec -np 8 /home/xyan11/software/cp2k-v2024.1/exe/local/cp2k_shell.psmp"


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

    torch_device: str = 'cuda'
    lammps_cmd: tuple[str, ...] = (
         '/lus/eagle/projects/MOFA/lward/lammps-mace/build-mace/lmp '
         '-k on g 1 -sf kk'
    ).split()

    nodes_per_cp2k: int = Field(default=2, init=False)
    """Number of nodes per CP2K task"""
    lammps_per_gpu: int = Field(default=4, init=False)
    """Number of LAMMPS to run per GPU"""

    ai_hosts: tuple[str, ...] = Field(default_factory=list)
    """Hosts which will run AI tasks"""
    lammps_hosts: tuple[str, ...] = Field(default_factory=list)
    """Hosts which will run LAMMPS tasks"""
    cp2k_hosts: tuple[str, ...] = Field(default_factory=list)
    """Hosts which will run CP2K tasks"""

    cpus_per_node: int = Field(default=32, init=False)
    """Number of CPUs to use per node"""
    gpus_per_node: int = Field(default=4, init=False)
    """Number of GPUs per compute node"""

    lammps_executors: list[str] = ['lammps']
    inference_executors: list[str] = ['inf']
    train_executors: list[str] = ['train']
    dft_executors: list[str] = ['cp2k']
    helper_executors: list[str] = ['helper']
    raspa_executors: list[str] = ['lammps']

    @computed_field
    @property
    def dft_cmd(self) -> str:
        # TODO (wardlt): Turn these into factory classes to ensure everything gets set on built
        return (f'mpiexec -n {self.nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                f'--hostfile {self.run_dir.absolute()}/cp2k-hostfiles/local_hostfile.`printf %03d $PARSL_WORKER_RANK` '
                '/lus/eagle/projects/MOFA/lward/cp2k-2025.1/set_affinity_gpu_polaris.sh '
                '/lus/eagle/projects/MOFA/lward/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp')
    @property
    def training_nodes(self) -> tuple[str, ...]:
        return self.ai_hosts[:self.num_training_nodes]

    @cached_property
    def hosts(self):
        """Lists of hosts on which this computation is running"""
        # Determine the number of nodes from the PBS_NODEFILE
        from parsl.executors.high_throughput.mpi_resource_management import get_nodes_in_batchjob, Scheduler
        hosts = tuple(get_nodes_in_batchjob(Scheduler.PBS))

        # TODO (wardlt): Make skipping rank 0 configurable
        if len(hosts) > 1000:
            hosts = hosts[1:]  # The service is running on Host 0

        # Determine the number of nodes to use for AI
        num_ai_hosts = max(self.num_training_nodes, min(int(self.ai_fraction * len(hosts)), len(hosts) - self.nodes_per_cp2k - 1))
        self.ai_hosts = hosts[:num_ai_hosts]
        if num_ai_hosts < self.num_training_nodes:
            raise ValueError(f'We need at least {self.num_training_nodes} AI workers. Increase node count or ai_fraction')

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
        return (len(self.ai_hosts) - self.num_training_nodes) * self.gpus_per_node

    @property
    def num_lammps_workers(self):
        return len(self.lammps_hosts) * self.gpus_per_node * self.lammps_per_gpu

    @property
    def num_cp2k_workers(self):
        return ceil(len(self.cp2k_hosts) / self.nodes_per_cp2k)

    def launch_monitor_process(self, freq: int = 20) -> Popen:
        log_dir = self.run_dir / 'logs'
        return Popen(
            args=f'mpiexec -n {len(self.hosts)} --ppn 1 --depth={self.cpus_per_node} '
                 f'--cpu-bind depth monitor_utilization --frequency {freq} {log_dir.absolute()}'.split()
        )

    def make_parsl_config(self) -> Config:
        assert len(self.hosts) > 0, 'No hosts detected'

        # Write the nodefiles
        ai_nodefile, lammps_nodefile = self._make_nodefiles(self.run_dir)

        # Use the same worker_init for most workers
        worker_init = """
module use /soft/modulefiles
module list
source /home/lward/miniconda3/bin/activate /lus/eagle/projects/MOFA/lward/mof-generation-at-scale/env
which python
hostname""".strip()

        # Divide CPUs on "sim" such that a from each NUMA affinity are set aside for helpers
        #  See https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
        sim_cores, helper_cores = self._assign_cores()
        sim_cores.reverse()

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
                        f"mpiexec -n {len(self.ai_hosts) - self.num_training_nodes} --ppn 1 --hostfile {ai_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='train',
                max_workers_per_node=4,
                cpu_affinity='block-reverse',
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
                max_workers_per_node=self.lammps_per_gpu * self.gpus_per_node,
                cpu_affinity='list:' + ":".join(sim_cores),
                available_accelerators=self.lammps_per_gpu * self.gpus_per_node,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec --no-abort-on-failure -n {len(self.lammps_hosts)} --ppn 1 --hostfile {lammps_nodefile} --depth=64 --cpu-bind depth"
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
            run_dir=str(self.run_dir),
            usage_tracking=3,
        )

    def _assign_cores(self):
        """Assign cores on nodes running LAMMPS to both LAMMPS and helper functions

        Returns:
            - List of cores to use for each LAMMPS worker
            - List of cores to use for each helper worker
        """

        lammps_per_node = self.gpus_per_node * self.lammps_per_gpu
        cpus_per_worker = self.cpus_per_node // lammps_per_node
        helpers_per_worker = 1  # One core per worker set aside for "helpers"
        sim_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - helpers_per_worker - 1}" for i in range(lammps_per_node)]
        helper_cores = [str(i) for w in range(lammps_per_node) for i in range((w + 1) * cpus_per_worker - helpers_per_worker, (w + 1) * cpus_per_worker)]
        return sim_cores, helper_cores

    def _make_nodefiles(self, run_dir: Path):
        """Write the nodefiles for each type of workers to disk

        Run after setting the run directory

        Writes nodefiles for the AI and LAMMPS tasks,
        and a directory of nodefiles to be used for each CP2K instance

        Args:
            run_dir: Run directory for the computation
        Returns:
            - Path to the AI nodefile
            - Path to the LAMMPS nodefile
        """
        assert len(self.hosts) > 0, 'No hosts detected'  # TODO (wardlt): Also builds the hosts list, make that clearer/auto

        ai_nodefile = run_dir / 'ai.hosts'
        ai_nodefile.write_text('\n'.join(self.ai_hosts[self.num_training_nodes:]))  # First are used for training
        lammps_nodefile = run_dir / 'lammps.hosts'
        lammps_nodefile.write_text('\n'.join(self.lammps_hosts) + '\n')
        cp2k_nodefile = run_dir / 'cp2k.hosts'
        cp2k_nodefile.write_text('\n'.join(self.cp2k_hosts) + '\n')

        # Make the nodefiles for the CP2K workers
        nodefile_path = run_dir / 'cp2k-hostfiles'
        nodefile_path.mkdir(parents=True)
        for i, nodes in enumerate(batched(self.cp2k_hosts, self.nodes_per_cp2k)):
            (nodefile_path / f'local_hostfile.{i:03d}').write_text("\n".join(nodes))
        return ai_nodefile, lammps_nodefile


class AuroraConfig(SingleJobHPCConfig):
    """Configuration for running on Aurora"""

    torch_device: str = 'xpu'
    lammps_cmd: tuple[str, ...] = (
        "/lus/flare/projects/MOFA/lward/lammps-kokkos/src/lmp_macesunspotkokkos "
        "-k on g 1 -sf kk"
    ).split()
    lammps_env: dict[str, str] = {'OMP_NUM_THREADS': '1'}
    raspa_cmd: tuple[str, ...] = (
        '/lus/flare/projects/MOFA/lward/gRASPA/graspa-sycl/bin/sycl.out'
    ).split()
    raspa_version: RASPAVersion = 'graspa_sycl'
    dft_version: RASPAVersion = 'pwdft'
    cpus_per_node: int = 96
    gpus_per_node: int = 12
    lammps_per_gpu: int = 1
    max_helper_nodes: int = 256
    nodes_per_cp2k: int = 1

    worker_init: str = """
# General environment variables
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
conda deactivate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Needed for LAMMPS
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH
    """.strip()

    @computed_field()
    @property
    def dft_cmd(self) -> str:
        assert self.run_dir is not None, 'This must be run after the Parsl config is built'
        return (f'mpiexec -n {self.nodes_per_cp2k * self.gpus_per_node} --ppn {self.gpus_per_node}'
                '--cpu-bind list:1-7:8-15:16-23:24-31:32-39:40-47:53-59:60-67:68-75:76-83:84-91:92-99 '
                '--mem-bind list:0:0:0:0:0:0:1:1:1:1:1:1 --env OMP_NUM_THREADS=1 '
                '/lus/flare/projects/MOFA/lward/mof-generation-at-scale/bin/gpu_dev_compact.sh '
                '/lus/flare/projects/MOFA/lward/PWDFT/build_sycl/pwdft')

    def make_parsl_config(self) -> Config:
        assert self.num_training_nodes == 1, 'Only supporting a single training node for now'
        # Set the run dir and write nodefiles to it
        ai_nodefile, lammps_nodefile = self._make_nodefiles(self.run_dir)

        # Make a helper node file from a subset of lammps nodes
        helper_nodefile = self.run_dir / 'helper.nodes'
        helper_nodefile.write_text("\n".join(self.lammps_hosts[:self.max_helper_nodes]) + "\n")

        # Determine which cores to use for AI tasks
        sim_cores, helper_cores = self._assign_cores()

        worker_init = """
# General environment variables
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Needed for LAMMPS
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Put RASPA2 on the path
export PATH=$PATH:`realpath conda-env/bin/`

cd $PBS_O_WORKDIR
pwd
which python
hostname"""

        return Config(
            executors=[
                HighThroughputExecutor(
                    label='inf',
                    max_workers_per_node=12,
                    cpu_affinity="block",
                    available_accelerators=12,
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            f"mpiexec --no-abort-on-failure -n {len(self.ai_hosts) - self.num_training_nodes} "
                            f"--ppn 1 --hostfile {ai_nodefile} --depth=104 --cpu-bind depth"
                        ),
                        worker_init=worker_init,
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
                HighThroughputExecutor(
                    label='train',
                    max_workers_per_node=12,
                    available_accelerators=12,
                    cpu_affinity="block",
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            f"mpiexec -n 1 --ppn 1 --host {self.ai_hosts[0]} --depth=104 --cpu-bind depth"
                        ),
                        worker_init=worker_init,
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
                HighThroughputExecutor(
                    label="lammps",
                    available_accelerators=12,
                    cpu_affinity='list:' + ":".join(sim_cores),
                    prefetch_capacity=0,
                    max_workers_per_node=12,
                    provider=LocalProvider(
                        worker_init=worker_init,
                        launcher=WrappedLauncher(
                            f"mpiexec --no-abort-on-failure -n {len(self.lammps_hosts)} --ppn 1 "
                            f"--hostfile {lammps_nodefile} --depth=104 --cpu-bind depth"
                        ),
                        min_blocks=1,
                        max_blocks=1,
                    ),
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
                            f"./envs/aurora/parallel.sh {helper_nodefile}"
                        ),
                        worker_init=worker_init,
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
            ],
            run_dir=str(self.run_dir)
        )
