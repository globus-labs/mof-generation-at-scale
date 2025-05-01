from collections import deque
from concurrent.futures import as_completed
from contextlib import AbstractContextManager
from dataclasses import asdict
from dataclasses import dataclass
from functools import cached_property
from functools import partial
from functools import update_wrapper
from itertools import product
from more_itertools import batched
from more_itertools import make_decorator
from platform import node
from pathlib import Path
from random import shuffle
from threading import Lock
from typing import TextIO

import argparse
import json
import os

from tqdm import tqdm
import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider
from parsl.launchers import MpiExecLauncher

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.exceptions import TimeoutException
from colmena.thinker import BaseThinker, ResourceCounter, task_submitter, result_processor
from colmena.queue.redis import RedisQueues

from proxystore.connectors.endpoint import EndpointConnector
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, register_store

from mofa.generator import run_generator
from mofa.hpc.colmena import DiffLinkerInference
from mofa.hpc.config import HPCConfig
from mofa.model import LigandTemplate
from mofa.model import NodeDescription
from mofa.octopus import OctopusQueues
from mofa.proxyqueue import ProxyQueues

# Hard-coded defaults
_model_path = "../../tests/files/difflinker/geom_difflinker_given_anchors.ckpt"
_templates = list(Path("../../input-files/zn-paddle-pillar/").glob("template*yml"))


def test_function(model_path: Path, n_atoms: int, template: Path, n_samples: int, device: str) -> float:
    """Run a LAMMPS simulation, report runtime and resultant traj

    Args:
        model_path: Path to the model
        n_atoms: Size of the ligand to generate
        template: Starting template
        n_samples: Number of samples per batch
        device: Device on which to run generation
    Returns:
        - Runtime (s)
    """
    from mofa.model import LigandTemplate
    from time import perf_counter

    # Run
    template = LigandTemplate.from_yaml(template)
    start_time = perf_counter()
    result = list(run_generator(
        model=model_path,
        templates=[template],
        n_atoms=n_atoms,
        n_samples=n_samples,
        device=device
    ))
    run_time = perf_counter() - start_time

    return run_time

@dataclass
class GeneratorConfig:
    """Configuration for the generation tasks"""

    generator_path: Path
    """Path to the DiffLinker model"""
    templates: list[LigandTemplate]
    """The templates being generated"""
    atom_counts: list[int]
    """Number of atoms within a linker to generate"""
    min_ligand_candidates: int
    """Minimum number of candidates of each anchor needed before assembling MOFs"""

    @cached_property
    def anchor_types(self) -> set[str]:
        return set(x.anchor_type for x in self.templates)

class MOFAThinker(BaseThinker, AbstractContextManager):
    """Thinker which schedules MOF generation and testing"""

    generate_queue: deque[tuple[int, int]]
    """Queue used to ensure we generate equal numbers of each type of ligand"""

    def __init__(self,
                 queues: ColmenaQueues,
                 out_dir: Path,
                 hpc_config: HPCConfig,
                 generator_config: GeneratorConfig,
                 node_template: NodeDescription):
        """
        Args:
            queues: Queues used to communicate with task server
            collection: MongoDB collection used for storing run results
            hpc_config: Configuration for the HPC tasks
            generator_config: Configuration for the ligand generator
            node_template: Template used for MOF assembly
        """
        if hpc_config.num_workers < 2:
            raise ValueError(f'There must be at least two workers. Supplied: {hpc_config}')
        self.assemble_workers = max(1, hpc_config.num_lammps_workers // 256)  # Ensure we keep a steady stream of MOFs
        super().__init__(queues, ResourceCounter(hpc_config.num_workers + self.assemble_workers, task_types=['generation', 'lammps', 'cp2k', 'assembly']))
        self.generator_config = generator_config
        self.node_template = node_template
        self.out_dir = out_dir
        self.hpc_config = hpc_config

        self.generate_queue = deque()  # Starts with one of each task (ligand, size)
        tasks = list(product(range(len(generator_config.templates)), generator_config.atom_counts))
        shuffle(tasks)
        self.generate_queue.extend(tasks)

        # Lists used to avoid duplicates
        self.seen: set[str] = set()

        # Set aside one GPU for generation
        self.rec.reallocate(None, 'generation', self.hpc_config.number_inf_workers)

        # Output files
        self._output_files: dict[str, Path | TextIO] = {}
        self.generate_write_lock: Lock = Lock()  # Two threads write to the same generation output
        for name in ['generation-results']:
            self._output_files[name] = out_dir / f'{name}.json'

    def __enter__(self):
        """Open the output files"""
        for name, path in self._output_files.items():
            self._output_files[name] = open(path, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj in self._output_files.values():
            obj.close()

    @task_submitter(task_type='generation')
    def submit_generation(self):
        """Submit MOF generation tasks when resources are available"""

        ligand_id, size = self.generate_queue.popleft()
        ligand = self.generator_config.templates[ligand_id]
        self.queues.send_inputs(
            input_kwargs={'model': self.generator_config.generator_path, 'templates': [ligand], 'n_atoms': size},
            topic='generation',
            method='run_generator',
            task_info={
                'task': (ligand_id, size),
                'model_version': self.model_iteration
            }
        )
        self.generate_queue.append((ligand_id, size))  # Push this generation task back on the queue
        self.logger.info(f'Requested more samples of type={ligand.anchor_type} size={size}')

    @result_processor(topic='generation')
    def store_generation(self, result: Result):
        """Receive generated ligands, append to the generation queue """

        # Lookup task information
        ligand_id, size = result.task_info['task']
        anchor_type = self.generator_config.templates[ligand_id].anchor_type

        # The generation topic includes both the generator and process functions
        self.logger.info(f'Generator task method={result.method} for anchor_type={anchor_type} size={size} finished')
        if result.method == 'run_generator':  # The generate method has finished making ligands
            # Start a new task
            self.rec.release('generation')
            with self.generate_write_lock:
                print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'], flush=True)
        else:
            # The message contains the ligands
            self.logger.info(f'Pushing linkers to the processing queue. Backlog: {self.ligand_process_queue.qsize()}')
            self.ligand_process_queue.put(result)

        if not result.success:
            self.logger.warning(f'Generation task failed: {result.failure_info.exception}')


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Version of DiffLinker to run', default=_model_path)
    parser.add_argument('--template-paths', nargs='+', help='Templates to use for test seeds', default=_templates)
    parser.add_argument('--num-samples', nargs='+', type=int, help='Number of samples per batch', default=[64, 32])
    parser.add_argument('--num-atoms', nargs='+', type=int, help='Number of atoms per molecule', default=[9, 12, 15])
    parser.add_argument('--device', help='Device on which to run DiffLinker', default='cuda')
    parser.add_argument('--config', help='Which compute configuration to use', default='local')
    args = parser.parse_args()

    run_dir = Path('run') / f'parallel-{args.compute_config}-{start_time.strftime("%d%b%y%H%M%S")}-{params_hash}'
    run_dir.mkdir(parents=True)
    
    if args.queue_type == "redis":
        store = Store(name='redis', connector=RedisConnector(hostname=args.redis_host, port=6379), metrics=True)
        register_store(store)

        queues = RedisQueues(
            hostname=args.redis_host,
            topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'],
            proxystore_name='redis',
            proxystore_threshold=args.proxy_threshold
        )

    elif args.queue_type == "octopus":
        store = Store(name='redis', connector=RedisConnector(hostname=args.redis_host, port=6379), metrics=True)
        register_store(store)
    
        queues = OctopusQueues(topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'])

    elif args.queue_type == "proxystream":
        store = Store("my-store", connector=EndpointConnector([os.environ["PROXYSTORE_ENDPOINT"]]))
        register_store(store)
    
        queues = ProxyQueues(topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'])

    # Load the ligand descriptions
    templates = []
    for path in args.template_paths:
        template = LigandTemplate.from_yaml(path)
        templates.append(template)

    # Make the generator settings and the function
    generator = GeneratorConfig(
        generator_path=args.generator_path,
        atom_counts=args.molecule_sizes,
        templates=templates,
        min_ligand_candidates=args.minimum_ligand_pool
    )

    gen_func = partial(run_generator, model=args.model_path, templates=templates, n_atoms=args.num_atoms, n_samples=args.num_samples, device=args.device)
    gen_func = make_decorator(batched)(args.gen_batch_size)(gen_func)  # Wraps gen_func in a decorator in one line
    update_wrapper(gen_func, run_generator)
    gen_method = DiffLinkerInference(
        function=gen_func,
        name='run_generator',
        store_return_value=True,
        streaming_queue=queues,
        store=store
    )

    # Select the correct configuraion
    if args.config == "local":
        config = Config(executors=[HighThroughputExecutor(max_workers_per_node=1, cpu_affinity='block')], run_dir=run_dir)
    elif args.config == "polaris":
        config = Config(retries=1, executors=[
            HighThroughputExecutor(
                max_workers_per_node=4,
                cpu_affinity='block-reverse',
                available_accelerators=4,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    account='MOFA',
                    queue='debug',
                    select_options="ngpus=4",
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""
                    module use /soft/modulefiles; module load conda;
conda activate /lus/eagle/projects/Diaspora/valerie/conda_envs/mofa 
cd $PBS_O_WORKDIR
pwd
which python
hostname
                    """,
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    cpus_per_node=32,
                    walltime="1:00:00",
                )
            )
        ], run_dir=run_dir)
    elif args.config.startswith("aurora"):
        # Map processes to specific tiles or whole devices
        if args.config == "aurora":
            accel_count = 12
        elif args.config == "aurora-device":
            accel_count = 6
        else:
            raise ValueError(f'Not supported: {args.config}')

        # Ensures processes are mapped to physical cores
        workers_per_socket = accel_count // 2
        cores_per_socket = 52
        cores_per_worker = cores_per_socket // workers_per_socket
        assigned_cores = []
        for socket in range(2):
            start = cores_per_socket * socket
            assigned_cores.extend(f"{start + w * cores_per_worker}-{start + (w + 1) * cores_per_worker - 1}" for w in range(workers_per_socket))

        config = Config(
            retries=2,
            executors=[
                HighThroughputExecutor(
                    label="aurora_test",
                    available_accelerators=accel_count,
                    cpu_affinity='list:' + ":".join(assigned_cores),  # Assigns cpus in sequent
                    prefetch_capacity=0,
                    max_workers_per_node=accel_count,
                    cores_per_worker=208 // accel_count,
                    provider=PBSProProvider(
                        account="MOFA",
                        queue="debug",
                        worker_init=f"""
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
export ZE_FLAT_DEVICE_HIERARCHY={'FLAT' if accel_count == 12 else 'COMPOSITE'}
cd $PBS_O_WORKDIR
pwd
which python
hostname
                        """,
                        walltime="1:00:00",
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        scheduler_options="#PBS -l filesystems=home:flare",
                        nodes_per_block=1,
                        min_blocks=0,
                        max_blocks=1,
                        cpus_per_node=208,
                    ),
                ),
            ],
            run_dir=run_dir
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    with (run_dir / 'compute-config.json').open('w') as fp:
        json.dump(asdict(config), fp)


    if args.launch_option in ['both', 'thinker']:
        # Make the thinker
        thinker = MOFAThinker(queues,
                            hpc_config=hpc_config,
                            generator_config=generator,
                            node_template=node_template,
                            out_dir=run_dir)


    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(test_function)

        # Submit all combinations
        futures = []
        for template, n_atoms, n_samples in product(args.template_paths, args.num_atoms, args.num_samples):
            kwargs = {'template': str(template), 'n_atoms': n_atoms, 'n_samples': n_samples}
            future = test_app(args.model_path, n_atoms, template, n_samples, device=args.device)
            future.info = kwargs
            futures.append(future)

        # Store results
        for future in tqdm(as_completed(futures), total=len(futures)):
            runtime = future.result()

            # Store the result
            with open('runtimes.json', 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'model_path': str(args.model_path),
                    'device': args.device,
                    'runtime': runtime,
                    'config': args.config,
                    **future.info
                }), file=fp)
