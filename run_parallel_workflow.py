"""An example of the workflow which runs all aspects of MOF generation in parallel"""
from contextlib import AbstractContextManager
from functools import partial, update_wrapper, cached_property
from subprocess import Popen
from typing import TextIO
from csv import DictWriter
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from collections import defaultdict
from itertools import product
from datetime import datetime
from collections import deque
from queue import Queue, Empty
from platform import node
from random import shuffle, choice
from pathlib import Path
from threading import Event, Lock
import logging
import hashlib
import json
import sys

import pymongo
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, register_store
from rdkit import RDLogger
from openbabel import openbabel as ob
from pymongo import MongoClient
from pymongo.collection import Collection
from more_itertools import batched, make_decorator
from colmena.models import Result
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue import ColmenaQueues
from colmena.queue.redis import RedisQueues
from colmena.thinker import BaseThinker, result_processor, task_submitter, ResourceCounter, event_responder, agent

from mofa.assembly.assemble import assemble_mof
from mofa.assembly.validate import process_ligands
from mofa.generator import run_generator, train_generator
from mofa.model import MOFRecord, NodeDescription, LigandTemplate, LigandDescription
from mofa.scoring.geometry import LatticeParameterChange
from mofa.simulation.cp2k import CP2KRunner, compute_partial_charges
from mofa.simulation.lammps import LAMMPSRunner
from mofa.utils.conversions import write_to_string
from mofa.hpc.colmena import DiffLinkerInference
from mofa import db as mofadb
from mofa.hpc.config import configs as hpc_configs, HPCConfig

RDLogger.DisableLog('rdApp.*')
ob.obErrorLog.SetOutputLevel(0)


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


@dataclass
class TrainingConfig:
    """Configuration for retraining tasks"""

    num_epochs: int
    """Number of epochs to use for training"""
    minimum_train_size: int
    """Trigger retraining after these many computations have completed successfully"""
    maximum_train_size: int
    """How many of the top MOFs to train on"""
    best_fraction: float
    """Percentile of top MOFs to include in training set"""
    maximum_strain: float
    """Only use MOFs with strains below this value in training set"""


class MOFAThinker(BaseThinker, AbstractContextManager):
    """Thinker which schedules MOF generation and testing"""

    mof_queue: deque[MOFRecord]
    """Priority queue of MOFs to be evaluated"""
    ligand_assembly_queue: dict[str, deque[LigandDescription]]
    """Queue of the latest ligands to be generated for each type"""
    generate_queue: deque[tuple[int, int]]
    """Queue used to ensure we generate equal numbers of each type of ligand"""

    def __init__(self,
                 queues: ColmenaQueues,
                 out_dir: Path,
                 hpc_config: HPCConfig,
                 simulation_budget: int,
                 generator_config: GeneratorConfig,
                 trainer_config: TrainingConfig,
                 node_template: NodeDescription):
        if hpc_config.num_workers < 2:
            raise ValueError(f'There must be at least two workers. Supplied: {hpc_config}')
        super().__init__(queues, ResourceCounter(hpc_config.num_workers, task_types=['generation', 'lammps', 'cp2k']))
        self.generator_config = generator_config
        self.trainer_config = trainer_config
        self.node_template = node_template
        self.out_dir = out_dir
        self.simulations_left = simulation_budget
        self.hpc_config = hpc_config

        # Set up the queues
        self.mof_queue = deque(maxlen=200)  # Starts empty

        self.generate_queue = deque()  # Starts with one of each task (ligand, size)
        tasks = list(product(range(len(generator_config.templates)), generator_config.atom_counts))
        shuffle(tasks)
        self.generate_queue.extend(tasks)

        self.ligand_process_queue: Queue[Result] = Queue()  # Ligands ready to be stored in queues
        self.ligand_assembly_queue = defaultdict(lambda: deque(maxlen=200))  # Starts empty

        self.post_md_queue: Queue[Result] = Queue()  # Holds MD results ready to be stored

        # Database of completed MOFs
        self.database: dict[str, MOFRecord] = {}

        # Set aside one GPU for generation
        self.rec.reallocate(None, 'generation', self.hpc_config.number_inf_workers)
        self.rec.reallocate(None, 'lammps', self.hpc_config.num_lammps_workers)
        self.rec.reallocate(None, 'cp2k', self.hpc_config.num_cp2k_workers)

        # Settings associated with MOF assembly
        self.mofs_per_call = hpc_config.num_lammps_workers + 4
        self.make_mofs = Event()  # Signal that we need new MOFs
        self.mofs_available = Event()  # Signal that new MOFs are done

        # Settings related for training
        self.start_train = Event()
        self.initial_weights = self.generator_config.generator_path  # Store the starting weights, which we'll always use as a starting point for training
        self.num_completed = 0  # Number of MOFs which have finished training
        self.model_iteration = 0  # Which version of the model we used for generating a ligand

        # Settings related to scheduling CP2K
        self.cp2k_ready = Event()
        self.cp2k_ran = set()

        # Connect to MongoDB
        self.mongo_client = MongoClient()
        self.collection: Collection = mofadb.initialize_database(self.mongo_client)

        # Output files
        self._output_files: dict[str, Path | TextIO] = {}
        self.generate_write_lock: Lock = Lock()  # Two threads write to the same generation output
        for name in ['generation-results', 'simulation-results', 'training-results']:
            self._output_files[name] = run_dir / f'{name}.json'

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
        self.logger.info(f'Generator task type={result.method} for anchor_type={anchor_type} size={size} finished')
        if result.method == 'run_generator':
            # Start a new task
            self.rec.release('generation')
            with self.generate_write_lock:
                print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'], flush=True)
        elif result.method == 'process_ligands':
            # Process them asynchronously
            self.ligand_process_queue.put(result)

        if not result.success:
            self.logger.warning(f'Generation task failed: {result.failure_info.exception}\nStack: {result.failure_info.traceback}')

    @agent()
    def process_ligands(self):
        """Store ligands to disk and process queues"""

        while not (self.done.is_set() and self.queues.wait_until_done(timeout=0.01)):
            # Wait for a result
            try:
                result = self.ligand_process_queue.get(block=True, timeout=1)
            except Empty:
                continue

            # Lookup task information
            ligand_id, size = result.task_info['task']
            model_version = result.task_info['model_version']
            anchor_type = self.generator_config.templates[ligand_id].anchor_type

            # Put ligands in the assembly queue
            if result.success:
                # Resolve the result file
                valid_ligands, all_records = result.value
                self.logger.info(f'Received {len(all_records)} {anchor_type} ligands of size {size} from model v{model_version}, '
                                 f'{len(valid_ligands)} ({len(valid_ligands) / len(all_records) * 100:.1f}%) are valid. '
                                 f'Processing backlog: {self.ligand_process_queue.qsize()}')
                result.task_info['process_done'] = datetime.now().timestamp()  # TODO (wardlt): exalearn/colmena#135

                # Add the model version to the ligand identity
                for record in all_records:
                    record['model_version'] = model_version
                for ligand in valid_ligands:
                    ligand.metadata['model_version'] = model_version

                # Append the ligands to the task queue
                self.ligand_assembly_queue[anchor_type].extend(valid_ligands)  # Shoves old ligands out of the deque
                self.logger.info(f'Current length of {anchor_type} queue: {len(self.ligand_assembly_queue[anchor_type])}')

                # Signal that we're ready for more MOFs
                if len(valid_ligands) > 0:
                    self.make_mofs.set()

                # Store the generated ligands
                record_file = self.out_dir / 'all_ligands.csv'
                first_write = not record_file.is_file()

                with record_file.open('a') as fp:
                    writer = DictWriter(fp, all_records[0].keys())
                    if first_write:
                        writer.writeheader()
                    writer.writerows(all_records)

            # Write the result file
            with self.generate_write_lock:
                print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'], flush=True)

    @event_responder(event_name='make_mofs')
    def assemble_new_mofs(self):
        """Pull from the list of ligands and create MOFs. Runs when new MOFs are available"""

        # Check that we have enough ligands to start assembly
        for anchor_type in self.generator_config.anchor_types:
            have = len(self.ligand_assembly_queue[anchor_type])
            if have < self.generator_config.min_ligand_candidates:
                self.logger.info(f'Too few candidate for anchor_type={anchor_type}. have={have}, need={self.generator_config.min_ligand_candidates}')
                return

        # Make a certain number of attempts
        num_added = 0
        attempts_remaining = self.mofs_per_call * 4
        while num_added < self.mofs_per_call and attempts_remaining > 0:
            attempts_remaining -= 1

            # Get a sample of ligands
            ligand_choices = {}
            requirements = {'COO': 2, 'cyano': 1}  # TODO (wardlt): Do not hard code this
            for anchor_type, count in requirements.items():
                ligand_choices[anchor_type] = [choice(self.ligand_assembly_queue[anchor_type])] * count

            # Attempt assembly
            try:
                new_mof = assemble_mof(
                    nodes=[self.node_template],
                    ligands=ligand_choices,
                    topology='pcu'
                )
            except (ValueError, KeyError, IndexError):
                continue

            # Check if a duplicate
            if new_mof.name in self.database:
                continue

            # Add it to the database and work queue
            num_added += 1
            self.database[new_mof.name] = new_mof
            self.mof_queue.append(new_mof)
            self.mofs_available.set()

        self.logger.info(f'Created {num_added} new MOFs. Current queue depth: {len(self.mof_queue)}')

    @task_submitter(task_type='lammps')
    def submit_lammps(self):
        """Submit an MD simulation"""

        # Block until new MOFs are available
        if len(self.mof_queue) <= self.rec.allocated_slots('lammps'):
            self.logger.info('MOF queue is low. Triggering more to be made.')
            self.make_mofs.set()
        if len(self.mof_queue) == 0:
            self.mofs_available.clear()
            self.make_mofs.set()
            self.logger.info('No MOFs are available for simulation. Waiting')
            self.mofs_available.wait()

        to_run = self.mof_queue.pop()
        self.queues.send_inputs(
            to_run,
            method='run_molecular_dynamics',
            topic='lammps',
            task_info={'name': to_run.name}
        )
        self.logger.info(f'Started MD simulation for mof={to_run.name}. '
                         f'Simulation queue depth: {len(self.mof_queue)}.')

        if self.simulations_left == 0:
            self.done.set()
            self.logger.info('No longer submitting tasks.')

    @result_processor(topic='lammps')
    def store_lammps(self, result: Result):
        """Gather MD results, push result to post-processing queue"""

        # Trigger a new simulation
        self.rec.release('lammps')

        # Retrieve the results
        if not result.success:
            self.logger.warning(f'MD task failed: {result.failure_info.exception}\nStack: {result.failure_info.traceback}')
        else:
            self.post_md_queue.put(result)

            self.simulations_left -= 1
            self.logger.info(f'Successful computation. Budget remaining: {self.simulations_left}')
        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['simulation-results'], flush=True)

    @agent()
    def process_md_results(self):
        """Process then store the result of MD"""

        while not (self.done.is_set() and self.queues.wait_until_done(timeout=0.01)):
            # Wait for a result
            try:
                result = self.post_md_queue.get(block=True, timeout=1)
            except Empty:
                continue

            # Store the trajectory
            traj = result.value
            name = result.task_info['name']
            record = self.database[name]
            self.logger.info(f'Received a trajectory of {len(traj)} frames for mof={name}. Backlog: {self.post_md_queue.qsize()}')

            # Compute the lattice strain
            scorer = LatticeParameterChange()
            traj_vasp = [write_to_string(t, 'vasp') for t in traj]
            record.md_trajectory['uff'] = traj_vasp
            strain = scorer.score_mof(record)
            record.structure_stability['uff'] = strain
            self.logger.info(f'Lattice change after MD simulation for mof={name}: {strain * 100:.1f}%')

            # Store the result in MongoDB
            mofadb.create_records(self.collection, [record])
            self.cp2k_ready.set()

            # Determine if we should retrain
            self.num_completed += 1
            if self.num_completed >= self.trainer_config.minimum_train_size:
                self.start_train.set()  # Either starts or indicates that we have new data

    @event_responder(event_name='start_train')
    def retrain(self):
        """Retrain difflinker. Starts when we first exceed the training set size"""

        self.logger.info('Started to retrain DiffLinker')
        last_train_size = 0
        while not self.done.is_set():
            # Get the top MOFs
            sort_field = 'structure_stability.uff'
            to_include = min(int(self.collection.estimated_document_count() * self.trainer_config.best_fraction), self.trainer_config.maximum_train_size)
            self.collection.create_index(sort_field)
            cursor = (
                self.collection.find(
                    {sort_field: {'$exists': True, '$lt': self.trainer_config.maximum_strain}},
                    {'md_trajectory': 0}  # Filter out the trajectory to save I/O
                )
                .sort(sort_field, pymongo.ASCENDING)
                .limit(to_include)
            )
            examples = []
            for record in cursor:
                record.pop("_id")
                record['times'] = {}
                record['md_trajectory'] = {}
                examples.append(MOFRecord(**record))
            if len(examples) == 0 or len(examples) == last_train_size:
                self.logger.info(f'The number of training examples with strain below {self.trainer_config.maximum_strain:.2f} is the same '
                                 f'as the last time we trained DiffLinker ({last_train_size}). Waiting for more data')
                self.start_train.clear()
                self.start_train.wait()
                continue
            self.logger.info(f'Gathered the top {len(examples)} records with strain below {self.trainer_config.maximum_strain:.2f} based on stability')
            last_train_size = len(examples)  # So we know what the training set size was for the next iteration

            # Determine the run directory
            self.model_iteration += 1
            train_dir = self.out_dir / 'retraining' / f'model-v{self.model_iteration}'
            train_dir.mkdir(parents=True)
            self.logger.info(f'Preparing to retrain Difflinker in {train_dir}')

            # Submit training using the latest model
            self.queues.send_inputs(
                self.initial_weights,
                input_kwargs={'examples': examples, 'run_directory': train_dir},
                method='train_generator',
                topic='training',
                task_info={'train_size': len(examples)}
            )
            self.logger.info('Submitted training. Waiting until complete')

            # Update the model
            result = self.queues.get_result(topic='training')
            result.task_info['train_size'] = len(examples)
            if result.success:
                self.generator_config.generator_path = result.value
                self.logger.info(f'Received training result. Updated generator path to {result.value}')
            else:
                self.logger.warning(f'Training failed: {result.failure_info.exception} - {result.failure_info.traceback}')

            print(result.json(exclude={'inputs', 'value'}), file=self._output_files['training-results'], flush=True)

    @task_submitter(task_type='cp2k')
    def submit_cp2k(self):
        """Start a CP2K submission"""

        # Query the database to find the best MOF we have not run CP2K on yet
        while True:  # Runs until something gets submitted
            sort_field = 'structure_stability.uff'
            self.collection.create_index(sort_field)
            cursor = (
                self.collection.find(
                    {sort_field: {'$exists': True}},
                )
                .sort(sort_field, pymongo.ASCENDING)
            )
            for record in cursor:
                # If has been run, skip
                if record['name'] in self.cp2k_ran:
                    continue

                # Add this to the list of things which have been run
                self.cp2k_ran.add(record['name'])
                record.pop("_id")
                record = MOFRecord(**record)
                self.queues.send_inputs(
                    record,
                    method='run_optimization',
                    topic='cp2k',
                    task_info={'mof': record.name}
                )
                self.logger.info(f'Submitted {record.name} to run with CP2K')
                return

            self.logger.info('No MOFs ready for CP2K. Waiting for MD to finish')
            self.cp2k_ready.clear()
            self.cp2k_ready.wait()

    @result_processor(topic='cp2k')
    def store_cp2k(self, result: Result):
        """Store the results for the CP2K, submit any post-processing"""

        # Trigger new CP2K
        if result.method == 'run_optimization':
            self.rec.release('cp2k')

        # If it's a failure, report to the user
        mof_name = result.task_info['mof']
        if not result.success:
            self.logger.warning(f'Task {result.method} failed for {mof_name}. Exception: {result.failure_info.exception}')
        elif result.method == 'run_optimization':
            # Submit post-processing to happen
            _, cp2k_path = result.value  # Not doing anything with the Atoms yet
            self.queues.send_inputs(cp2k_path, method='compute_partial_charges', task_info=result.task_info, topic='cp2k')
            self.logger.info(f'Completed CP2K computation for {mof_name}. Runtime: {result.time.running:.2f} s. Started partial charge computation')
        elif result.method == 'compute_partial_charges':
            self.logger.info(f'Partial charges are complete for {mof_name}')
        else:
            raise ValueError(f'Method not supported: {result.method}')
        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['simulation-results'], flush=True)


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--simulation-budget', type=int, help='Number of simulations to submit before exiting')

    group = parser.add_argument_group(title='MOF Settings', description='Options related to the MOF type being generated')
    group.add_argument('--node-path', required=True, help='Path to a node record')

    group = parser.add_argument_group(title='Generator Settings', description='Options related to how the generation is performed')
    group.add_argument('--ligand-templates', required=True, nargs='+',
                       help='Path to YAML files containing a description of the ligands to be created')
    group.add_argument('--generator-path', required=True,
                       help='Path to the PyTorch files describing model architecture and weights')
    group.add_argument('--molecule-sizes', nargs='+', type=int, default=list(range(6, 21)),
                       help='Sizes of molecules we should generate')
    group.add_argument('--num-samples', type=int, default=16, help='Number of molecules to generate at each size')
    group.add_argument('--gen-batch-size', type=int, default=16, help='Number of ligands to stream per batch')

    group = parser.add_argument_group('Retraining Settings', description='How often to retain, what to train on, etc')
    group.add_argument('--generator-config-path', required=True, help='Path to the generator training configuration')
    group.add_argument('--retrain-freq', type=int, default=8, help='Trigger retraining after these many successful computations')
    group.add_argument('--maximum-train-size', type=int, default=256, help='Maximum number of MOFs to use for retraining')
    group.add_argument('--num-epochs', type=int, default=128, help='Number of training epochs')
    group.add_argument('--best-fraction', type=float, default=0.5, help='What percentile of MOFs to include in training')
    group.add_argument('--maximum-strain', type=float, default=0.5, help='Maximum strain allowed MOF used in training set')

    group = parser.add_argument_group(title='Assembly Settings', description='Options related to MOF assembly')
    group.add_argument('--max-assemble-attempts', default=100,
                       help='Maximum number of attempts to create a MOF')
    group.add_argument('--minimum-ligand-pool', type=int, default=4, help='Minimum number of ligands before MOF assembly')

    group = parser.add_argument_group(title='Simulation Settings Settings', description='Options related to property calculations')
    group.add_argument('--md-timesteps', default=100000, help='Number of timesteps for the UFF MD simulation', type=int)
    group.add_argument('--md-snapshots', default=100, help='Maximum number of snapshots during MD simulation', type=int)
    group.add_argument('--retain-lammps', action='store_true', help='Keep LAMMPS output files after it finishes')
    group.add_argument('--dft-opt-steps', default=8, help='Maximum number of DFT optimization steps', type=int)

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--lammps-on-ramdisk', action='store_true', help='Write LAMMPS outputs to a RAM Disk')
    group.add_argument('--compute-config', default='local', help='Configuration for the HPC system')
    group.add_argument('--ai-fraction', default=0.1, type=float, help='Fraction of workers devoted to AI tasks')
    group.add_argument('--dft-fraction', default=0.1, type=float, help='Fraction of workers devoted to DFT tasks')
    group.add_argument('--redis-host', default=node(), help='Host for the Redis server')

    args = parser.parse_args()

    # Load the example MOF
    # TODO (wardlt): Use Pydantic for JSON I/O
    node_template = NodeDescription(**json.loads(Path(args.node_path).read_text()))

    # Make the run directory
    run_params = args.__dict__.copy()
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    run_dir = Path('run') / f'parallel-{args.compute_config}-{start_time.strftime("%d%b%y%H%M%S")}-{params_hash}'
    run_dir.mkdir(parents=True)

    # Open a proxystore with Redis
    store = Store(name='redis', connector=RedisConnector(hostname=args.redis_host, port=6379), metrics=True)
    register_store(store)

    # Configure to a use Redis queue, which allows streaming results form other nodes
    queues = RedisQueues(
        hostname=args.redis_host,
        topics=['generation', 'lammps', 'cp2k', 'training'],
        proxystore_name='redis',
        proxystore_threshold=10000
    )

    # Load the ligand descriptions
    templates = []
    for path in args.ligand_templates:
        template = LigandTemplate.from_yaml(path)
        templates.append(template)

    # Load the HPC configuration
    hpc_config = hpc_configs[args.compute_config]()
    hpc_config.ai_fraction = args.ai_fraction
    hpc_config.dft_fraction = args.dft_fraction

    # Make the Parsl configuration
    config = hpc_config.make_parsl_config(run_dir)
    with (run_dir / 'compute-config.json').open('w') as fp:
        json.dump(asdict(hpc_config), fp)

    # Make the generator settings and the function
    generator = GeneratorConfig(
        generator_path=args.generator_path,
        atom_counts=args.molecule_sizes,
        templates=templates,
        min_ligand_candidates=args.minimum_ligand_pool
    )
    gen_func = partial(run_generator, n_samples=args.num_samples, device=hpc_config.torch_device)
    gen_func = make_decorator(batched)(args.gen_batch_size)(gen_func)  # Wraps gen_func in a decorator in one line
    update_wrapper(gen_func, run_generator)
    gen_method = DiffLinkerInference(
        function=gen_func,
        name='run_generator',
        store_return_value=True,
        streaming_queue=queues,
        store=store
    )

    # Make the training function
    trainer = TrainingConfig(
        maximum_train_size=args.maximum_train_size,
        num_epochs=args.num_epochs,
        minimum_train_size=args.retrain_freq,
        best_fraction=args.best_fraction,
        maximum_strain=args.maximum_strain
    )
    train_func = partial(train_generator, config_path=args.generator_config_path,
                         num_epochs=trainer.num_epochs, device=hpc_config.torch_device)
    update_wrapper(train_func, train_generator)

    # Make the LAMMPS function
    lmp_runner = LAMMPSRunner(hpc_config.lammps_cmd,
                              lmp_sims_root_path='/dev/shm/lmp_run' if args.lammps_on_ramdisk else str(run_dir / 'lmp_run'),
                              lammps_environ=hpc_config.lammps_env,
                              delete_finished=not args.retain_lammps)
    md_fun = partial(lmp_runner.run_molecular_dynamics, timesteps=args.md_timesteps, report_frequency=max(1, args.md_timesteps / args.md_snapshots))
    update_wrapper(md_fun, lmp_runner.run_molecular_dynamics)

    # Make the CP2K function
    cp2k_runner = CP2KRunner(
        cp2k_invocation=hpc_config.cp2k_cmd,
        run_dir=run_dir / 'cp2k-runs'
    )
    cp2k_fun = partial(cp2k_runner.run_optimization, steps=args.dft_opt_steps)  # Optimizes starting from assembled structure
    update_wrapper(cp2k_fun, cp2k_runner.run_optimization)

    # Launch MongoDB as a subprocess
    mongo_dir = run_dir / 'db'
    mongo_dir.mkdir(parents=True)
    mongo_proc = Popen(
        f'mongod --dbpath {mongo_dir.absolute()} --logpath {(run_dir / "mongo.log").absolute()}'.split(),
        stderr=(run_dir / 'mongo.err').open('w')
    )

    # Make the thinker
    thinker = MOFAThinker(queues,
                          hpc_config=hpc_config,
                          generator_config=generator,
                          trainer_config=trainer,
                          simulation_budget=args.simulation_budget,
                          node_template=node_template,
                          out_dir=run_dir)

    # Turn on logging
    my_logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'run.log')]
    for logger in [my_logger, thinker.logger]:
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    my_logger.info(f'Running job in {run_dir} on {hpc_config.num_workers} workers')

    # Save the run parameters to disk
    (run_dir / 'params.json').write_text(json.dumps(run_params))

    # Launch the thinker and task server
    doer = ParslTaskServer(
        methods=[
            (gen_method, {'executors': hpc_config.inference_executors}),
            (train_func, {'executors': hpc_config.train_executors}),
            (md_fun, {'executors': hpc_config.lammps_executors}),
            (cp2k_fun, {'executors': hpc_config.cp2k_executors}),
            (compute_partial_charges, {'executors': hpc_config.helper_executors}),
            (process_ligands, {'executors': hpc_config.helper_executors})
        ],
        queues=queues,
        config=config
    )

    # Launch the utilization logging
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True)
    util_proc = hpc_config.launch_monitor_process(log_dir.absolute())
    if util_proc.poll() is not None:
        raise ValueError('Monitor process failed to run!')
    my_logger.info(f'Launched monitoring process. pid={util_proc.pid}')

    try:
        doer.start()
        my_logger.info(f'Running parsl. pid={doer.pid}')

        with thinker:  # Opens the output files
            thinker.run()
    finally:
        queues.send_kill_signal()

        # Kill the services launched during workflow
        util_proc.terminate()
        mongo_proc.terminate()
        mongo_proc.poll()

        # Close the proxy store
        store.close()
