"""An example of the workflow which runs all aspects of MOF generation in parallel"""
from contextlib import AbstractContextManager
from functools import partial, update_wrapper
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
from threading import Event
import logging
import hashlib
import json
import sys

import pymongo
from rdkit import Chem
from rdkit import RDLogger
from openbabel import openbabel as ob
from pymongo import MongoClient
from pymongo.collection import Collection
from more_itertools import batched, make_decorator
from colmena.models import Result
from colmena.models.methods import PythonGeneratorMethod
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue import ColmenaQueues
from colmena.queue.redis import RedisQueues
from colmena.thinker import BaseThinker, result_processor, task_submitter, ResourceCounter, event_responder, agent

from mofa.assembly.assemble import assemble_mof
from mofa.generator import run_generator, train_generator
from mofa.model import MOFRecord, NodeDescription, LigandTemplate, LigandDescription
from mofa.scoring.geometry import LatticeParameterChange
from mofa.simulation.lammps import LAMMPSRunner
from mofa.utils.conversions import write_to_string
from mofa.utils.xyz import xyz_to_mol
from mofa import db as mofadb
from mofa.hpc.config import configs as hpc_configs, HPCConfig

RDLogger.DisableLog('rdApp.*')
ob.obErrorLog.SetOutputLevel(0)


def process_ligand(ligand: LigandDescription) -> dict:
    """Assess whether a ligand is valid and prepare it for the next step

    Args:
        ligand: Ligand to be processed
    Returns:
        Record describing the ligand suitable for serialization into CSV file
    """
    # Store the ligand information for debugging purposes
    record = {"anchor_type": ligand.anchor_type,
              "smiles": None,
              "xyz": ligand.xyz,
              "prompt_atoms": ligand.prompt_atoms,
              "valid": False}

    # Try constrained optimization on the ligand
    try:
        ligand.anchor_constrained_optimization()
    except (ValueError, AttributeError,):
        return record

    # Parse each new ligand, determine whether it is a single molecule
    try:
        mol = xyz_to_mol(ligand.xyz)
    except (ValueError,):
        return record

    # Store the smiles string
    Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol)
    record['smiles'] = smiles

    if len(Chem.GetMolFrags(mol)) > 1:
        return record

    # If passes, save the SMILES string and store the molecules
    ligand.smiles = Chem.MolToSmiles(mol)

    # Update the record, add to ligand queue and prepare it for writing to disk
    record['valid'] = True
    return record


@dataclass
class GeneratorConfig:
    """Configuration for the generation tasks"""

    generator_path: Path
    """Path to the DiffLinker model"""
    templates: list[LigandTemplate]
    """The templates being generated"""
    atom_counts: list[int]
    """Number of atoms within a linker to generate"""


@dataclass
class TrainingConfig:
    """Configuration for retraining tasks"""

    num_epochs: int
    """Number of epochs to use for training"""
    retrain_freq: int
    """Trigger retraining after these many computations have completed successfully"""
    maximum_train_size: int
    """How many of the top MOFs to train on"""
    best_fraction: float
    """Percentile of top MOFs to include in training set"""


class MOFAThinker(BaseThinker, AbstractContextManager):
    """Thinker which schedules MOF generation and testing"""

    mof_queue: deque[MOFRecord]
    """Priority queue of MOFs to be evaluated"""
    ligand_queue: dict[str, deque[LigandDescription]]
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
        super().__init__(queues, ResourceCounter(hpc_config.num_workers, task_types=['generation', 'simulation']))
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

        self.ligand_queue = defaultdict(lambda: deque(maxlen=200))  # Starts empty

        self.post_md_queue: Queue[Result] = Queue()  # Holds MD results ready to be stored

        # Database of completed MOFs
        self.database: dict[str, MOFRecord] = {}

        # Set aside one GPU for generation
        self.rec.reallocate(None, 'generation', self.hpc_config.num_ai_workers)
        self.rec.reallocate(None, 'simulation', self.hpc_config.num_sim_workers)

        # Settings associated with MOF assembly
        self.mofs_per_call = hpc_config.num_sim_workers + 4
        self.make_mofs = Event()  # Signal that we need new MOFs
        self.mofs_available = Event()  # Signal that new MOFs are done

        # Settings related for training
        self.start_train = Event()
        self.num_completed = 0  # Number of MOFs which have finished training
        self.model_iteration = 0  # Which version of the model we used for generating a ligand

        # Connect to MongoDB
        self.mongo_client = MongoClient()
        self.collection: Collection = mofadb.initialize_database(self.mongo_client)

        # Output files
        self._output_files: dict[str, Path | TextIO] = {}
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

        # If "complete," then this is signifying the generator has finished and should not contain any ligands
        if result.complete:
            # Start a new task
            self.rec.release('generation')
            self.logger.info(f'Generator task for anchor_type={anchor_type} size={size} finished')

            print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'], flush=True)
            return

        # Retrieve the results
        if not result.success:
            self.logger.warning(f'Generation task failed: {result.failure_info.exception}\nStack: {result.failure_info.traceback}')
            print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'])
            return
        new_ligands: list[LigandDescription] = result.value
        self.logger.info(f'Received {len(new_ligands)} new ligands of anchor_type={anchor_type} size={size}')

        # Check if they are valid
        #  TODO (wardlt): Make this parallel
        all_records = []
        valid_count = 0

        for ligand in new_ligands:
            record = process_ligand(ligand)
            ligand.metadata['model_version'] = result.task_info['model_version']
            all_records.append(record)
            if record['valid']:
                valid_count += 1
                self.ligand_queue[anchor_type].append(ligand)  # Shoves old ligands out of the deque

            # TODO (wardlt): Remove this hack when DiffLinker works with COO properly
            if anchor_type != "COO":
                # begin of swap cyano for COO
                coo_ligand = ligand.swap_cyano_with_COO()
                coo_record = process_ligand(coo_ligand)
                all_records.append(coo_record)
                if coo_record['valid']:
                    self.ligand_queue["COO"].append(coo_ligand)
        self.logger.info(f'{valid_count} of {len(new_ligands)} are valid. ({valid_count / len(new_ligands) * 100:.1f}%)')

        # Write record of generation tasks to disk
        if valid_count > 0:
            # Signal that we're ready for more MOFs
            self.make_mofs.set()

        # Store the generated ligands
        record_file = self.out_dir / 'all_ligands.csv'
        first_write = not record_file.is_file()

        with record_file.open('a') as fp:
            writer = DictWriter(fp, all_records[0].keys())
            if first_write:
                writer.writeheader()
            writer.writerows(all_records)

        # Store the task information
        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'], flush=False)

    @event_responder(event_name='make_mofs')
    def assemble_new_mofs(self):
        """Pull from the list of ligands and create MOFs. Runs when new MOFs are available"""

        # Check that we have enough ligands to start assembly
        requirements = {'COO': 2, 'cyano': 1}
        for anchor_type, count in requirements.items():
            have = len(self.ligand_queue[anchor_type])
            if have < count:
                self.logger.info(f'Too few candidate for anchor_type={anchor_type}. have={have}, need={count}')
                return

        # Make a certain number of attempts
        num_added = 0
        attempts_remaining = self.mofs_per_call * 4
        while num_added < self.mofs_per_call and attempts_remaining > 0:
            attempts_remaining -= 1

            # Get a sample of ligands
            ligand_choices = {}
            for anchor_type, count in requirements.items():
                ligand_choices[anchor_type] = [choice(self.ligand_queue[anchor_type])] * count

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

    @task_submitter(task_type='simulation')
    def submit_simulation(self):
        """Submit an MD simulation"""

        # Block until new MOFs are available
        if len(self.mof_queue) <= self.rec.allocated_slots('simulation'):
            self.logger.info('MOF queue is low. Triggering more to be made.')
            self.make_mofs.set()
        if len(self.mof_queue) == 0:
            self.mofs_available.clear()
            self.make_mofs.set()
            self.logger.info('No MOFs are available for simulation. Waiting')
            self.mofs_available.wait()

        to_run = self.mof_queue.popleft()
        self.queues.send_inputs(
            to_run,
            method='run_molecular_dynamics',
            topic='simulation',
            task_info={'name': to_run.name}
        )
        self.logger.info(f'Started MD simulation for mof={to_run.name}. '
                         f'Simulation queue depth: {len(self.mof_queue)}.')

        if self.simulations_left == 0:
            self.done.set()
            self.logger.info('No longer submitting tasks.')

    @result_processor(topic='simulation')
    def store_simulation(self, result: Result):
        """Gather MD results, push result to post-processing queue"""

        # Trigger a new simulation
        self.rec.release('simulation')

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

            # Determine if we should retrain
            self.num_completed += 1
            if self.num_completed % self.trainer_config.retrain_freq == 0:
                self.start_train.set()
                self.logger.info('Triggered retraining')

    @event_responder(event_name='start_train')
    def retrain(self):
        """Retrain difflinker"""

        # Determine the run directory
        self.model_iteration += 1
        train_dir = self.out_dir / 'retraining' / f'model-v{self.model_iteration}'
        train_dir.mkdir(parents=True)
        self.logger.info(f'Preparing to retrain Difflinker in {train_dir}')

        # Get the top MOFs
        sort_field = 'structure_stability.uff'
        to_include = min(int(self.collection.estimated_document_count() * self.trainer_config.best_fraction), self.trainer_config.maximum_train_size)
        self.collection.create_index(sort_field)
        cursor = (
            self.collection.find(
                {sort_field: {'$exists': True}},
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
        self.logger.info(f'Gathered the top {len(examples)} records based on stability')

        # Submit training using the latest model
        self.queues.send_inputs(
            self.generator_config.generator_path,
            input_kwargs={'examples': examples, 'run_directory': train_dir},
            method='train_generator',
            topic='training',
        )
        self.logger.info('Submitted task for execution on any worker. Waiting until complete')

        # Update the model
        result = self.queues.get_result(topic='training')
        result.task_info['train_size'] = len(examples)
        assert result.success, f'Training failed: {result.failure_info.exception} - {result.failure_info.traceback}'
        self.generator_config.generator_path = result.value
        self.logger.info(f'Received training result. Updated generator path to {result.value}')

        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['training-results'], flush=True)


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
    group.add_argument('--molecule-sizes', nargs='+', type=int, default=(10, 11, 12), help='Sizes of molecules we should generate')
    group.add_argument('--num-samples', type=int, default=16, help='Number of molecules to generate at each size')
    group.add_argument('--gen-batch-size', type=int, default=4, help='Number of ligands to stream per batch')

    group = parser.add_argument_group('Retraining Settings', description='How often to retain, what to train on, etc')
    group.add_argument('--generator-config-path', required=True, help='Path to the generator training configuration')
    group.add_argument('--retrain-freq', type=int, default=8, help='Trigger retraining after these many successful computations')
    group.add_argument('--maximum-train-size', type=int, default=256, help='Maximum number of MOFs to use for retraining')
    group.add_argument('--num-epochs', type=int, default=128, help='Number of training epochs')
    group.add_argument('--best-fraction', type=float, default=0.5, help='What percentile of MOFs to include in training')

    group = parser.add_argument_group(title='Assembly Settings', description='Options related to MOF assembly')
    group.add_argument('--max-assemble-attempts', default=100,
                       help='Maximum number of attempts to create a MOF')

    group = parser.add_argument_group(title='Simulation Settings Settings', description='Options related to MOF assembly')
    group.add_argument('--md-timesteps', default=100000, help='Number of timesteps for the UFF MD simulation', type=int)
    group.add_argument('--md-snapshots', default=100, help='Maximum number of snapshots during MD simulation', type=int)

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--compute-config', default='local', help='Configuration for the HPC system')
    group.add_argument('--sim-fraction', default=0.9, type=float, help='Fraction of workers devoted to AI tasks')
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

    # Configure to a use Redis queue, which allows streaming results form other nodes
    queues = RedisQueues(hostname=args.redis_host, topics=['generation', 'simulation', 'training'])

    # Load the ligand descriptions
    templates = []
    for path in args.ligand_templates:
        template = LigandTemplate.from_yaml(path)
        templates.append(template)

    # Load the HPC configuration
    hpc_config = hpc_configs[args.compute_config]()
    hpc_config.sim_fraction = args.sim_fraction
    with (run_dir / 'compute-config.json').open('w') as fp:
        json.dump(asdict(hpc_config), fp)

    # Make the generator settings and the function
    generator = GeneratorConfig(
        generator_path=args.generator_path,
        atom_counts=args.molecule_sizes,
        templates=templates
    )
    gen_func = partial(run_generator, n_samples=args.num_samples, device=hpc_config.torch_device)
    gen_func = make_decorator(batched)(args.gen_batch_size)(gen_func)  # Wraps gen_func in a decorator in one line
    update_wrapper(gen_func, run_generator)
    gen_method = PythonGeneratorMethod(
        function=gen_func,
        name='run_generator',
        store_return_value=True,
        streaming_queue=queues
    )

    # Make the training function
    trainer = TrainingConfig(
        maximum_train_size=args.maximum_train_size,
        num_epochs=args.num_epochs,
        retrain_freq=args.retrain_freq,
        best_fraction=args.best_fraction
    )
    train_func = partial(train_generator, config_path=args.generator_config_path,
                         num_epochs=trainer.num_epochs, device=hpc_config.torch_device)
    update_wrapper(train_func, train_generator)

    # Make the LAMMPS function
    lmp_runner = LAMMPSRunner(hpc_config.lammps_cmd,
                              lmp_sims_root_path=str(run_dir / 'lmp_run'),
                              lammps_environ=hpc_config.lammps_env)
    md_fun = partial(lmp_runner.run_molecular_dynamics, timesteps=args.md_timesteps, report_frequency=max(1, args.md_timesteps / args.md_snapshots))
    update_wrapper(md_fun, lmp_runner.run_molecular_dynamics)

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

    # Make the Parsl configuration
    config = hpc_config.make_parsl_config(run_dir)

    # Launch the thinker and task server
    doer = ParslTaskServer(
        methods=[
            (gen_method, {'executors': hpc_config.ai_executors}),
            (train_func, {'executors': hpc_config.ai_executors}),
            (md_fun, {'executors': hpc_config.sim_executors})
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
