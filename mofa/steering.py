"""Steering algorithm used by the parallel workflow"""
import shutil
from collections import deque, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from itertools import product
from pathlib import Path
from queue import Queue, Empty
from random import shuffle
from threading import Event, Lock
from typing import TextIO

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.exceptions import TimeoutException
from colmena.thinker import BaseThinker, ResourceCounter, task_submitter, result_processor, agent, event_responder
from pymongo.collection import Collection

from mofa import db as mofadb
from mofa.finetune.difflinker import DiffLinkerCurriculum
from mofa.hpc.config import HPCConfig

from mofa.model import LigandTemplate, MOFRecord, LigandDescription, NodeDescription
from mofa.scoring.geometry import LatticeParameterChange
from mofa.selection.dft import DFTSelector
from mofa.selection.md import MDSelector
from mofa.utils.conversions import write_to_string


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
    curriculum: DiffLinkerCurriculum
    """Method used to create the dataset"""


@dataclass
class SimulationConfig:
    """Configuration for the simulation inputs"""

    md_level: str = 'mace'
    """Name of the accuracy level at which to run MD"""
    md_length: int = 3000
    """Number of timesteps to perform with MD at different levels"""
    md_report: int = 1000
    """How frequently to report MD frames"""


class MOFAThinker(BaseThinker, AbstractContextManager):
    """Thinker which schedules MOF generation and testing"""

    stability_queue: deque[MOFRecord]
    """Priority queue of MOFs to be evaluated for stability"""
    ligand_assembly_queue: dict[str, deque[LigandDescription]]
    """Queue of the latest ligands to be generated for each type"""
    generate_queue: deque[tuple[int, int]]
    """Queue used to ensure we generate equal numbers of each type of ligand"""

    def __init__(self,
                 queues: ColmenaQueues,
                 collection: Collection,
                 out_dir: Path,
                 hpc_config: HPCConfig,
                 simulation_budget: int,
                 generator_config: GeneratorConfig,
                 trainer_config: TrainingConfig,
                 simulation_config: SimulationConfig,
                 md_selector: MDSelector,
                 dft_selector: DFTSelector,
                 node_template: NodeDescription):
        """
        Args:
            queues: Queues used to communicate with task server
            collection: MongoDB collection used for storing run results
            hpc_config: Configuration for the HPC tasks
            simulation_budget: Number of LAMMPS simulations to run
            generator_config: Configuration for the ligand generator
            trainer_config: Configuration for training DiffLinker
            simulation_config: Configuration for the simulation tasks
            md_selector: Method used to select which LAMMPS simulations to perform
            dft_selector: Method used to select which DFT simulations to perform
            node_template: Template used for MOF assembly
        """
        if hpc_config.num_workers < 2:
            raise ValueError(f'There must be at least two workers. Supplied: {hpc_config}')
        self.assemble_workers = max(1, hpc_config.num_lammps_workers // 256)  # Ensure we keep a steady stream of MOFs
        super().__init__(queues, ResourceCounter(hpc_config.num_workers + self.assemble_workers, task_types=['generation', 'lammps', 'cp2k', 'assembly']))
        self.generator_config = generator_config
        self.trainer_config = trainer_config
        self.node_template = node_template
        self.out_dir = out_dir
        self.simulations_left = simulation_budget
        self.hpc_config = hpc_config
        self.sim_config = simulation_config
        self.md_selector = md_selector
        self.dft_selector = dft_selector

        # Set up the queues
        self.stability_queue = deque(maxlen=8 * self.hpc_config.num_lammps_workers)  # Starts empty

        self.generate_queue = deque()  # Starts with one of each task (ligand, size)
        tasks = list(product(range(len(generator_config.templates)), generator_config.atom_counts))
        shuffle(tasks)
        self.generate_queue.extend(tasks)

        self.ligand_process_queue: Queue[Result] = Queue()  # Ligands ready to be stored in queues
        self.ligand_assembly_queue = defaultdict(lambda: deque(maxlen=50 * self.hpc_config.number_inf_workers))  # Starts empty

        self.post_md_queue: Queue[Result] = Queue()  # Holds MD results ready to be stored

        # Lists used to avoid duplicates
        self.seen: set[str] = set()

        # Set aside one GPU for generation
        self.rec.reallocate(None, 'generation', self.hpc_config.number_inf_workers)
        self.rec.reallocate(None, 'lammps', self.hpc_config.num_lammps_workers)
        self.rec.reallocate(None, 'cp2k', self.hpc_config.num_cp2k_workers)
        self.rec.reallocate(None, 'assembly', self.assemble_workers)

        # Settings associated with MOF assembly
        self.mofs_per_call = min(hpc_config.num_lammps_workers + 4, 128)
        self.make_mofs = Event()  # Signal that we need new MOFs
        self.mofs_available = Event()  # Signal that new MOFs are done

        # Settings related for training
        self.start_train = Event()
        self.initial_weights = self.generator_config.generator_path  # Store the starting weights, which we'll always use as a starting point for training
        self.num_lammps_completed = 0  # Number of MOFs which have finished stability
        self.num_raspa_completed = 0  # Number for which we have gas storage
        self.model_iteration = 0  # Which version of the model we used for generating a ligand

        # Settings related to scheduling CP2K
        self.cp2k_ready = Event()

        # Connect to MongoDB
        self.collection = collection

        # Output files
        self._output_files: dict[str, Path | TextIO] = {}
        self.generate_write_lock: Lock = Lock()  # Two threads write to the same generation output
        for name in ['generation-results', 'simulation-results', 'training-results', 'assembly-results']:
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
            else:
                self.logger.warning(f'Generation failed: {result.failure_info.exception}')

            # Write the result file
            with self.generate_write_lock:
                print(result.json(exclude={'inputs', 'value'}), file=self._output_files['generation-results'], flush=True)

    @event_responder(event_name='make_mofs')
    def submit_assembly(self):
        """Pull from the list of ligands and create MOFs"""

        # Check that we have enough ligands to start assembly
        while True:
            for anchor_type in self.generator_config.anchor_types:
                have = len(self.ligand_assembly_queue[anchor_type])
                if have < self.generator_config.min_ligand_candidates:
                    self.make_mofs.clear()
                    while not self.make_mofs.wait(timeout=1.):
                        if self.done.is_set():
                            return
                    break
            else:
                break

        # Submit a batch of assembly tasks
        for i in range(self.assemble_workers):
            self.rec.acquire('assembly', 1)
            self.queues.send_inputs(
                dict((k, list(v)) for k, v in self.ligand_assembly_queue.items()),
                [self.node_template],
                self.mofs_per_call,
                4,
                method='assemble_many',
                topic='assembly',
                task_info={'to_make': self.mofs_per_call, 'worker': i}
            )

    @result_processor(topic='assembly')
    def store_assembly(self, result: Result):
        """Store the MOFs in the ready for LAMMPS queue"""

        # Trigger a new one to run
        self.rec.release('assembly')

        # Skip if it failed
        if not result.success:
            self.logger.warning(f'Assembly task failed: {result.failure_info.exception}')
            return

        # Add them to the database
        num_added = 0
        for new_mof in result.value:
            # Avoid duplicates
            if new_mof.name in self.seen:
                continue

            # Add it to the database and work queue
            num_added += 1
            self.seen.add(new_mof.name)
            self.stability_queue.append(new_mof)
            self.mofs_available.set()

        self.logger.info(f'Created {num_added} new MOFs. Current queue depth: {len(self.stability_queue)}')

        # Save the result
        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['assembly-results'], flush=True)

    @task_submitter(task_type='lammps')
    def submit_lammps(self):
        """Submit an MD simulation"""

        # Block until new MOFs are available
        if len(self.stability_queue) <= self.rec.allocated_slots('lammps'):
            self.logger.info('MOF queue is low. Triggering more to be made.')
            self.make_mofs.set()
        if len(self.stability_queue) == 0 and self.md_selector.count_available() == 0:
            self.mofs_available.clear()
            self.make_mofs.set()
            self.logger.info('No MOFs are available for simulation. Waiting')

            while not self.mofs_available.wait(timeout=0.5):
                if self.done.is_set():
                    return

        # Determine which to run next and submit it
        try:
            to_run = self.md_selector.select_next(self.stability_queue)
        except ValueError:
            self.logger.warning(f'Tried to select when none available. {self.collection.estimated_document_count()} docs')
            self.logger.warning(f'{self.md_selector.count_available()} are available for MD')
            raise

        if 'relaxed' not in to_run.times:
            self.queues.send_inputs(
                to_run,
                method='run_optimization_ff',
                topic='lammps',
                task_info={'name': to_run.name,
                           'level': self.sim_config.md_level}
            )
            mofadb.create_records(self.collection, [to_run])
            self.logger.info(f'Started initial relaxation for mof={to_run.name}')
        else:
            self.queues.send_inputs(
                to_run, self.sim_config.md_length,
                method='run_molecular_dynamics',
                topic='lammps',
                task_info={'name': to_run.name,
                           'length': self.sim_config.md_length,
                           'level': self.sim_config.md_level}
            )
            self.logger.info(f'Started MD simulation for mof={to_run.name}. '
                             f'Simulation queue depth: {len(self.stability_queue)}.')

        # Mark that it's in progress
        mofadb.mark_in_progress(self.collection, to_run, 'stability')

    @result_processor(topic='lammps')
    def store_lammps(self, result: Result):
        """Gather MD results, push result to post-processing queue"""

        # Trigger a new simulation
        self.rec.release('lammps')

        # Retrieve the results
        if not result.success:
            self.logger.warning(f'MD task failed: {result.failure_info.exception}')
        else:
            self.post_md_queue.put(result)
            self.simulations_left -= 1
            self.logger.info(f'Successful computation. Budget remaining: {self.simulations_left}')

        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['simulation-results'], flush=True)

        # Check if termination conditions are met
        if self.simulations_left == 0:
            self.done.set()
            self.logger.info('Finished running LAMMPS simulations. Will start finishing all remaining work')

    @agent()
    def process_md_results(self):
        """Process then store the result of MD"""

        while not (self.done.is_set() and self.queues.wait_until_done(timeout=0.01)):
            # Wait for a result
            try:
                result = self.post_md_queue.get(block=True, timeout=1)
            except Empty:
                continue

            # Pull the record
            name = result.task_info['name']
            level = result.task_info['level']
            record = mofadb.get_records(self.collection, [name])[0]

            # Route based on whether it was relaxation or MD
            if result.method == 'run_molecular_dynamics':
                traj = result.value
                self.logger.info(f'Received a trajectory of {len(traj)} frames for mof={name} at level={level}.'
                                 f' Backlog: {self.post_md_queue.qsize()}')

                # Compute the lattice strain
                scorer = LatticeParameterChange(md_level=level)
                traj = [(i, write_to_string(t, 'vasp')) for i, t in traj]
                if level not in record.md_trajectory:
                    record.md_trajectory[level] = []
                record.md_trajectory[level].extend(traj)

                latest_length, _ = record.md_trajectory[level][-1]
                strain = scorer.score_mof(record)
                record.structure_stability[level] = strain
                record.times['md-done'] = datetime.now()
                self.logger.info(f'Lattice change after {latest_length} timesteps of MD for mof={name} level={level}: {strain * 100:.1f}%')

                # Upload the strain result to the DB
                mofadb.update_records(self.collection, [record])

                # Determine if we should retrain
                self.num_lammps_completed += 1
                if self.num_lammps_completed >= self.trainer_config.curriculum.min_strain_counts \
                        and self.num_raspa_completed < self.trainer_config.curriculum.min_gas_counts:
                    self.start_train.set()  # Either starts or indicates that we have new data

            elif result.method == 'run_optimization_ff':
                self.logger.info(f'Completed a relaxation for for mof={name} level={level}')
                relaxed, _ = result.value

                # Update the structure in the database and mark as relaxed
                relaxed_vasp = write_to_string(relaxed, 'vasp')
                self.collection.update_one({'name': name}, {
                    '$set': {
                        'structure': relaxed_vasp,
                        'times.relaxed': datetime.now()
                    }
                })
            else:
                raise ValueError(f'Unrecognized method: {result.method}')

            # Make record available for next steps
            mofadb.mark_completed(self.collection, record, 'stability')
            self.cp2k_ready.set()
            self.mofs_available.set()

    @event_responder(event_name='start_train')
    def retrain(self):
        """Retrain difflinker. Starts when we first exceed the training set size"""

        self.logger.info('Started to retrain DiffLinker')
        last_train_size = 0
        while not self.done.is_set():
            # Determine how to select the best MOFs
            examples = self.trainer_config.curriculum.get_training_set()
            if (len(examples) == 0 or len(examples) == last_train_size) and len(examples) < self.trainer_config.curriculum.max_size:
                self.logger.info(f'The number of training examples for with strain below {self.trainer_config.curriculum.max_strain:.2f}'
                                 f' ({len(examples)}) is the same as the last time we trained DiffLinker ({last_train_size}). Waiting for more data')
                self.start_train.clear()
                while not self.start_train.wait(timeout=1.):
                    if self.done.is_set():
                        break
                continue
            self.logger.info(f'Gathered a training set of {len(examples)} examples.')
            last_train_size = len(examples)  # So we know what the training set size was for the next iteration

            # Determine the run directory
            attempt_id = self.model_iteration + 1
            train_dir = self.out_dir / 'retraining' / f'model-v{attempt_id}'
            train_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f'Preparing to retrain Difflinker in {train_dir}')

            # Submit training using the latest model
            for i in range(self.hpc_config.num_training_ranks):
                self.queues.send_inputs(
                    self.initial_weights,
                    input_kwargs={'examples': examples, 'run_directory': train_dir},
                    method='train_generator',
                    topic='training',
                    task_info={'train_size': len(examples), 'rank': i}
                )
            self.logger.info(f'Submitted {self.hpc_config.num_training_ranks} training tasks. Waiting until complete')

            # Wait until the model finishes training
            result = None
            for i in range(self.hpc_config.num_training_ranks):
                while True:
                    try:
                        result = self.queues.get_result(topic='training', timeout=1)
                        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['training-results'], flush=True)
                    except TimeoutException:
                        if self.done.is_set():
                            return
                        else:
                            continue
                    break
            model_dir = Path(self.out_dir / 'models')
            model_dir.mkdir(exist_ok=True)

            # Update the model if successful
            if result.success:
                self.model_iteration = attempt_id
                new_model_path = model_dir / f'model-v{self.model_iteration}.ckpt'
                shutil.copyfile(result.value, new_model_path)
                self.generator_config.generator_path = new_model_path
                result.task_info['model_updated'] = datetime.now().timestamp()
                self.logger.info(f'Received training result. Updated generator path to {new_model_path}, version number to {self.model_iteration}')
            else:
                self.logger.warning(f'Training failed: {result.failure_info.exception}')
            shutil.rmtree(train_dir)  # Clear training directory when done

    @task_submitter(task_type='cp2k')
    def submit_cp2k(self):
        """Start a CP2K submission"""

        # Query the database to find the best MOF we have not run CP2K on yet
        while True:  # Runs until something gets submitted
            try:
                record = self.dft_selector.select_next()
            except ValueError:
                self.logger.info('No MOFs ready for CP2K. Waiting for MD to finish')
                self.cp2k_ready.clear()

                while not self.cp2k_ready.wait(timeout=1.):
                    if self.done.is_set():
                        return
            else:
                # Add this to the list of things which have been run
                mofadb.mark_in_progress(self.collection, record, 'dft')
                self.queues.send_inputs(
                    record,
                    method='run_optimization',
                    topic='cp2k',
                    task_info={'mof': record.name}
                )
                self.logger.info(f'Submitted {record.name} to run with CP2K')

                # TODO (wardlt): Temporary hack: sleep to avoid overloading mpiexec
                from time import sleep
                sleep(0.1)  # Limits us to 100 tasks per second
                return

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
            result.task_info['cp2k_path'] = str(cp2k_path)
            self.queues.send_inputs(cp2k_path,
                                    method='compute_partial_charges',
                                    task_info=result.task_info,
                                    topic='cp2k')
            self.logger.info(f'Completed CP2K computation for {mof_name}. Runtime: {result.time.running:.2f} s. Started partial charge computation')
        elif result.method == 'compute_partial_charges':
            cp2k_path = result.task_info['cp2k_path']
            self.queues.send_inputs(
                mof_name, cp2k_path,
                method='run_gcmc',
                topic='cp2k',
                task_info=result.task_info
            )
            self.logger.info(f'Partial charges are complete for {mof_name}. Submitted RASPA')
        elif result.method == 'run_gcmc':
            # Store result
            uptake_mean, uptake_std, _, _ = result.value
            record = mofadb.get_records(self.collection, [mof_name])[0]
            record.gas_storage['CO2'] = uptake_mean
            record.times['raspa-done'] = datetime.now()
            mofadb.update_records(self.collection, [record])

            # Update and trigger training, in case it's blocked
            self.num_raspa_completed += 1
            if self.num_raspa_completed > self.trainer_config.curriculum.min_gas_counts:
                self.start_train.set()
            self.logger.info(f'Stored gas storage capacity for {mof_name}: {uptake_mean:.3e} +/- {uptake_std:.3e} mol/kg. Completed {self.num_raspa_completed}')
        else:
            raise ValueError(f'Method not supported: {result.method}')
        print(result.json(exclude={'inputs', 'value'}), file=self._output_files['simulation-results'], flush=True)
