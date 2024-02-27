"""An example of the workflow which runs all aspects of MOF generation in parallel"""
import json
import logging
import hashlib
import sys
from functools import partial, update_wrapper
from platform import node
from csv import DictWriter
from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict
from itertools import product
from datetime import datetime
from collections import deque
from random import shuffle
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from openbabel import openbabel as ob
from parsl import Config, HighThroughputExecutor
from colmena.models import Result
from colmena.task_server import ParslTaskServer
from colmena.queue import ColmenaQueues, PipeQueues
from colmena.thinker import BaseThinker, agent, result_processor, task_submitter, ResourceCounter

from mofa.assembly.assemble import assemble_mof
from mofa.generator import run_generator
from mofa.model import MOFRecord, NodeDescription, LigandTemplate, LigandDescription
from mofa.scoring.geometry import MinimumDistance, LatticeParameterChange
from mofa.simulation.lammps import LAMMPSRunner
from mofa.utils.conversions import write_to_string
from mofa.utils.xyz import xyz_to_mol

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


class MOFAThinker(BaseThinker):
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
                 num_nodes: int,
                 generator_config: GeneratorConfig):
        super().__init__(queues, ResourceCounter(num_nodes, task_types=['generation', 'simulation']))
        self.generator_config = generator_config
        self.out_dir = out_dir

        # Set up the queues
        self.mof_queue = deque(maxlen=200)  # Starts empty

        self.generate_queue = deque()  # Starts with one of each task (ligand, size)
        tasks = list(product(range(len(generator_config.templates)), generator_config.atom_counts))
        shuffle(tasks)
        self.generate_queue.extend(tasks)

        self.ligand_queue = defaultdict(lambda: deque(maxlen=200))  # Starts empty

        # Set aside one node for generation
        self.rec.reallocate(None, 'generation', 1)

    @task_submitter(task_type='generation')
    def submit_generation(self):
        """Submit MOF generation tasks when resources are available"""

        ligand_id, size = self.generate_queue.popleft()
        ligand = self.generator_config.templates[ligand_id]
        self.queues.send_inputs(
            input_kwargs={'templates': [ligand], 'n_atoms': size},
            topic='generation',
            method='run_generator',
            task_info={'task': (ligand_id, size)}
        )
        self.logger.info(f'Requested more samples of type={ligand.anchor_type} size={size}')

    @result_processor(topic='generation')
    def store_generation(self, result: Result):
        """Receive generated ligands, append to the generation queue """

        # Start a new task
        ligand_id, size = result.task_info['task']
        self.generate_queue.append((ligand_id, size))  # Push this generation task back on the queue
        self.rec.release('generation')

        # Retrieve the results
        if not result.success:
            self.logger.warning(f'Generation task failed: {result.failure_info.exception}\nStack: {result.failure_info.traceback}')
            self.done.set()
            return
        new_ligands: list[LigandDescription] = result.value
        anchor_type = self.generator_config.templates[ligand_id].anchor_type
        self.logger.info(f'Received {len(new_ligands)} new ligands of anchor_type={anchor_type} size={size}')

        # Check if they are valid
        #  TODO (wardlt): Make this parallel
        all_records = []
        valid_count = 0
        for ligand in new_ligands:
            # Store the ligand information for debugging purposes
            record = {"anchor_type": ligand.anchor_type,
                      "smiles": None,
                      "xyz": ligand.xyz,
                      "anchor_atoms": ligand.anchor_atoms,
                      "valid": False}
            all_records.append(record)

            # Try constrained optimization on the ligand
            try:
                ligand.anchor_constrained_optimization()
            except (ValueError, AttributeError,):
                continue

            # Parse each new ligand, determine whether it is a single molecule
            try:
                mol = xyz_to_mol(ligand.xyz)
            except (ValueError,):
                continue

            # Store the smiles string
            Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol)
            record['smiles'] = smiles

            if len(Chem.GetMolFrags(mol)) > 1:
                continue

            # If passes, save the SMILES string and store the molecules
            ligand.smiles = Chem.MolToSmiles(mol)

            # Update the record, add to ligand queue and prepare it for writing to disk
            record['valid'] = True
            valid_count += 1
            self.ligand_queue[anchor_type].append(ligand)
        self.logger.info(f'{valid_count} of {len(new_ligands)} are valid. ({valid_count / len(new_ligands) * 100:.1f}%)')

        # Write record of generation tasks to disk
        record_file = self.out_dir / 'all_ligands.csv'
        first_write = not record_file.is_file()

        with record_file.open('a') as fp:
            writer = DictWriter(fp, all_records[0].keys())
            if first_write:
                writer.writeheader()
            writer.writerows(all_records)

        if valid_count > 0:
            self.logger.info('We have at least one valid linker. Good enough for now')
            self.done.set()


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='MOF Settings', description='Options related to the MOF type being generated')
    group.add_argument('--node-path', required=True, help='Path to a node record')

    group = parser.add_argument_group(title='Generator Settings', description='Options related to how the generation is performed')
    group.add_argument('--ligand-templates', required=True, nargs='+',
                       help='Path to YAML files containing a description of the ligands to be created')
    group.add_argument('--generator-path', required=True,
                       help='Path to the PyTorch files describing model architecture and weights')
    group.add_argument('--molecule-sizes', nargs='+', type=int, default=(10, 11, 12), help='Sizes of molecules we should generate')
    group.add_argument('--num-samples', type=int, default=16, help='Number of molecules to generate at each size')

    group = parser.add_argument_group(title='Assembly Settings', description='Options related to MOF assembly')
    group.add_argument('--num-to-assemble', default=4, type=int, help='Number of MOFs to create from generated ligands')
    group.add_argument('--max-assemble-attempts', default=100,
                       help='Maximum number of attempts to create a MOF')

    group = parser.add_argument_group(title='Simulation Settings Settings', description='Options related to MOF assembly')
    group.add_argument('--md-timesteps', default=100000, help='Number of timesteps for the UFF MD simulation', type=int)
    group.add_argument('--md-snapshots', default=100, help='Maximum number of snapshots during MD simulation', type=int)

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--torch-device', default='cpu', help='Device on which to run torch operations')

    args = parser.parse_args()

    # Load the example MOF
    # TODO (wardlt): Use Pydantic for JSON I/O
    node_record = NodeDescription(**json.loads(Path(args.node_path).read_text()))

    # Make the run directory
    run_params = args.__dict__.copy()
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    run_dir = Path('run') / f'parallel-{node()}-{start_time.strftime("%d%b%y%H%M%S")}-{params_hash}'
    run_dir.mkdir(parents=True)

    # Load the ligand descriptions
    templates = []
    for path in args.ligand_templates:
        template = LigandTemplate.from_yaml(path)
        templates.append(template)

    # Make the generator settings and the function
    generator = GeneratorConfig(
        generator_path=args.generator_path,
        atom_counts=args.molecule_sizes,
        templates=templates
    )

    gen_func = partial(run_generator, model=generator.generator_path, n_samples=args.num_samples, device=args.torch_device)
    update_wrapper(gen_func, run_generator)

    # Make the thinker
    queues = PipeQueues(topics=['generation'])
    thinker = MOFAThinker(queues, num_nodes=1, generator_config=generator, out_dir=run_dir)

    # Turn on logging
    my_logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'run.log')]
    for logger in [my_logger, thinker.logger]:
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    my_logger.info(f'Running job in {run_dir}')

    # Save the run parameters to disk
    (run_dir / 'params.json').write_text(json.dumps(run_params))

    # Launch the thinker and task server
    config = Config(
        executors=[HighThroughputExecutor(max_workers=1)],
        run_dir=str(run_dir / 'runinfo')
    )
    doer = ParslTaskServer(
        methods=[gen_func],
        queues=queues,
        config=config
    )

    try:
        doer.start()
        my_logger.info(f'Running parsl. pid={doer.pid}')

        thinker.run()
    finally:
        queues.send_kill_signal()

    exit()
