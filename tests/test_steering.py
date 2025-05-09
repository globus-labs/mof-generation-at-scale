"""Test for the Colmena steering algorithm"""
from pathlib import Path
from time import sleep
import pickle as pkl
import warnings
import logging
import gzip

import numpy as np
from colmena.queue import ColmenaQueues, PipeQueues
from colmena.exceptions import TimeoutException
from colmena.models import Result
from proxystore.store import Store
from proxystore.connectors.file import FileConnector
from pytest import fixture, raises
from mongomock import MongoClient
from ase import io as aseio

from mofa.finetune.difflinker import DiffLinkerCurriculum
from mofa.selection.dft import DFTSelector
from mofa.selection.md import MDSelector
from mofa.simulation.lammps import LAMMPSRunner
from mofa.assembly.validate import process_ligands
from mofa.generator import run_generator
from mofa.model import LigandTemplate, NodeDescription, MOFRecord
from mofa.steering import MOFAThinker, GeneratorConfig, TrainingConfig, SimulationConfig
from mofa.hpc.config import LocalConfig
from mofa.db import initialize_database, create_records


def _pull_tasks(queues: ColmenaQueues) -> list[tuple[str, Result]]:
    """Pull all tasks available on the queue"""
    tasks = []
    while True:
        try:
            task = queues.get_task(timeout=1)
        except TimeoutException:
            break
        tasks.append(task)
    return tasks


@fixture()
def client_collection():
    client = MongoClient()
    coll = initialize_database(client)
    return client, coll


@fixture()
def queues(tmpdir):
    store_dir = Path(tmpdir) / 'store'
    store = Store(name='store', connector=FileConnector(store_dir=str(store_dir)), register=True)
    with store:
        yield PipeQueues(topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'], proxystore_name='store', proxystore_threshold=1000)


@fixture()
def hpc_config(tmpdir):
    run_dir = Path(tmpdir) / 'run'
    run_dir.mkdir()
    return LocalConfig(run_dir=run_dir)


@fixture()
def cache_dir(file_path):
    c = file_path / 'steering' / 'results'
    c.mkdir(parents=True, exist_ok=True)
    return c


@fixture()
def ligand_templates(file_path):
    temp_dir = file_path / '..' / '..' / 'input-files' / 'zn-paddle-pillar'
    ligands = []
    for path in temp_dir.glob('template_*_prompt.yml'):
        ligands.append(LigandTemplate.from_yaml(path))
    return ligands


@fixture()
def gen_config(file_path, ligand_templates):
    return GeneratorConfig(
        generator_path=file_path / "difflinker" / "geom_difflinker.ckpt",
        templates=ligand_templates,
        atom_counts=[10],
        min_ligand_candidates=32,
    )


@fixture()
def trn_config(coll):
    return TrainingConfig(
        num_epochs=1,
        curriculum=DiffLinkerCurriculum(
            max_size=4,
            max_strain=0.5,
            strain_method='uff',
            collection=coll
        )
    )


@fixture()
def node_template(file_path):
    node_path = file_path / 'assemble/nodes/zinc_paddle_pillar.xyz'
    return NodeDescription(
        xyz=Path(node_path).read_text(),
        smiles='[Zn][O]([Zn])([Zn])[Zn]',  # Is this right?
    )


@fixture()
def sim_config():
    return SimulationConfig(
        md_level='uff',
        md_length=10000
    )


@fixture()
def md_selector(coll):
    return MDSelector(
        md_level='uff',
        collection=coll,
        maximum_steps=5000
    )


@fixture()
def dft_selector(coll):
    return DFTSelector(
        md_level='uff',
        collection=coll
    )


@fixture()
def thinker(queues, coll, md_selector, dft_selector, hpc_config, gen_config, trn_config, sim_config, node_template, tmpdir):
    thinker = MOFAThinker(
        queues=queues,
        collection=coll,
        out_dir=hpc_config.run_dir,
        hpc_config=hpc_config,
        simulation_budget=8,
        generator_config=gen_config,
        trainer_config=trn_config,
        simulation_config=sim_config,
        md_selector=md_selector,
        dft_selector=dft_selector,
        node_template=node_template
    )

    # Route logs to disk
    handlers = [logging.FileHandler(hpc_config.run_dir / 'run.log')]
    for logger in [thinker.logger, logging.getLogger('colmena')]:
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    with thinker:
        thinker.start()
        yield thinker
        thinker.done.set()
        queues._all_complete.set()  # Act like all tasks have finished
        queues._active_tasks.clear()
    thinker.join()


def make_gen_outputs(task: Result, cache_dir: Path):
    cache_dir.mkdir(exist_ok=True)

    # Assemble the name of this task
    task.deserialize()
    templates: list[LigandTemplate] = task.kwargs['templates']
    assert len(templates) == 1
    result_name = f'{templates[0].anchor_type}-{task.kwargs["n_atoms"]}.pkl.gz'
    result_path = cache_dir.joinpath(result_name)

    # Get the result
    if result_path.is_file():
        with gzip.open(result_path, 'r') as fp:
            result = pkl.load(fp)
    else:
        result = list(run_generator(
            **task.kwargs,
            n_samples=32,
            n_steps=64,
            device='cpu'
        ))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = process_ligands(result)
        with gzip.open(result_path, "w") as fp:
            pkl.dump(result, fp)

    # Make a message with the model
    done_res = task.model_copy(deep=True)
    done_res.set_result(None, intermediate=False)
    done_res.serialize()

    task.method = 'process_ligands'
    task.set_result(result, intermediate=True)
    task.serialize()
    return done_res, task


def make_lammps_output(task: Result, cache_dir: Path):
    # Make the task name
    task.deserialize()
    record: MOFRecord = task.args[0]
    length: int = task.args[1]
    result_name = f'{record.name}_{length}.extxyz.gz'
    result_path = cache_dir.joinpath(result_name)

    if result_path.is_file():
        with gzip.open(result_path, 'rt') as fp:
            frames = aseio.read(fp, index=':', format='extxyz')
        frames = [(length // 10 * i, frame) for i, frame in enumerate(frames)]
    else:
        lmp = LAMMPSRunner(
            lammps_command=["lmp_serial"],
            lmp_sims_root_path=cache_dir / "lmp_sims",
            lammps_environ={'OMP_NUM_THREADS': '1'},
            delete_finished=False,
        )
        frames = lmp.run_molecular_dynamics(mof=record, timesteps=length, report_frequency=length // 10)
        with gzip.open(result_path, 'wt') as fp:
            aseio.write(fp, [f for _, f in frames], format='extxyz')

    # Make a message with the model
    done_res = task.model_copy(deep=True)
    done_res.set_result(frames)
    done_res.serialize()

    return done_res


def make_cp2k_outputs(task: Result):
    task.deserialize()

    # Get the results
    if task.method == 'run_optimization':
        result = ('a', '/here/')
    elif task.method == 'compute_partial_charges':
        result = ('a',)
    elif task.method == 'run_gcmc':
        result = (1, 2, 0.1, 0.2)
    else:
        raise ValueError()

    # Make a message with the output
    done_res = task.model_copy(deep=True)
    done_res.set_result(result)
    done_res.serialize()

    return done_res


def test_generator(thinker, queues, cache_dir):
    """Ensure generator tasks are properly circulated"""
    assert (thinker.out_dir / 'simulation-results.json').exists()  # Created on startup
    tasks = _pull_tasks(queues)
    assert queues.active_count == 1
    assert len(tasks) == 1
    topic, task = tasks[0]
    assert topic == 'generation'

    # Get the generator output and feed back to the thinker
    done_result, task = make_gen_outputs(task, cache_dir / 'generate')
    queues.send_result(task)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 0  # The ligands won't create new tasks

    queues.send_result(done_result)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 1  # Sending a completed task will trigger new updates


def test_simulation_pipeline(thinker, queues, cache_dir, example_record):
    """Step through the entire simulation pipeline"""
    # Pull the generate task out of the queues (it is there on startup and irrelevant here)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 1

    # Ensure there are no MOFs ready for running
    assert thinker.md_selector.count_available() == 0

    # Insert a MOF record into queue
    thinker.stability_queue.append(example_record)
    thinker.mofs_available.set()
    sleep(0.5)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 1
    assert thinker.collection.count_documents({'in_progress': 'stability'}) == 1

    # Run initial relaxation
    _, task = tasks[0]
    assert task.method == 'run_optimization_ff'
    task.deserialize()
    task.set_result((task.args[0].atoms, None))
    task.serialize()
    queues.send_result(task)

    # Check that it has updated
    sleep(1.5)
    assert thinker.collection.count_documents({'times.relaxed': {'$exists': True}}) == 1
    assert thinker.collection.count_documents({'in_progress': 'stability'}) == 1

    # We should have two tasks now: one CP2K and another LAMMPS
    tasks = _pull_tasks(queues)
    assert len(tasks) == 2

    tasks_by_method = dict(
        (task.method, task) for _, task in tasks
    )
    assert 'run_molecular_dynamics' in tasks_by_method

    # Do the MD steps
    done_result = make_lammps_output(tasks_by_method['run_molecular_dynamics'], cache_dir / 'lammps')
    assert done_result.complete
    queues.send_result(done_result)
    sleep(0.5)

    # Check that the database has the content
    assert thinker.collection.count_documents({'in_progress': 'stability'}) == 0
    assert thinker.collection.count_documents({}) == 1
    assert thinker.cp2k_ready.is_set()

    # Check that the CP2K is getting started
    assert 'run_optimization' in tasks_by_method

    # Check that no other compounds are eligible
    assert thinker.collection.count_documents({'in_progress': 'dft'}) == 1, \
        thinker.collection.find_one({})
    with raises(ValueError, match='criteria'):
        thinker.dft_selector.select_next()

    # "Run" the optimization
    task = tasks_by_method['run_optimization']
    result = make_cp2k_outputs(task)
    queues.send_result(result)

    tasks = _pull_tasks(queues)
    assert len(tasks) == 1
    _, task = tasks[0]
    assert task.method == 'compute_partial_charges'

    # "Run" the partial charges
    result = make_cp2k_outputs(task)
    queues.send_result(result)

    tasks = _pull_tasks(queues)
    assert len(tasks) == 1
    _, task = tasks[0]
    assert task.method == 'run_gcmc'

    # "Run" the GCMC
    result = make_cp2k_outputs(task)
    queues.send_result(result)

    tasks = _pull_tasks(queues)
    assert len(tasks) == 0

    assert thinker.collection.count_documents({'gas_storage.CO2': {'$exists': True}}) == 1


def test_retrain(thinker, queues, coll, example_record):
    """Make sure retraining can be triggered properly"""
    # Pull the generate task out of the queues (it is there on startup and irrelevant here)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 1

    # Make a series of records with increasingly-larger strains
    for strain in np.random.uniform(0., 0.2, 32):
        example_record.structure_stability['uff'] = strain
        create_records(coll, [example_record])
    assert len(thinker.trainer_config.curriculum.get_training_set()) > 0

    # Call retraining
    thinker.start_train.set()
    sleep(0.5)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 1
    _, task = tasks[0]
    assert task.method == 'train_generator'
