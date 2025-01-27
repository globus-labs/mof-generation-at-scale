"""Test for the Colmena steering algorithm"""
from pathlib import Path
import pickle as pkl
import warnings
import logging
import gzip

from colmena.exceptions import TimeoutException
from colmena.models import Result
from pytest import fixture
from mongomock import MongoClient
from colmena.queue import PipeQueues

from mofa.assembly.validate import process_ligands
from mofa.generator import run_generator
from mofa.model import LigandTemplate, NodeDescription
from mofa.steering import MOFAThinker, GeneratorConfig, TrainingConfig
from mofa.hpc.config import LocalConfig


def _pull_tasks(queues) -> list[Result]:
    """Pull all tasks available on the queue"""
    tasks = []
    while True:
        try:
            task = queues.get_task(timeout=0.5)
        except TimeoutException:
            break
        tasks.append(task)
    return tasks


@fixture()
def queues():
    return PipeQueues(topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'])


@fixture()
def hpc_config():
    return LocalConfig()


@fixture()
def cache_dir(file_path):
    return file_path / 'steering' / 'results'


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
def trn_config():
    return TrainingConfig(
        num_epochs=1,
        minimum_train_size=4,
        maximum_train_size=10,
        best_fraction=0.5,
        maximum_strain=0.5
    )


@fixture()
def node_template(file_path):
    node_path = file_path / 'assemble/nodes/zinc_paddle_pillar.xyz'
    return NodeDescription(
        xyz=Path(node_path).read_text(),
        smiles='[Zn][O]([Zn])([Zn])[Zn]',  # Is this right?
    )


@fixture()
def thinker(queues, hpc_config, gen_config, trn_config, node_template, tmpdir):
    run_dir = Path(tmpdir) / 'run'
    run_dir.mkdir()
    thinker = MOFAThinker(
        queues=queues,
        mongo_client=MongoClient(),
        out_dir=run_dir,
        hpc_config=hpc_config,
        simulation_budget=8,
        generator_config=gen_config,
        trainer_config=trn_config,
        node_template=node_template,
    )

    # Route logs to disk
    handlers = [logging.FileHandler(run_dir / 'run.log')]
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
    cache_dir.mkdir(parents=True, exist_ok=True)

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

    task.set_result(result, intermediate=True)
    task.serialize()
    return done_res, task


def test_generator(thinker, queues, cache_dir):
    """Ensure generator tasks are properly circulated"""
    assert (thinker.out_dir / 'simulation-results.json').exists()
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

    assert thinker.out_dir.joinpath('all_ligands.csv').exists()

    queues.send_result(done_result)
    tasks = _pull_tasks(queues)
    assert len(tasks) == 1  # Sending a completed task will trigger new updates
