"""Test for the Colmena steering algorithm"""
from pathlib import Path
import logging

from colmena.exceptions import TimeoutException
from pytest import fixture
from mongomock import MongoClient
from colmena.queue import PipeQueues

from mofa.model import LigandTemplate, NodeDescription
from mofa.steering import MOFAThinker, GeneratorConfig, TrainingConfig
from mofa.hpc.config import LocalConfig


@fixture()
def queues():
    return PipeQueues(topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'])


@fixture()
def hpc_config():
    return LocalConfig()


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


def test_start(thinker, queues):
    assert (thinker.out_dir / 'simulation-results.json').exists()
    tasks = []
    while True:
        try:
            task = queues.get_task(timeout=0.5)
        except TimeoutException:
            break
        tasks.append(task)
    assert queues.active_count == 1
    assert len(tasks) == 1
    topic, task = tasks[0]
    assert topic == 'generation'
