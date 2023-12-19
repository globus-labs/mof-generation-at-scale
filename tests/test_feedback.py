from pathlib import Path

from pytest import fixture, mark
import torch
import os

from mofa.feedback import collect_hp_linkers, ROOT_DIR

@fixture
def node_name():
    return "TEMPNODE"

@mark.paramaterize(["input_path", "output_path"], [Path(os.path.join(ROOT_DIR, "lammps/pdb/ligands")), ])
def test_feedback(input_path, output_path, node_name):
    collect_hp_linkers()
