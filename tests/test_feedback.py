from pathlib import Path

from pytest import fixture, mark
import torch

from mofa.feedback import collect_hp_linkers

@fixture


def test_feedback(input_path, output_path, node_name):
    collect_hp_linkers()
