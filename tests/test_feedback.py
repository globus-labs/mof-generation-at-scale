from pathlib import Path

from pytest import fixture, mark
import torch

from mofa.feedback import collect_hp_linkers

def test_feedback():
    collect_hp_linkers()
