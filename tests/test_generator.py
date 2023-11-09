import pytest
from pytest import fixture
from mofa.generator import train_generator, run_generator
import numpy as np
import torch
from mofa.utils.src.lightning import DDPM
from mofa.utils.difflinker_sample_and_analyze import main_run

def test_cuda():
    assert torch.cuda.is_available()

@fixture()
def load_model():
    model = "mofa/models/geom_difflinker.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    return ddpm

def test_load_model(load_model):
    print(load_model.__class__.__name__)
    print("Successful?")

def test_training():
    ...

def test_sampling():
    ...

def test_fragmentation():
    ...
