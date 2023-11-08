import pytest
from pytest import fixture
from mofa.generator import train_generator, run_generator
import numpy as np
import torch
from 

@fixture()
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    return ddpm

def test_training():
    ...

def test_sampling():
    ...

def test_fragmentation():
    ...
