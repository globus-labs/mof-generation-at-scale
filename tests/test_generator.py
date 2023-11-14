import pytest
from pytest import fixture
from mofa.generator import train_generator, run_generator
import numpy as np
import torch
from mofa.utils.src.lightning import DDPM
from mofa.utils.src.linkzer_size_lightning import SizeClassifier
from mofa.utils.difflinker_sample_and_analyze import main_run

def test_cuda():
    assert torch.cuda.is_available()

@fixture()
def load_denoising_model():
    model = "mofa/models/geom_difflinker.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    return ddpm

@fixture()
def load_size_gnn_model():
    model = "mofa/models/geom_size_gnn.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizegnn = SizeClassifier.load_from_checkpoint(model, map_location=device).eval().to(device)
    return sizegnn

def test_load_model(load_denoising_model, load_size_gnn_model):
    print(load_denoising_model.__class__.__name__)
    print(load_size_gnn_model.__class__.__name__)
    print("Successful?")

def test_training():
    ...

def test_sampling():
    ...

def test_fragmentation():
    ...
