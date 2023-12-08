import pytest
from pytest import fixture, mark
from mofa.generator import train_generator, run_generator
import numpy as np
import torch
from mofa.utils.src.lightning import DDPM
from mofa.utils.src.linker_size_lightning import SizeClassifier
from mofa.utils.difflinker_sample_and_analyze import main_run
from mofa.fragment import fragment_mof_linkers
import os

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

@mark.slow
def test_training():
    train_generator()

# https://docs.pytest.org/en/7.1.x/how-to/parametrize.html
@mark.parametrize('n_atoms', [3, 4])
def test_sampling_num_atoms(n_atoms):
    run_generator(n_atoms=n_atoms)

@mark.parametrize('n_atoms', [3])
@mark.parametrize('node', ['CuCu', 'ZnZn', 'ZnOZnZnZn'])
@mark.parametrize('n_samples', [1, 3])
def test_sampling_num_atoms(n_atoms, node, n_samples):
    run_generator(n_atoms=n_atoms, node=node, input_path=f"mofa/data/fragments_all/{node}/hMOF_frag_frag.sdf", n_samples=n_samples)

