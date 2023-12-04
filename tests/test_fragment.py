import pytest
from pytest import fixture, mark
import numpy as np
import torch
from mofa.fragment import fragment_mof_linkers
from mofa.generator import train_generator, run_generator
import os

@mark.parametrize('n_atoms', [3])
@mark.parametrize('node', ['CuCu', 'ZnZn', 'ZnOZnZnZn'])
@mark.parametrize('n_samples', [1, 3])
def test_sampling_num_atoms(n_atoms, node, n_samples):
    run_generator(n_atoms=n_atoms, node=node, n_samples=n_samples)

@mark.parametrize('nodes', [['CuCu'], ['CuCu', 'ZnZn']])
def test_fragmentation(nodes):
    # fragmentation(nodes)
    # process_fragments(nodes)
    fragment_mof_linkers(nodes)

