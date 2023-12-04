import pytest
from pytest import fixture, mark
import numpy as np
import torch
from mofa.fragment import fragment_mof_linkers
from mofa.generator import train_generator, run_generator
import os

@mark.parametrize('nodes', [['CuCu'], ['CuCu', 'ZnZn']])
def test_fragmentation(nodes):
    # fragmentation(nodes)
    # process_fragments(nodes)
    fragment_mof_linkers(nodes)

