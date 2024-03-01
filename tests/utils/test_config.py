"""Ensure HPC configs work"""
import os
from pathlib import Path

from mofa.hpc.config import configs


def test_local(tmpdir):
    config = configs['local']()
    assert config.torch_device == 'cuda'
    assert config.num_workers == 2
    parsl_cfg = config.make_parsl_config(Path(tmpdir))
    assert str(tmpdir) in parsl_cfg.run_dir


def test_polaris(tmpdir):
    hostfile_path = tmpdir / 'HOSTFILE'
    with open(hostfile_path, 'w') as fp:
        print('localhost', file=fp)
    os.environ['PBS_NODEFILE'] = str(hostfile_path)

    try:
        config = configs['polaris']()
        assert config.num_workers == 4
        parsl_cfg = config.make_parsl_config(Path(tmpdir))
        assert str(tmpdir) in parsl_cfg.run_dir
        assert parsl_cfg.executors[0].provider.nodes_per_block == 1
    finally:
        del os.environ['PBS_NODEFILE']
