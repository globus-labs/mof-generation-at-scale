"""Ensure HPC configs work"""
import os
from pathlib import Path

from pytest import mark

from mofa.hpc.config import configs, PolarisConfig, AuroraConfig


@mark.parametrize('name', ['local', 'localXY'])
def test_local(tmpdir, name):
    config = configs[name]()
    assert config.torch_device == 'cuda'
    assert config.num_workers == 3
    parsl_cfg = config.make_parsl_config(Path(tmpdir))
    assert str(tmpdir) in parsl_cfg.run_dir


def test_polaris(tmpdir):
    hostfile_path = tmpdir / 'HOSTFILE'
    with open(hostfile_path, 'w') as fp:
        for i in range(5):
            print(f'host-{i}', file=fp)
    os.environ['PBS_NODEFILE'] = str(hostfile_path)

    try:
        config: PolarisConfig = configs['polaris']()
        config.dft_fraction = 0.5
        config.nodes_per_cp2k = 2
        config.ai_fraction = 0.5

        # Make sure the nodes are split appropriately
        hosts = config.hosts
        assert len(hosts) == 5
        assert len(config.ai_hosts) == 2
        assert len(config.cp2k_hosts) == 2
        assert len(config.lammps_hosts) == 1

        assert config.num_workers == 4 + 16 + 1
        parsl_cfg = config.make_parsl_config(Path(tmpdir))
        assert str(tmpdir) in parsl_cfg.run_dir

        # Make sure nodes are allocated appropriately
        assert config.number_inf_workers == 4
        assert config.num_lammps_workers == 16
        assert config.num_cp2k_workers == 1

        # Check the CPU affinity
        assert parsl_cfg.executors[0].cpu_affinity == 'list:24-31:16-23:8-15:0-7'
        assert parsl_cfg.executors[2].cpu_affinity.startswith('list:30-30:28-28')
        assert parsl_cfg.executors[-1].cpu_affinity.startswith('list:1:3:5:')

        # Make the cp2k call
        cmd = config.cp2k_cmd
        assert str(config.run_dir) in cmd

    finally:
        del os.environ['PBS_NODEFILE']


def test_aurora(tmpdir):
    hostfile_path = tmpdir / 'HOSTFILE'
    with open(hostfile_path, 'w') as fp:
        for i in range(20):
            print(f'host-{i}', file=fp)
    os.environ['PBS_NODEFILE'] = str(hostfile_path)

    try:
        config = AuroraConfig()
        config.ai_fraction = 0.1
        config.dft_fraction = 0.25
        config.make_parsl_config(Path(tmpdir))

        assert 'flare' in config.graspa_cmd[0]

        # Check that it has the correct GPU settings
        assert config.gpus_per_node == 12
        assert config.torch_device == 'xpu'

        assert len(config.hosts) == 20
        assert len(config.ai_hosts) == 2
    finally:
        del os.environ['PBS_NODEFILE']
