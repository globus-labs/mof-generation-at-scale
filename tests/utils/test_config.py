"""Ensure HPC configs work"""
import os
from pathlib import Path

from mofa.hpc.config import configs, PolarisConfig


def test_local(tmpdir):
    config = configs['local']()
    assert config.torch_device == 'cuda'
    assert config.num_workers == 3
    parsl_cfg = config.make_parsl_config(Path(tmpdir))
    assert str(tmpdir) in parsl_cfg.run_dir


def test_polaris(tmpdir):
    hostfile_path = tmpdir / 'HOSTFILE'
    with open(hostfile_path, 'w') as fp:
        for i in range(4):
            print(f'host-{i}', file=fp)
    os.environ['PBS_NODEFILE'] = str(hostfile_path)

    try:
        config: PolarisConfig = configs['polaris']()
        config.dft_fraction = 0.5
        config.nodes_per_cp2k = 2
        config.ai_fraction = 0.1

        # Make sure the nodes are split appropriately
        hosts = config.hosts
        assert len(hosts) == 4
        assert len(config.ai_hosts) == 1
        assert len(config.cp2k_hosts) == 2
        assert len(config.lammps_hosts) == 1

        assert config.num_workers == 4 + 4 + 1
        parsl_cfg = config.make_parsl_config(Path(tmpdir))
        assert str(tmpdir) in parsl_cfg.run_dir

        # Make sure nodes are allocated appropriately
        assert config.num_ai_workers == 4
        assert config.num_lammps_workers == 4
        assert config.num_cp2k_workers == 1

        # Check the CPU affinity
        assert parsl_cfg.executors[0].cpu_affinity == 'list:24-31:16-23:8-15:0-7'
        assert parsl_cfg.executors[1].cpu_affinity == 'list:24-30:16-22:8-14:0-6'
        assert parsl_cfg.executors[-1].cpu_affinity == 'list:7:15:23:31'

        # Make the cp2k call
        cmd = config.cp2k_cmd
        assert str(config.run_dir) in cmd

    finally:
        del os.environ['PBS_NODEFILE']
