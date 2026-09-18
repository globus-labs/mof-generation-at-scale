#!/usr/bin/env python
"""Check MOFA's environment without importing the separate LAMMPS environment."""
import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import shutil
import subprocess
import sys


def check_python_environment(root: Path):
    """Reject a wrong interpreter before checking the pinned package stack."""
    expected = Path(os.environ.get('MOFA_ENV') or root / 'mofa_env_py312').resolve()
    active = Path(sys.prefix).resolve()
    if active != expected or sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError(
            f'Expected MOFA environment {expected} with Python 3.12.13; '
            f'running {sys.executable} (prefix {active}, Python {sys.version.split()[0]}). '
            f'Source {root / "bin/activate-mofa-polaris.sh"} with MOFA_ENV={expected} '
            'before running this check. Polaris workflow jobs select mofa_env_py312 automatically.'
        )
    print(f'MOFA environment: {active}', flush=True)
    print(f'Python: {sys.executable} ({sys.version.split()[0]})', flush=True)


def check_mace_import_order(root: Path):
    """Exercise native-library loading without imports in this process masking it."""
    calculation = '''
from mofa.simulation.mace import load_model
from ase import Atoms
import numpy as np
atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.calc = load_model('cpu')
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
if not np.isfinite(energy) or not np.isfinite(forces).all():
    raise RuntimeError('MACE returned non-finite energy or forces')
print('MACE energy and forces: OK', flush=True)
'''
    for label, first_import in [
        ('Open Babel before MACE', 'from openbabel import pybel'),
        ('MOFA model before MACE', 'import mofa.model'),
    ]:
        print(f'Checking {label} in a fresh process', flush=True)
        subprocess.run(
            [sys.executable, '-X', 'faulthandler', '-c', first_import + '\n' + calculation],
            cwd=root, check=True, timeout=300,
        )
        print(f'{label}: OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Run a Torch CUDA operation (compute nodes only)')
    parser.add_argument('--external', action='store_true', help='Check external executable, environment, and data paths')
    parser.add_argument('--models', action='store_true', help='Load DiffLinker and check MACE import order, energy and forces on CPU')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    check_python_environment(root)
    from packaging.version import Version

    for package, expected in [('torch', '2.5.0+cu124'), ('numpy', '1.26.4'), ('redis', '5.3.1'),
                              ('colmena', '0.7.2'), ('mace-torch', '0.3.13'),
                              ('parsl', '2026.7.27'), ('openbabel-wheel', '3.1.1.23')]:
        installed = version(package)
        if Version(installed) != Version(expected):
            raise RuntimeError(f'{package}: expected {expected}, found {installed}')
        print(f'{package}: {installed}', flush=True)

    # Multiple bindings can overwrite each other's Python modules and libraries.
    if list((Path(sys.prefix) / 'conda-meta').glob('openbabel-*.json')):
        raise RuntimeError('Conda Open Babel overlaps openbabel-wheel; use a fresh environment')
    try:
        version('openbabel')
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError('Both openbabel and openbabel-wheel are installed; use a fresh environment')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)

    for module in ['openbabel.openbabel', 'ase', 'rdkit.Chem', 'pymatgen.core',
                   'pytorch_lightning', 'mace.calculators', 'colmena.task_server.parsl',
                   'proxystore.connectors.redis', 'mofa.hpc.config', 'mofa.generator']:
        import_module(module)
        print(f'Import {module}: OK', flush=True)
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    print('MPI:', MPI.Get_library_version().splitlines()[0], flush=True)
    for command in ['redis-server', 'redis-cli', 'mongod', 'chargemol', 'simulate', 'monitor_utilization']:
        resolved = shutil.which(command)
        if resolved is None:
            raise RuntimeError(f'Missing {command}; build a fresh environment with bash create_conda.sh --prefix PATH')
        print(f'{command}: {resolved}', flush=True)

    if args.external:
        lammps = Path(os.environ['LAMMPS_ROOT'])
        cp2k = Path(os.environ['CP2K_ROOT'])
        for executable in [lammps / 'build-mliap-no-mpi/lmp',
                           cp2k / 'exe/local_cuda/cp2k_shell.ssmp',
                           cp2k / 'exe/local_cuda/cp2k_shell.psmp',
                           root / 'bin/run-lammps-polaris.sh', root / 'bin/run-cp2k-polaris.sh']:
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise RuntimeError(f'Missing executable: {executable}')
        for path in [Path(os.environ['LAMMPS_VENV']) / 'bin/activate',
                     Path(os.environ['CP2K_DATA_DIR']) / 'BASIS_MOLOPT',
                     Path(os.environ['CP2K_DATA_DIR']) / 'GTH_POTENTIALS',
                     root / 'input-files/mace/mace-mp0_medium-mliap_lammps.pt']:
            if not path.is_file():
                raise RuntimeError(f'Missing runtime input: {path}')

    if args.models:
        # Also populates the foundation-model cache before compute jobs need it.
        check_mace_import_order(root)
        from mofa.utils.src.lightning import DDPM
        checkpoint = root / 'models/geom-300k/geom_difflinker_epoch=997_new.ckpt'
        DDPM.load_from_checkpoint(str(checkpoint), map_location='cpu').eval()
        print('DiffLinker checkpoint: OK', flush=True)

    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; run the GPU check inside a PBS compute job')
        value = torch.ones((16, 16), device='cuda')
        assert (value @ value).sum().item() == 4096
        torch.cuda.synchronize()
        print(f'Torch CUDA: OK ({torch.cuda.get_device_name()})', flush=True)
    print('MOFA preflight passed', flush=True)


if __name__ == '__main__':
    main()
