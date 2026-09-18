"""Exercise launcher boundaries without loading site modules or needing GPUs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mofa launchers ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = os.environ.copy()
        for name in ['CONDA_EXE', 'CP2K_BINARY', 'CP2K_DATA_DIR']:
            self.env.pop(name, None)
        module_stub = self.root / 'modules.sh'
        module_stub.write_text('module() { echo "module $*"; }\n')
        self.env.update(BASH_ENV=str(module_stub), CUDA_HOME='/mock/cuda',
                        LAMMPS_ROOT=str(self.root / 'lammps'),
                        LAMMPS_VENV=str(self.root / 'lammps/.venv'),
                        CP2K_ROOT=str(self.root / 'cp2k'),
                        CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='8',
                        PYTHONPATH='/parent/python')

    def executable(self, relative, body):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/bash\n' + body)
        path.chmod(0o755)

    def run_wrapper(self, name, *args):
        return subprocess.run(['bash', str(ROOT / 'bin' / name), *args],
                              env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)

    def test_lammps_activates_own_venv_and_preserves_arguments_and_gpu(self):
        activate = self.root / 'lammps/.venv/bin/activate'
        activate.parent.mkdir(parents=True)
        activate.write_text('export LAMMPS_ACTIVE=yes\n')
        self.executable('lammps/build-mliap-no-mpi/lmp',
                        'printf "%s\\n" "$LAMMPS_ACTIVE" "$CUDA_VISIBLE_DEVICES" '
                        '"${PYTHONPATH-unset}" "$@"\nexit 7\n')
        result = self.run_wrapper('run-lammps-polaris.sh', '-i', 'input with spaces.lmp')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout.splitlines(), ['yes', '2', 'unset', '-i', 'input with spaces.lmp'])

    def test_cp2k_modes_keep_stdout_clean_and_use_external_data(self):
        data = self.root / 'cp2k/data'
        data.mkdir(parents=True)
        for name in ['BASIS_MOLOPT', 'GTH_POTENTIALS']:
            (data / name).touch()
        for binary in ['cp2k_shell.ssmp', 'cp2k_shell.psmp', 'cp2k.psmp']:
            self.executable('cp2k/exe/local_cuda/' + binary,
                            'printf "%s\\n" "${0##*/}" "$CP2K_DATA_DIR" "$OMP_NUM_THREADS" "$@"\n')
            self.env['CP2K_BINARY'] = binary
            result = self.run_wrapper('run-cp2k-polaris.sh', '-i', 'input with spaces.inp')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             [binary, str(data), '8', '-i', 'input with spaces.inp'])

    def test_missing_external_installation_fails_before_loading_modules(self):
        for name in ['run-cp2k-polaris.sh', 'run-lammps-polaris.sh']:
            result = self.run_wrapper(name)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, '')
            self.assertNotIn('module reset', result.stderr)


if __name__ == '__main__':
    unittest.main()
