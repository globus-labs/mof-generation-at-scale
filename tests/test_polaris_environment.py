"""Check environment selection before a Polaris job can launch any work."""
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CHECK_ENVIRONMENT = runpy.run_path(str(ROOT / 'bin/check-polaris.py'))['check_python_environment']


class PolarisEnvironmentTests(unittest.TestCase):
    def test_shared_paths_default_to_py312_when_unset_or_empty(self):
        for value in [None, '']:
            with self.subTest(value=value):
                env = os.environ.copy()
                env.pop('BASH_ENV', None)
                env.pop('MOFA_ENV', None)
                if value is not None:
                    env['MOFA_ENV'] = value
                result = subprocess.run(
                    ['bash', '-eu', '-c', 'source "$1"; printf "%s" "$MOFA_ENV"',
                     'check-paths', str(ROOT / 'bin/polaris-paths.sh')],
                    env=env, capture_output=True, text=True, check=True,
                )
                self.assertEqual(result.stdout, str(ROOT / 'mofa_env_py312'))

    def test_jobs_override_an_inherited_old_environment_before_activation(self):
        with tempfile.TemporaryDirectory(prefix='mofa environment ') as tmp:
            repo = Path(tmp)
            (repo / 'bin').mkdir()
            # Stop at activation so this test cannot launch services or compute work.
            (repo / 'bin/activate-mofa-polaris.sh').write_text(
                'printf "SELECTED=%s\\n" "$MOFA_ENV"\nreturn 42\n'
            )
            env = os.environ.copy()
            env.pop('BASH_ENV', None)
            env.update(PBS_O_WORKDIR=tmp, MOFA_ENV='/old/mofa_env')
            for name in ['run-polaris-repo.sh', 'run-polaris-repo-test.sh',
                         'run-polaris-local-smoke.sh']:
                with self.subTest(script=name):
                    shutil.copyfile(ROOT / name, repo / name)
                    result = subprocess.run(
                        ['bash', str(repo / name)], env=env,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 42, result.stderr)
                    self.assertIn(f'SELECTED={repo / "mofa_env_py312"}', result.stdout)

    def test_missing_environment_stops_before_module_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop('BASH_ENV', None)
            env['MOFA_ENV'] = str(Path(tmp) / 'missing')
            result = subprocess.run(
                ['bash', '-eu', '-c',
                 'module() { echo UNEXPECTED_MODULE_CALL; }; source "$1"',
                 'check-activation', str(ROOT / 'bin/activate-mofa-polaris.sh')],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn('Missing MOFA Python:', result.stderr)
            self.assertNotIn('UNEXPECTED_MODULE_CALL', result.stdout)

    def test_preflight_rejects_wrong_environment_even_with_correct_python(self):
        with patch.dict(os.environ, {'MOFA_ENV': ''}), \
                patch('sys.prefix', str(ROOT / 'mofa_env')), \
                patch('sys.version_info', (3, 12, 13)):
            with self.assertRaisesRegex(RuntimeError, 'Expected MOFA environment .*mofa_env_py312'):
                CHECK_ENVIRONMENT(ROOT)

    def test_preflight_rejects_wrong_python_in_selected_environment(self):
        with patch.dict(os.environ, {'MOFA_ENV': str(ROOT / 'mofa_env_py312')}), \
                patch('sys.prefix', str(ROOT / 'mofa_env_py312')), \
                patch('sys.version_info', (3, 10, 21)), \
                patch('sys.version', '3.10.21'):
            with self.assertRaisesRegex(RuntimeError, 'Python 3.10.21'):
                CHECK_ENVIRONMENT(ROOT)

    def test_preflight_accepts_selected_py312_environment_and_build_overrides(self):
        for name in ['mofa_env_py312', 'candidate_environment']:
            with self.subTest(environment=name), \
                    patch.dict(os.environ, {'MOFA_ENV': str(ROOT / name)}), \
                    patch('sys.prefix', str(ROOT / name)), \
                    patch('sys.version_info', (3, 12, 13)), \
                    redirect_stdout(StringIO()) as output:
                CHECK_ENVIRONMENT(ROOT)
                self.assertIn(f'MOFA environment: {ROOT / name}', output.getvalue())


if __name__ == '__main__':
    unittest.main()
