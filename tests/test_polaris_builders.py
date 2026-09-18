"""Exercise external build boundaries without site modules, installs, or GPUs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    'cp2k': '34658f03b9867a3335ace1b4af7b7736c33523bb',
    'lammps': 'd51bbd4983a26e2da6f1550e1b41592690b02a90',
}


class ExternalBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mofa-builders-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'mofa'
        for directory in ['bin', 'polaris-build']:
            shutil.copytree(ROOT / directory, self.repo / directory)
        self.log = self.root / 'calls.log'
        self.log.touch()
        self.env = os.environ.copy()
        for name in ['CONDA_EXE', 'LAMMPS_VENV', 'CP2K_CLEAN_TOOLCHAIN',
                     'CP2K_BUILD_JOBS', 'LAMMPS_BUILD_JOBS', 'MOFA_CONDA_MODULE',
                     'CP2K_BINARY', 'PYTHONHOME', 'PYTHONPATH']:
            self.env.pop(name, None)
        self.env.update(PBS_JOBID='123.test', PBS_O_WORKDIR=str(self.repo),
                        TEST_ROOT=str(self.root), TEST_LOG=str(self.log),
                        CUDA_HOME=str(self.root / 'cuda-13.0.1'),
                        CRAY_ACCEL_TARGET='nvidia80')
        for name, revision in REVISIONS.items():
            source = self.root / name
            source.mkdir()
            (source / 'revision').write_text(revision)
            (source / 'CMakeLists.txt').touch()
            self.env[name.upper() + '_ROOT'] = str(source)
        (self.root / 'lammps/cmake').mkdir()
        (self.root / 'lammps/cmake/CMakeLists.txt').touch()
        data = self.root / 'cp2k/data'
        data.mkdir()
        for name in ['BASIS_MOLOPT', 'GTH_POTENTIALS']:
            (data / name).touch()
        self.env['CP2K_DATA_DIR'] = str(data)
        (self.root / 'libsci/lib').mkdir(parents=True)
        (self.root / 'libsci/lib/libsci.so').touch()
        self.stub('python', '''
record "python $*"
if [[ "$1 $2" == '-m venv' ]]; then
    mkdir -p "$3/bin"
    cp "$0" "$3/bin/python"
    printf 'export VIRTUAL_ENV=%q\\nexport PATH=%q/bin:$PATH\\n' "$3" "$3" > "$3/bin/activate"
fi
''')
        self.stub('lmp', 'echo "ML-IAP ML-SNAP KOKKOS PYTHON EXTRA-MOLECULE"\n')
        self.stub('cp2k-exe', 'echo "CP2K 2025.2"\n')
        toolchain = self.root / 'cp2k/tools/toolchain'
        toolchain.mkdir(parents=True)
        (self.root / 'cp2k/arch').mkdir()
        self.stub('cp2k/tools/toolchain/install_cp2k_toolchain.sh', '''
record "toolchain $*"
[[ "${TEST_TOOLCHAIN_FAIL:-0}" == 0 ]] || exit 42
mkdir -p install/arch
printf '# mock arch\\n' > install/arch/local_cuda.psmp
printf 'export MOCK_CP2K_SETUP=1\\n' > install/setup
''')
        bootstrap = self.root / 'stubs.sh'
        bootstrap.write_text('''
record() { printf '%s\\n' "$*" >> "$TEST_LOG"; }
module() {
    record "module $*"
    if [[ "$*" == '-t list' ]]; then echo 'PrgEnv-gnu/8'; fi
    return 0
}
git() { cat "$2/revision"; }
conda() {
    record "conda $*"
    while [[ "$1" != --prefix ]]; do shift; done
    mkdir -p "$2/bin"
    cp "$TEST_ROOT/python" "$2/bin/python"
}
nvcc() { if [[ "$1" == --version ]]; then echo 'release 13.0'; fi; return 0; }
cc() {
    case "$1" in
        --version) echo 'GCC 14' ;;
        --cray-print-opts=libs) echo "-L$TEST_ROOT/libsci/lib -lmpi_gtl_cuda" ;;
    esac
    return 0
}
CC() { cc "$@"; }
ftn() { cc "$@"; }
cmake() {
    record "cmake $*"
    [[ "${TEST_CMAKE_FAIL:-0}" == 0 ]] || return 41
    mkdir -p "$LAMMPS_ROOT/build-mliap-no-mpi"
    cp "$TEST_ROOT/lmp" "$LAMMPS_ROOT/build-mliap-no-mpi/lmp"
}
make() {
    record "make $*"
    for arch in local local_cuda; do
        mkdir -p "exe/$arch"
        for name in cp2k.ssmp cp2k.psmp cp2k_shell.ssmp cp2k_shell.psmp; do
            cp "$TEST_ROOT/cp2k-exe" "exe/$arch/$name"
        done
    done
}
readelf() { echo "$CUDA_HOME/lib64"; }
ldd() {
    echo "libnvrtc.so => $CUDA_HOME/lib64/libnvrtc.so (0x1)"
    echo "libmpi_gtl_cuda.so => /mock/libmpi_gtl_cuda.so (0x2)"
    echo "libsci.so => $TEST_ROOT/libsci/lib/libsci.so (0x3)"
}
''')
        self.env['BASH_ENV'] = str(bootstrap)

    def stub(self, name, body):
        path = self.root / name
        path.write_text('#!/bin/bash\nset -e\n' + body)
        path.chmod(0o755)

    def run_build(self, name, *args):
        return subprocess.run(
            ['bash', str(self.repo / f'polaris-build/build-{name}-polaris.sh'), *args],
            cwd=self.repo, env=self.env, text=True, capture_output=True, timeout=30,
        )

    def run_wrapper(self, name, *args):
        return subprocess.run(
            ['bash', str(self.repo / f'bin/run-{name}-polaris.sh'), *args],
            cwd=self.repo, env=self.env, text=True, capture_output=True, timeout=30,
        )

    def test_help_needs_no_allocation_or_modules(self):
        self.env.pop('PBS_JOBID')
        for name in REVISIONS:
            with self.subTest(name=name):
                result = self.run_build(name, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(name.upper() + '_ROOT', result.stdout)
        self.assertEqual(self.log.read_text(), '')

    def test_login_node_builds_stop_before_modules(self):
        self.env.pop('PBS_JOBID')
        for name in REVISIONS:
            with self.subTest(name=name):
                self.assertEqual(self.run_build(name).returncode, 2)
        self.assertEqual(self.log.read_text(), '')

    def test_missing_and_repository_local_roots_are_rejected(self):
        alias = self.root / 'repo-alias'
        alias.symlink_to(self.repo, target_is_directory=True)
        for name in REVISIONS:
            for value in ['', 'deps/' + name, str(self.repo), str(alias)]:
                with self.subTest(name=name, value=value):
                    self.env[name.upper() + '_ROOT'] = value
                    self.assertEqual(self.run_build(name).returncode, 2)
        self.assertEqual(self.log.read_text(), '')

    def test_wrong_source_revision_stops_before_modules(self):
        for name in REVISIONS:
            with self.subTest(name=name):
                (self.root / name / 'revision').write_text('wrong')
                self.assertEqual(self.run_build(name).returncode, 2)
        self.assertEqual(self.log.read_text(), '')

    def test_lammps_preserves_existing_environment(self):
        existing = self.root / 'lammps/.venv'
        existing.mkdir()
        (existing / 'keep').write_text('existing environment')
        result = self.run_build('lammps')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual((existing / 'keep').read_text(), 'existing environment')
        self.assertEqual(self.log.read_text(), '')

    def test_lammps_external_venv_and_binary_match_the_wrapper(self):
        venv = self.root / 'custom venv'
        self.env.update(LAMMPS_VENV=str(venv), LAMMPS_BUILD_JOBS='3')
        result = self.run_build('lammps')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.log.read_text()
        self.assertIn(f'--prefix {self.root}/lammps/.python python=3.12.13', calls)
        self.assertIn(f'Python_EXECUTABLE={venv}/bin/python', calls)
        self.assertIn('BUILD_MPI=OFF', calls)
        self.assertIn('Kokkos_ARCH_AMPERE80=ON', calls)
        self.assertIn('PKG_EXTRA-MOLECULE=ON', calls)
        self.assertIn('--parallel 3', calls)
        self.assertIn('module load cudatoolkit-standalone/13.0.1', calls)
        self.assertTrue((venv / 'bin/activate').is_file())
        self.assertTrue((self.root / 'lammps/build-mliap-no-mpi/lmp').is_file())
        self.assertTrue((self.root / 'lammps/mofa-lammps-pip-freeze.txt').is_file())
        self.assertFalse((self.repo / 'deps').exists())
        result = self.run_wrapper('lammps', '-help')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ML-IAP', result.stdout)

    def test_lammps_build_failure_is_propagated(self):
        self.env['TEST_CMAKE_FAIL'] = '1'
        result = self.run_build('lammps')
        self.assertEqual(result.returncode, 41, result.stdout + result.stderr)
        self.assertNotIn('LAMMPS installed:', result.stdout)
        self.assertNotIn('--target install-python', self.log.read_text())

    def test_cp2k_build_preserves_outputs_and_passes_build_options(self):
        old = self.root / 'cp2k/exe/local_cuda'
        old.mkdir(parents=True)
        (old / 'keep').write_text('previous build')
        self.env['CP2K_BUILD_JOBS'] = '3'
        result = self.run_build('cp2k')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        backups = list(old.parent.glob('local_cuda.prev-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'keep').read_text(), 'previous build')
        calls = self.log.read_text()
        self.assertIn('--enable-cray=yes --enable-cuda --math-mode=cray', calls)
        self.assertIn('make -j 3 ARCH=local_cuda VERSION=ssmp psmp', calls)
        self.assertIn('module load cudatoolkit-standalone/13.0.1', calls)
        self.assertTrue((old / 'cp2k_shell.ssmp').is_file())
        self.assertTrue((old / 'cp2k_shell.psmp').is_file())
        self.assertFalse((self.repo / 'deps').exists())
        result = self.run_wrapper('cp2k', '--version')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'CP2K 2025.2')

    def test_cp2k_clean_option_preserves_dependencies_even_on_failure(self):
        install = self.root / 'cp2k/tools/toolchain/install'
        install.mkdir()
        (install / 'keep').write_text('old dependencies')
        self.env.update(CP2K_CLEAN_TOOLCHAIN='1', TEST_TOOLCHAIN_FAIL='1')
        result = self.run_build('cp2k')
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        backups = list(install.parent.glob('install.prev-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'keep').read_text(), 'old dependencies')
        self.assertNotIn('make ', self.log.read_text())
        self.assertNotIn('CP2K installed:', result.stdout)


if __name__ == '__main__':
    unittest.main()
