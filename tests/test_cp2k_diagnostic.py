"""Offline validation of the CP2K diagnostic's input and acceptance checks."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cp2k_diagnostic", ROOT / "cp2k-test/diagnose.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)

OUTPUT = '''DBCSR| ACC: Number of devices/node 1
GLOBAL| Total number of message passing processes 8
GLOBAL| Number of threads for this process 8
*** SCF run converged in 44 steps ***
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -682.916998751338269
PROGRAM ENDED AT 2026-09-17
'''


class CP2KDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cp2k diagnostic ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "cp2k.out"

    def test_real_structure_is_read_without_modifying_source(self):
        path = ROOT / "cp2k-test/cp2k.inp"
        before = path.read_bytes()
        atoms = diagnostic.input_atoms(path)
        self.assertEqual(len(atoms), 112)
        self.assertTrue(all(atoms.pbc))
        self.assertGreater(atoms.get_volume(), 10000)
        self.assertEqual(path.read_bytes(), before)

    def test_executable_and_ase_startup_logs(self):
        self.output.write_text(OUTPUT)
        self.assertAlmostEqual(diagnostic.validate_output(self.output, True)["last_energy_hartree"], -682.9169987513383)
        self.output.write_text("\n".join(OUTPUT.splitlines()[1:-1]))
        (self.root / "stdout.txt").write_text("Received: DBCSR| ACC: Number of devices/node 1\n")
        self.assertEqual(diagnostic.validate_output(self.output)["scf_convergences"], 1)
        with self.assertRaisesRegex(RuntimeError, "termination"):
            diagnostic.validate_output(self.output, True)

    def test_unconverged_and_incorrect_rank_counts_fail(self):
        for text in (OUTPUT.replace("SCF run converged", "SCF run NOT converged"),
                     OUTPUT.replace("passing processes 8", "passing processes 4"),
                     OUTPUT.replace("devices/node 1", "devices/node 0")):
            self.output.write_text(text)
            with self.assertRaises(RuntimeError):
                diagnostic.validate_output(self.output, True)

    def test_shell_gpu_timings_without_startup_banner(self):
        # Real CP2K 2025.2 shell output from job 7631545 omitted the banner.
        shell_output = "\n".join(OUTPUT.splitlines()[1:])
        timing = " pw_gpu_c1dr3d_3d_ps 568 13.2 13.130 13.292 31.024 31.147\n"
        self.output.write_text(shell_output + "\n" + timing)
        self.assertEqual(diagnostic.validate_output(self.output)["accelerator_evidence"],
                         "gpu_fft_timings")
        with self.assertRaisesRegex(RuntimeError, "accelerator"):
            diagnostic.validate_output(self.output, True)
        for evidence in ("CP2K| cp2kflags: elpa_nvidia_gpu\n",
                         timing.replace("568", "0"), timing.replace("31.147", "0.000"),
                         "DBCSR| ACC: Number of devices/node 0\n" + timing):
            self.output.write_text(shell_output + "\n" + evidence)
            with self.assertRaisesRegex(RuntimeError, "accelerator"):
                diagnostic.validate_output(self.output)

    def test_timeout_and_stderr_are_preserved(self):
        # Use a local shell so starting Python on a network filesystem cannot
        # consume the timeout before the child emits the diagnostic.
        result = diagnostic.execute_stage(
            ["/bin/sh", "-c", "printf 'child diagnostic\\n' >&2; exec sleep 20"],
            self.root / "timeout", os.environ.copy(), 2, [False])
        self.assertEqual(result["status"], "timeout")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("child diagnostic", (self.root / "timeout/stderr.txt").read_text())


if __name__ == "__main__":
    unittest.main()
