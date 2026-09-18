"""Exercise the diagnostic with real child processes and simulated GPU services."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lammps_diagnostic", ROOT / "lammps-test/diagnose.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)

INPUT = '''units metal
atom_style atomic
read_data data.lmp
pair_style mliap unified "model with spaces.pt" 0
pair_coeff * * C
variable Nevery equal 50
thermo 10
minimize 0. 1.0e-2 100 10000
reset_timestep 0
thermo ${Nevery}
run 100
'''

INVENTORY = "\n".join(f"{i}, GPU-{i}, 0000:{i}:00.0, A100, 570, Default, Disabled" for i in range(4))
HEALTH = '<nvidia_smi_log>' + ''.join(
    f'<gpu><uuid>GPU-{i}</uuid><gpu_recovery_action>None</gpu_recovery_action></gpu>'
    for i in range(4)) + '</nvidia_smi_log>'


class FakeHardware(diagnostic.Diagnostic):
    def __init__(self, args):
        super().__init__(args)
        self.controls = []
        self.start_calls = 0
        self.existing_mps = False
        self.start_failure = False
        self.observe_clients = True
        self.server_error = False

    def command(self, argv, **kwargs):
        output, code = "", 0
        if argv[0] == "nvidia-smi":
            output = HEALTH if "-x" in argv else INVENTORY
        elif argv[0] == "ps" and self.existing_mps:
            output = "123 /usr/bin/nvidia-cuda-mps-control -d\n"
        elif argv == ["nvidia-cuda-mps-control", "-d"]:
            self.start_calls += 1
            code = 1 if self.start_failure else 0
        elif argv == ["nvidia-cuda-mps-control"]:
            cmd = kwargs["stdin"].strip()
            self.controls.append(cmd)
            if cmd == "get_server_list":
                output = "123\n"
            elif cmd.startswith("get_client_list") and self.observe_clients:
                output = "\n".join(str(c["process"].pid) for c in self.clients
                                   if c["process"].poll() is None)
                if self.server_error:
                    (self.mps_dir / "logs/server.log").write_text("Server encountered a fatal exception\n")
        return dict(command=argv, returncode=code, stdout=output, stderr="")


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lammps diagnostic ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "in.lammps").write_text(INPUT)
        (self.source / "data.lmp").write_text("fixture data\n")
        (self.source / "model with spaces.pt").touch()
        self.nodefile = self.root / "nodes"
        self.nodefile.write_text(socket.gethostname() + "\n")
        self.wrapper = self.root / "lmp-wrapper"
        self.wrapper.write_text("#!" + sys.executable + "\n" + textwrap.dedent('''\
            import json, os, pathlib, re, time
            cwd = pathlib.Path.cwd()
            pathlib.Path('observed.json').write_text(json.dumps({
                'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
                'omp': os.environ['OMP_NUM_THREADS'],
                'cpus': sorted(os.sched_getaffinity(0)),
                'mps_pipe': os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}))
            time.sleep(float(os.environ.get('FAKE_SLEEP', '0.3')))
            if os.environ.get('FAKE_FAIL_GPU') == os.environ['CUDA_VISIBLE_DEVICES']:
                print('CUDA error: out of memory', flush=True)
                raise SystemExit(1)
            steps = int(re.search(r'^run (\\d+)', pathlib.Path('in.lammps').read_text(), re.M)[1])
            pathlib.Path('log.lammps').write_text(
                'KOKKOS mode with Kokkos version 5\\n  will use up to 1 GPU(s) per node\\n'
                'Step Temp PotEng\\n0 300 -20\\n' + str(steps) + ' 301 -21\\n'
                'Loop time of 0.1 on 1 procs for ' + str(steps) + ' steps with 10 atoms\\n'
                'Total wall time: 0:00:01\\n')
            '''))
        self.wrapper.chmod(0o755)

    def args(self, **overrides):
        values = dict(output=self.root / "results", input_dir=self.source, wrapper=self.wrapper,
                      max_per_gpu=4, full=False, stage_timeout=10, prepare_only=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def run_fake(self, runner, **env):
        with patch.dict(os.environ, dict(PBS_JOBID="test-job", PBS_NODEFILE=str(self.nodefile), **env)), \
             patch.object(diagnostic.shutil, "which", return_value="/mock/command"):
            # Env is captured on construction, before PBS setup in this helper.
            runner.env.update(env)
            return runner.run()

    def test_short_and_full_inputs_preserve_source(self):
        steps = diagnostic.prepare_mace(self.source, self.root / "short")
        text = (self.root / "short/in.lammps").read_text()
        self.assertEqual(steps, 20)
        self.assertIn("minimize 0. 1.0e-2 5 10000", text)
        self.assertIn("run 20", text)
        self.assertIn("thermo 1", text)
        self.assertIn(str(self.source / "model with spaces.pt"), text)
        self.assertEqual((self.source / "in.lammps").read_text(), INPUT)
        self.assertEqual(diagnostic.prepare_mace(self.source, self.root / "full", True), 100)
        self.assertIn("minimize 0. 1.0e-2 100 10000", (self.root / "full/in.lammps").read_text())

    def test_prepare_only_never_calls_gpu_commands(self):
        runner = diagnostic.Diagnostic(self.args(prepare_only=True))
        with patch.object(runner, "command", side_effect=AssertionError("GPU command in prepare-only")):
            self.assertEqual(runner.run(), 0)
        self.assertEqual(runner.summary["status"], "prepared_only")
        self.assertEqual([s[3] for s in runner.summary["stage_plan"]], [1, 1, 1, 2, 3, 4])

    @unittest.skipUnless(len(os.sched_getaffinity(0)) >= 16, "Needs 16 allowed CPUs for binding test")
    def test_six_stages_run_sixteen_independent_clients(self):
        runner = FakeHardware(self.args())
        self.assertEqual(self.run_fake(runner), 0)
        self.assertEqual(runner.start_calls, 1)
        self.assertIn("quit", runner.controls)
        stages = runner.summary["stages"]
        self.assertEqual([len(s["clients"]) for s in stages], [4, 4, 4, 8, 12, 16])
        final = stages[-1]
        self.assertTrue(final["all_mps_clients_observed"])
        self.assertEqual(len({c["cpu"] for c in final["clients"]}), 16)
        for stage in stages:
            for record in stage["clients"]:
                path = runner.output / record["directory"]
                observed = json.loads((path / "observed.json").read_text())
                self.assertEqual(observed["gpu"], record["gpu_uuid"])
                self.assertEqual(observed["cpus"], [record["cpu"]])
                self.assertEqual(observed["omp"], "1")
                self.assertEqual(bool(observed["mps_pipe"]), stage["mps"])
                self.assertEqual(record["completed_step"], 20)
                self.assertTrue((path / "result.json").exists())
        self.assertFalse(runner.mps_dir.exists())
        self.assertEqual(runner.summary["highest_verified_clients_per_gpu_on_all_four_gpus"], 4)

    def test_failure_stops_escalation_only_on_failed_gpu(self):
        runner = FakeHardware(self.args(max_per_gpu=2))
        self.assertEqual(self.run_fake(runner, FAKE_FAIL_GPU="GPU-2"), 1)
        stages = runner.summary["stages"]
        failed = [c for c in stages[0]["clients"] if c["gpu_uuid"] == "GPU-2"]
        self.assertEqual(failed[0]["failure"], "out_of_memory")
        for stage in stages[1:]:
            self.assertNotIn("2", stage["gpu_indices"])
        self.assertEqual(len(stages[-1]["clients"]), 6)

    def test_existing_mps_is_not_stopped(self):
        runner = FakeHardware(self.args())
        runner.existing_mps = True
        self.assertEqual(self.run_fake(runner), 1)
        self.assertEqual(runner.start_calls, 0)
        self.assertNotIn("quit", runner.controls)
        self.assertIn("Existing MPS", runner.summary["error"])

    def test_failed_startup_still_cleans_own_mps(self):
        runner = FakeHardware(self.args(max_per_gpu=1))
        runner.start_failure = True
        self.assertEqual(self.run_fake(runner), 1)
        self.assertIn("quit", runner.controls)
        self.assertTrue((runner.output / "mps-start.json").exists())

    def test_timeout_is_not_reported_as_cuda_error(self):
        runner = FakeHardware(self.args(stage_timeout=0.15))
        self.assertEqual(self.run_fake(runner, FAKE_SLEEP="10"), 1)
        stage = runner.summary["stages"][0]
        self.assertTrue(all(c["failure"] == "timeout" for c in stage["clients"]))
        self.assertTrue(all(c["exit_code"] != 0 for c in stage["clients"]))
        self.assertEqual(len(runner.summary["stages"]), 1)

    def test_successful_exit_without_mps_connectivity_does_not_pass(self):
        runner = FakeHardware(self.args(max_per_gpu=1))
        runner.observe_clients = False
        self.assertEqual(self.run_fake(runner), 1)
        final = runner.summary["stages"][-1]
        self.assertFalse(final["all_mps_clients_observed"])
        self.assertTrue(all(c["failure"] == "mps_overlap_not_observed" for c in final["clients"]))

    def test_server_fault_is_preserved_and_stops_escalation(self):
        runner = FakeHardware(self.args(max_per_gpu=2))
        runner.server_error = True
        self.assertEqual(self.run_fake(runner), 1)
        self.assertEqual(len(runner.summary["stages"]), 3)
        self.assertIn("fatal", runner.summary["stages"][-1]["mps_log_errors"][0])
        self.assertIn("fatal", (runner.output / "mps-logs/server.log").read_text())

    def test_full_mode_checks_only_selected_concurrency_after_baselines(self):
        self.assertEqual([s[3] for s in diagnostic.stage_specs(4, full=True)], [1, 1, 4])

    def test_ecc_counters_cover_old_and_new_nvidia_schemas(self):
        xml = ('<nvidia_smi_log><gpu><uuid>GPU-a</uuid><ecc_errors><volatile>'
               '<double_bit><total>2</total></double_bit><dram_uncorrectable>3</dram_uncorrectable>'
               '<dram_correctable>7</dram_correctable></volatile></ecc_errors></gpu></nvidia_smi_log>')
        self.assertEqual(diagnostic.uncorrectable_counts(xml), {
            'GPU-a/volatile/double_bit/total': 2, 'GPU-a/volatile/dram_uncorrectable': 3})

    def test_inventory_and_health_reject_reset_and_mig(self):
        self.assertEqual(len(diagnostic.parse_inventory(INVENTORY)), 4)
        with self.assertRaises(ValueError):
            diagnostic.parse_inventory(INVENTORY.replace("Disabled", "[GPU requires reset]", 1))
        with self.assertRaises(ValueError):
            diagnostic.parse_inventory(INVENTORY.replace("Disabled", "Enabled", 1))
        self.assertEqual(diagnostic.health_issues(HEALTH), [])
        self.assertIn("GPU-0", diagnostic.health_issues(HEALTH.replace(
            "<gpu_recovery_action>None", "<gpu_recovery_action>Reset", 1))[0])

    def test_nan_and_truncated_trajectory_are_not_successes(self):
        folder = self.root / "log-test"
        folder.mkdir()
        base = ('KOKKOS mode\nwill use up to 1 GPU(s) per node\n'
                'Step Temp\n0 300\n20 301\n'
                'Loop time of 0.2 on 1 procs for 20 steps with 10 atoms\nTotal wall time: 0:00:01\n')
        path = folder / "log.lammps"
        path.write_text(base)
        self.assertNotIn("validation_error", diagnostic.check_lammps(folder, 20))
        path.write_text(base.replace("20 301", "20 nan"))
        self.assertIn("non-finite", diagnostic.check_lammps(folder, 20)["validation_error"])
        path.write_text(base.replace("20 301", "10 301"))
        self.assertIn("Did not complete", diagnostic.check_lammps(folder, 20)["validation_error"])


if __name__ == "__main__":
    unittest.main()
