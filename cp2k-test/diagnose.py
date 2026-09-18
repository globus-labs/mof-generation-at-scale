#!/usr/bin/env python3
"""Two-node CP2K executable and ASE-shell checks, with and without MPS."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_diagnostic", ROOT / "lammps-test/diagnose.py")
gpu_diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_diag)


def run_command(argv, env=None, stdin=None, timeout=15):
    try:
        p = subprocess.run(argv, env=env, input=stdin, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        return dict(command=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(command=argv, returncode=124, stdout="", stderr="Command timed out")


def validate_output(path, require_exit=False):
    text = path.read_text(errors="replace") if path.exists() else ""
    if "SCF run NOT converged" in text or "SCF run not converged" in text:
        raise RuntimeError(f"SCF did not converge: {path}")
    if "SCF run converged" not in text:
        raise RuntimeError(f"No converged SCF recorded: {path}")
    if not re.search(r"Total number of message passing processes\s+8\b", text):
        raise RuntimeError(f"Output does not confirm eight MPI ranks: {path}")
    if not re.search(r"Number of threads for this process\s+8\b", text):
        raise RuntimeError(f"Output does not confirm eight OpenMP threads: {path}")
    startup = path.parent / "stdout.txt"
    accelerator_text = text + (startup.read_text(errors="replace") if startup.exists() else "")
    device_counts = re.findall(r"ACC: Number of devices/node\s+(\d+)\b", accelerator_text)
    if device_counts and any(count != "1" for count in device_counts):
        raise RuntimeError(f"Output does not confirm one visible accelerator per rank: {path}")
    accelerator_evidence = "device_count_header"
    if not device_counts:
        # CP2K 2025.2 shell mode suppresses the DBCSR startup banner. Its final
        # timing table still records executed GPU FFT routines. Build flags or
        # a GPU-related citation alone do not demonstrate accelerator use.
        gpu_timings = re.findall(
            r"^\s*pw_gpu_(?:c1dr3d|r3dc1d)_3d_ps\s+(\d+)\s+"
            r"[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$", text, re.M)
        if require_exit or not any(int(calls) > 0 and float(total) > 0
                                   for calls, total in gpu_timings):
            raise RuntimeError(f"Output does not confirm accelerator use: {path}")
        accelerator_evidence = "gpu_fft_timings"
    if require_exit and "PROGRAM ENDED AT" not in text:
        raise RuntimeError(f"Missing normal CP2K termination marker: {path}")
    energies = re.findall(r"ENERGY\| Total FORCE_EVAL[^\n]*?\s([-+\d.Ee]+)\s*$", text, re.M)
    if not energies:
        energies = re.findall(r"Total energy:\s*([-+\d.Ee]+)", text)
    if not energies or not all(math.isfinite(float(e)) for e in energies):
        raise RuntimeError(f"Missing or non-finite energies: {path}")
    return dict(scf_convergences=text.count("SCF run converged"), last_energy_hartree=float(energies[-1]),
                accelerator_evidence=accelerator_evidence)


def input_atoms(path):
    """Read coordinates and lattice from the existing, ASE-generated CP2K input."""
    from ase import Atoms
    from ase.calculators.cp2k import parse_input

    root = parse_input(path.read_text())
    coords = [line.split() for line in root.get_subsection("FORCE_EVAL/SUBSYS/COORD").keywords]
    cell = {line.split()[0]: line.split()[1:]
            for line in root.get_subsection("FORCE_EVAL/SUBSYS/CELL").keywords}
    if cell.get("PERIODIC") != ["XYZ"] or not coords:
        raise ValueError("Expected a periodic XYZ structure with explicit coordinates")
    return Atoms(symbols=[c[0] for c in coords], positions=[[float(v) for v in c[1:]] for c in coords],
                 cell=[[float(v) for v in cell[axis]] for axis in "ABC"], pbc=True)


def ase_check(output, command):
    """Use MOFA's settings, inline SET_POS, and the same shell for two evaluations."""
    import numpy as np
    from ase.calculators.cp2k import CP2K
    from mofa.simulation.dft.cp2k import _cp2k_options, _file_dir

    atoms = input_atoms(ROOT / "cp2k-test/cp2k.inp")
    output.mkdir(exist_ok=True)
    # Shell redirection captures the real MPI/CP2K stderr, not only Python stderr.
    command = command + " 2> " + shlex.quote(str(output / "launcher.stderr"))
    metadata = dict(command=command, atoms=len(atoms), evaluations=[], status="running",
                    set_pos_file=False, python=sys.executable)
    gpu_diag.write_json(output / "ase-result.json", metadata)
    os.chdir(output)
    try:
        with CP2K(command=command, label="cp2k", debug=True,
                  inp=(_file_dir / "cp2k-default-template.inp").read_text(),
                  max_scf=128, **_cp2k_options["default"]) as calc:
            atoms.calc = calc
            metadata["shell_version"] = calc._shell.version
            metadata["shell_pid"] = calc._shell._child.pid
            for evaluation in range(2):
                if evaluation:
                    atoms.positions[0, 0] += 0.001  # Force a second shell transaction.
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()
                stress = atoms.get_stress()
                if not math.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
                    raise RuntimeError("ASE received non-finite energy, forces or stress")
                metadata["evaluations"].append(dict(energy_eV=energy, force_shape=list(forces.shape),
                                                    max_force=float(np.abs(forces).max())))
                gpu_diag.write_json(output / "ase-result.json", metadata)
        metadata.update(status="passed", **validate_output(output / "cp2k.out", require_exit=False))
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        gpu_diag.write_json(output / "ase-result.json", metadata)


def node_check(output, base, action, label):
    """Run once on each allocated node, never changing an existing MPS daemon."""
    node = output / "nodes" / socket.gethostname().split(".")[0]
    node.mkdir(parents=True, exist_ok=True)
    manifest = node / "state.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {}
    env = gpu_diag.clean_environment()
    env.update(CUDA_MPS_PIPE_DIRECTORY=str(base / "pipes"), CUDA_MPS_LOG_DIRECTORY=str(base / "logs"))

    def save_command(name, argv, stdin=None, timeout=15):
        result = run_command(argv, env=env, stdin=stdin, timeout=timeout)
        gpu_diag.write_json(node / (name + ".json"), result)
        if result["returncode"]:
            raise RuntimeError(f"{socket.gethostname()}: {name} failed: {result['stderr']}")
        return result["stdout"]

    def archive():
        if (base / "logs").exists():
            shutil.copytree(base / "logs", node / "mps-logs", dirs_exist_ok=True)

    if action == "stop":
        try:
            if state.get("mps_started"):
                save_command("mps-stop", ["nvidia-cuda-mps-control"], stdin="quit\n")
                state["mps_started"] = False
                gpu_diag.write_json(manifest, state)
                archive()
                shutil.rmtree(base)
        finally:
            archive()
        return

    inventory = save_command(label + "-inventory", ["nvidia-smi", f"--query-gpu={gpu_diag.GPU_FIELDS}", "--format=csv,noheader"])
    xml = save_command(label + "-health", ["nvidia-smi", "-q", "-x"])
    gpus = gpu_diag.parse_inventory(inventory)
    issues = gpu_diag.health_issues(xml)
    counts = gpu_diag.uncorrectable_counts(xml)
    if state.get("gpus") and state["gpus"] != gpus:
        issues.append("GPU inventory changed")
    issues += [f"New uncorrectable ECC errors: {key}" for key, value in counts.items()
               if key in state.get("ecc", {}) and value > state["ecc"][key]]
    if issues:
        raise RuntimeError(f"{socket.gethostname()}: " + "; ".join(issues))
    state.update(gpus=gpus, ecc=counts)
    gpu_diag.write_json(manifest, state)
    if action == "preflight":
        processes = save_command("initial-processes", ["ps", "-eo", "pid=,args="])
        if gpu_diag.existing_mps_processes(processes):
            raise RuntimeError("Existing MPS daemon prevents an isolated baseline; it was not stopped")
    elif action == "start":
        base.mkdir(mode=0o700)
        for name in ("pipes", "logs"):
            (base / name).mkdir(mode=0o700)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(g["uuid"] for g in gpus)
        state["mps_started"] = True  # Enables cleanup of partially completed startup.
        gpu_diag.write_json(manifest, state)
        try:
            save_command("mps-start", ["nvidia-cuda-mps-control", "-d"])
            save_command("mps-server-start", ["nvidia-cuda-mps-control"], stdin=f"start_server -uid {os.getuid()}\n")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                pids = save_command("mps-servers", ["nvidia-cuda-mps-control"], stdin="get_server_list\n", timeout=3)
                if re.search(r"(?m)^\s*\d+\s*$", pids):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("MPS server did not become available")
        finally:
            archive()
    elif action == "snapshot":
        if state.get("mps_started"):
            save_command(label + "-mps-servers", ["nvidia-cuda-mps-control"], stdin="get_server_list\n")
        archive()
    print(f"{socket.gethostname()}: {action} passed", flush=True)


def execute_stage(command, directory, env, timeout, interrupted):
    """Keep stdout/stderr and kill only this launcher's process group on timeout."""
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(command=command, started_at=gpu_diag.utc(), status="running")
    gpu_diag.write_json(directory / "launch.json", record)
    start = time.monotonic()
    with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
        proc = subprocess.Popen(command, shell=isinstance(command, str), cwd=ROOT, env=env,
                                stdout=out, stderr=err, start_new_session=True)
        heartbeat = start + 30
        try:
            while proc.poll() is None:
                if interrupted[0] or time.monotonic() - start > timeout:
                    record["status"] = "interrupted" if interrupted[0] else "timeout"
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
                    break
                if time.monotonic() >= heartbeat:
                    print(f"{gpu_diag.utc()} {directory.name}: active for {time.monotonic()-start:.0f}s", flush=True)
                    heartbeat = time.monotonic() + 30
                time.sleep(0.2)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            record.update(exit_code=proc.poll(), elapsed_seconds=round(time.monotonic()-start, 3),
                          finished_at=gpu_diag.utc())
            if record["status"] == "running":
                record["status"] = "passed" if proc.returncode == 0 else "failed"
            gpu_diag.write_json(directory / "launch.json", record)
    return record


def run_diagnostic(output):
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Use a fresh, two-node PBS compute allocation")
    hosts = list(dict.fromkeys(Path(os.environ["PBS_NODEFILE"]).read_text().split()))
    if len(hosts) != 2:
        raise RuntimeError("Exactly two allocated nodes are required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cp2k-hostfiles").mkdir()
    hostfile = output / "cp2k-hostfiles/local_hostfile.0000"
    hostfile.write_text("\n".join(hosts) + "\n")
    from mofa.utils.config import load_variable
    config = load_variable(ROOT / "configs/polaris/polaris-repo.py", "hpc_config")
    config.run_dir = output
    shell_command = config.dft_cmd
    if config.nodes_per_cp2k != 2 or config.gpus_per_node != 4:
        raise RuntimeError("Workflow configuration no longer specifies two nodes/four ranks each")
    executable_command = shell_command.replace("CP2K_BINARY=cp2k_shell.psmp", "CP2K_BINARY=cp2k.psmp")
    if executable_command == shell_command:
        raise RuntimeError("Could not derive the executable launch from the workflow shell launch")
    env = gpu_diag.clean_environment()
    env.update(PARSL_WORKER_RANK="0", OMP_NUM_THREADS="8")
    base = Path("/tmp") / ("mofa-cp2k-" + str(time.time_ns()))
    helper = ["mpiexec", "--no-vni", "-n", "2", "--ppn", "1", "--hostfile", str(hostfile),
              sys.executable, str(Path(__file__).resolve()), "node", "--output", str(output),
              "--mps-base", str(base)]
    summary = dict(status="running", hosts=hosts, started_at=gpu_diag.utc(), stages=[],
                   shell_command=shell_command, executable_command=executable_command)
    interrupted = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: interrupted.__setitem__(0, signum))
    deadline = time.monotonic() + 55 * 60

    def save():
        gpu_diag.write_json(output / "summary.json", summary)

    def nodes(action, label):
        result = execute_stage(helper + ["--action", action, "--label", label],
                               output / ("node-" + label), env, 90, interrupted)
        if result["status"] != "passed":
            raise RuntimeError(f"Node {action} failed; see node-{label}/stderr.txt")

    print(f"Results: {output}", flush=True)
    save()
    try:
        nodes("preflight", "before")
        for mps in (False, True):
            if mps:
                nodes("start", "mps-start")
                env.update(CUDA_MPS_PIPE_DIRECTORY=str(base / "pipes"), CUDA_MPS_LOG_DIRECTORY=str(base / "logs"))
            for interface in ("executable", "ase"):
                if interrupted[0] or time.monotonic() >= deadline:
                    raise RuntimeError("Interrupted or 55-minute execution deadline reached")
                label = f"{len(summary['stages'])+1:02d}-{interface}-mps-{'on' if mps else 'off'}"
                stage = output / label
                stage.mkdir()
                if interface == "executable":
                    shutil.copy2(ROOT / "cp2k-test/cp2k.inp", stage / "cp2k.inp")
                    # Preserve the source; don't accept unconverged SCF as a passing diagnostic.
                    path = stage / "cp2k.inp"
                    path.write_text(path.read_text().replace("         IGNORE_CONVERGENCE_FAILURE\n", ""))
                    command = ("cd " + shlex.quote(str(stage)) + " && " + executable_command +
                               " -i cp2k.inp -o cp2k.out")
                else:
                    command_file = stage / "command.json"
                    gpu_diag.write_json(command_file, {"command": shell_command})
                    command = [sys.executable, "-u", str(Path(__file__).resolve()), "ase",
                               "--output", str(stage), "--command-file", str(command_file)]
                print(f"{gpu_diag.utc()} Starting {label}", flush=True)
                result = execute_stage(command, stage, env, min(1200, deadline-time.monotonic()), interrupted)
                result.update(name=label, mps=mps, interface=interface)
                if result["status"] == "passed":
                    try:
                        result.update(validate_output(stage / "cp2k.out", require_exit=interface == "executable"))
                    except Exception as exc:
                        result.update(status="failed", validation_error=str(exc))
                summary["stages"].append(result)
                save()
                nodes("snapshot", label + "-after")
                print(f"{gpu_diag.utc()} {label}: {result['status']}", flush=True)
                if result["status"] != "passed":
                    raise RuntimeError(f"{label} failed; inspect its captured stderr and shell protocol")
        summary["status"] = "passed"
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        # Always target only the per-node daemons recorded in this run's manifest.
        interrupted[0] = False
        try:
            nodes("stop", "cleanup")
        except Exception as exc:
            summary.update(status="failed", cleanup_error=str(exc))
        summary["finished_at"] = gpu_diag.utc()
        save()
        print(f"Status: {summary['status']}\nResults: {output}", flush=True)
    return 0 if summary["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "node", "ase"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mps-base", type=Path)
    parser.add_argument("--action", choices=("preflight", "snapshot", "start", "stop"))
    parser.add_argument("--label", default="check")
    parser.add_argument("--command-file", type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / "run" / f"cp2k-diagnostic-{os.environ.get('PBS_JOBID', 'local')}-{time.time_ns()}").resolve()
    if args.mode == "node":
        node_check(output, args.mps_base, args.action, args.label)
    elif args.mode == "ase":
        ase_check(output, json.loads(args.command_file.read_text())["command"])
    else:
        return run_diagnostic(output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
