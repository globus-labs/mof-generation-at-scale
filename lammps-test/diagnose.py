#!/usr/bin/env python3
"""Standalone, standard-library-only Polaris LAMMPS/MPS diagnostic.

No MOFA services, GPU resets, package changes, or job submissions are performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
GPU_FIELDS = "index,uuid,pci.bus_id,name,driver_version,compute_mode,mig.mode.current"
SAMPLE_FIELDS = "index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu"
MPS_NAMES = {"nvidia-cuda-mps-control", "nvidia-cuda-mps-server"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stage_specs(max_per_gpu=4, full=False):
    stages = [("01-lj-no-mps", "lj", False, 1),
              ("02-mace-no-mps", "mace", False, 1)]
    levels = [max_per_gpu] if full else range(1, max_per_gpu + 1)
    stages += [(f"{i:02d}-mace-mps-{n}", "mace", True, n)
               for i, n in enumerate(levels, 3)]
    return stages


def prepare_mace(source, destination, full=False):
    """Change only copies; reject unfamiliar input rather than silently mis-test."""
    text = (source / "in.lammps").read_text()
    runs = re.findall(r"^\s*run\s+(\d+)\s*(?:#.*)?$", text, re.M)
    if len(runs) != 1 or int(runs[0]) <= 0:
        raise ValueError("Expected exactly one literal, positive run length in in.lammps")
    if len(re.findall(r"^\s*read_data\s+data\.lmp\s*$", text, re.M)) != 1:
        raise ValueError("Expected one read_data data.lmp in in.lammps")
    pair_lines = re.findall(r"^\s*pair_style\s+mliap\s+unified\s+(.+)$", text, re.M)
    if len(pair_lines) != 1:
        raise ValueError("Expected one pair_style mliap unified model in in.lammps")
    model_args = shlex.split(pair_lines[0])
    model = Path(model_args[0])
    if not model.is_absolute():
        model = (source / model).resolve()
    if not model.is_file():
        raise ValueError(f"MACE model does not exist: {model}")
    # LAMMPS accepts double-quoted paths. The model is shared read-only by clients.
    model_path = str(model).replace('\\', '\\\\').replace('"', '\\"')
    text = re.sub(r"^\s*pair_style\s+mliap\s+unified\s+.+$",
                  lambda _: f'pair_style mliap unified "{model_path}" ' + " ".join(model_args[1:]),
                  text, count=1, flags=re.M)
    steps = int(runs[0]) if full else 20
    if not full:
        text, count = re.subn(r"^(\s*minimize\s+\S+\s+\S+)\s+\d+(\s+\d+.*)$",
                              r"\g<1> 5\2", text, flags=re.M)
        if count != 1:
            raise ValueError("Expected one minimize command with literal iteration limits")
        text = re.sub(r"^\s*run\s+\d+\s*(?:#.*)?$", "run 20", text, flags=re.M)
        text = re.sub(r"^\s*variable\s+Nevery\s+equal\s+.+$",
                      "variable Nevery equal 1", text, flags=re.M)
    # Frequent thermo output does not change dynamics, including in --full mode.
    text = re.sub(r"^\s*thermo\s+.+$", "thermo 1", text, flags=re.M)
    destination.mkdir(parents=True)
    (destination / "in.lammps").write_text(text)
    shutil.copy2(source / "data.lmp", destination / "data.lmp")
    return steps


def parse_inventory(text):
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    gpus = []
    for row in rows:
        if len(row) != 7 or not row[0].isdigit() or not row[1].startswith("GPU-"):
            raise ValueError(f"Invalid GPU inventory: {row}")
        if any("requires reset" in value.lower() or "gpu is lost" in value.lower() for value in row):
            raise ValueError(f"Unhealthy GPU: {row}")
        if row[6].strip().lower() not in {"disabled", "n/a", "[n/a]"}:
            raise ValueError(f"This four-GPU diagnostic requires MIG disabled: {row}")
        gpus.append(dict(zip(GPU_FIELDS.split(","), (value.strip() for value in row))))
    if len(gpus) != 4 or len({g["uuid"] for g in gpus}) != 4:
        raise ValueError(f"Expected four distinct GPUs, found {len(gpus)}")
    return sorted(gpus, key=lambda g: int(g["index"]))


def health_issues(xml_text):
    """Return explicit recovery/reset requests; retain full XML for other evidence."""
    issues = []
    for gpu in ET.fromstring(xml_text).findall("gpu"):
        identifier = gpu.findtext("uuid", gpu.get("id", "unknown"))
        for item in gpu.iter():
            value = (item.text or "").strip()
            if "recovery_action" in item.tag and value.lower() not in {"", "none", "n/a", "[n/a]"}:
                issues.append(f"{identifier}: {item.tag}={value}")
            if item.tag in {"reset_required", "drain_and_reset_recommended"} and value.lower() == "yes":
                issues.append(f"{identifier}: {item.tag}={value}")
            if "requires reset" in value.lower() or "gpu is lost" in value.lower():
                issues.append(f"{identifier}: {value}")
    return issues


def uncorrectable_counts(xml_text):
    """Track increases in volatile ECC errors across NVIDIA XML schema versions."""
    counts = {}
    for gpu in ET.fromstring(xml_text).findall("gpu"):
        identifier = gpu.findtext("uuid", gpu.get("id", "unknown"))
        volatile = gpu.find("ecc_errors/volatile")
        if volatile is None:
            continue

        def visit(element, path):
            path = path + "/" + element.tag
            value = (element.text or "").strip()
            if ("uncorrectable" in path or "double_bit" in path) and value.isdigit():
                counts[identifier + path] = int(value)
            for child in element:
                visit(child, path)

        visit(volatile, "")
    return counts


def failure_category(text):
    lower = text.lower()
    if any(s in lower for s in ("out of memory", "out_of_memory", "memoryallocation", "cudaerrormemoryallocation")):
        return "out_of_memory"
    if any(s in lower for s in ("cudaerrormps", "mps client failed", "invalid cuda_visible_devices")):
        return "mps_error"
    if any(s in lower for s in ("cuinit failed", "cuda error", "cudaerror", "cudagetdevicecount", "gpu requires reset")):
        return "cuda_error"
    if re.search(r"(?im)^\s*(?:ERROR(?:\s|:)|Exception:)", text):
        return "simulation_error"
    return None


def check_lammps(directory, steps):
    """Check actual completion and thermo values, not just process exit status."""
    path = directory / "log.lammps"
    if not path.is_file():
        return {"validation_error": "Missing log.lammps"}
    text = path.read_text(errors="replace")
    loops = re.findall(r"Loop time of ([\d.eE+-]+).*? for (\d+) steps", text)
    rows, columns = [], None
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "Step":
            columns = len(fields)
            continue
        if columns and len(fields) == columns:
            try:
                row = [float(field) for field in fields]
            except ValueError:
                columns = None
            else:
                rows.append(row)
        elif fields:
            columns = None
    result = {"completed_step": rows[-1][0] if rows and math.isfinite(rows[-1][0]) else None,
              "md_loop_seconds": float(loops[-1][0]) if loops else None}
    if not rows or not all(math.isfinite(x) for row in rows for x in row):
        result["validation_error"] = "Missing or non-finite thermo output"
    elif not loops or int(loops[-1][1]) != steps or rows[-1][0] != steps:
        result["validation_error"] = f"Did not complete the requested {steps} MD steps"
    elif "Total wall time:" not in text:
        result["validation_error"] = "Missing normal LAMMPS termination marker"
    elif "KOKKOS mode" not in text or not re.search(r"will use up to 1 GPU", text):
        result["validation_error"] = "Log does not confirm one Kokkos GPU per process"
    return result


def existing_mps_processes(text):
    found = []
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) >= 2 and Path(fields[1]).name in MPS_NAMES:
            found.append(line.strip())
    return found


def clean_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA_MPS_")}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONNOUSERSITE="1", OMP_PROC_BIND="false")
    return env


class Diagnostic:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.env = clean_environment()
        self.deadline = time.monotonic() + 55 * 60
        self.stop_signal = None
        self.clients = []
        self.mps_dir = None
        self.mps_env = None
        self.mps_started = False
        self.mps_log_offsets = {}
        self.gpus = []
        self.ecc_counts = {}
        self.summary = {"started_at": utc(), "hostname": socket.gethostname(),
                        "job_id": os.environ.get("PBS_JOBID"), "mode": "full" if args.full else "short",
                        "max_per_gpu": args.max_per_gpu, "stages": [], "status": "running"}

    def save(self):
        write_json(self.output / "summary.json", self.summary)

    def command(self, argv, *, env=None, stdin=None, timeout=10):
        try:
            proc = subprocess.run(argv, env=env or self.env, input=stdin, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            return {"command": argv, "returncode": proc.returncode,
                    "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as exc:
            return {"command": argv, "returncode": 124, "stdout": "",
                    "stderr": f"Command timed out after {exc.timeout} seconds"}

    def snapshot(self, label):
        folder = self.output / "health" / label
        folder.mkdir(parents=True)
        inventory = self.command(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"])
        details = self.command(["nvidia-smi", "-q", "-x"])
        write_json(folder / "inventory.json", inventory)
        write_json(folder / "query.json", details)
        (folder / "nvidia-smi.xml").write_text(details["stdout"])
        if inventory["returncode"] or details["returncode"]:
            raise RuntimeError(f"GPU health query failed; see {folder}")
        gpus = parse_inventory(inventory["stdout"])
        issues = health_issues(details["stdout"])
        if issues:
            raise RuntimeError("GPU health failure: " + "; ".join(issues))
        if self.gpus and gpus != self.gpus:
            raise RuntimeError("GPU inventory or configuration changed during the test")
        counts = uncorrectable_counts(details["stdout"])
        increased = [key for key, value in counts.items()
                     if key in self.ecc_counts and value > self.ecc_counts[key]]
        self.ecc_counts = counts
        if increased:
            raise RuntimeError("New uncorrectable GPU ECC errors: " + "; ".join(increased))
        return gpus

    def control(self, command):
        result = self.command(["nvidia-cuda-mps-control"], env=self.mps_env,
                              stdin=command + "\n", timeout=3)
        with (self.output / "mps-control.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": utc(), "input": command, **result}) + "\n")
        return result

    def mps_clients(self):
        servers = self.control("get_server_list")
        pids = set()
        for pid in re.findall(r"(?m)^\s*(\d+)\s*$", servers["stdout"]):
            result = self.control(f"get_client_list {pid}")
            if result["returncode"] == 0:
                pids.update(int(p) for p in re.findall(r"(?m)^\s*(\d+)\s*$", result["stdout"]))
        return pids

    def archive_mps(self, destination):
        if self.mps_dir is not None:
            shutil.copytree(self.mps_dir / "logs", destination, dirs_exist_ok=True)

    def new_mps_errors(self):
        errors = []
        if self.mps_dir is None:
            return errors
        for path in (self.mps_dir / "logs").glob("*.log"):
            data = path.read_bytes()
            offset = self.mps_log_offsets.get(path.name, 0)
            if offset > len(data):
                offset = 0
            text = data[offset:].decode(errors="replace")
            self.mps_log_offsets[path.name] = len(data)
            for line in text.splitlines():
                if re.search(r"fatal|failed to start|invalid cuda_visible_devices|out of memory|"
                             r"not a valid gpu id|gpu requires reset", line, re.I):
                    errors.append(f"{path.name}: {line}")
        return errors

    def start_mps(self):
        self.mps_dir = Path(tempfile.mkdtemp(prefix="mofa-lammps-mps-", dir="/tmp"))
        for name in ("pipes", "logs"):
            (self.mps_dir / name).mkdir(mode=0o700)
        self.mps_env = dict(self.env, CUDA_MPS_PIPE_DIRECTORY=str(self.mps_dir / "pipes"),
                            CUDA_MPS_LOG_DIRECTORY=str(self.mps_dir / "logs"),
                            CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in self.gpus))
        self.summary["mps_directories"] = {key: value for key, value in self.mps_env.items()
                                           if key.startswith("CUDA_MPS_")}
        self.mps_started = True  # Also clean up a partially completed startup.
        start = self.command(["nvidia-cuda-mps-control", "-d"], env=self.mps_env)
        write_json(self.output / "mps-start.json", start)
        if start["returncode"]:
            raise RuntimeError("MPS control daemon failed to start")
        start = self.control(f"start_server -uid {os.getuid()}")
        if start["returncode"]:
            raise RuntimeError("MPS server startup command failed")
        deadline = min(self.deadline, time.monotonic() + 30)
        while time.monotonic() < deadline and not self.stop_signal:
            servers = self.control("get_server_list")
            pids = re.findall(r"(?m)^\s*(\d+)\s*$", servers["stdout"])
            if servers["returncode"] == 0 and pids:
                return  # Actual CUDA client connectivity is checked in each stage.
            time.sleep(0.5)
        raise RuntimeError("MPS did not publish a server within 30 seconds")

    def stop_clients(self, clients):
        """Signal only process groups created by this diagnostic, including children."""
        for client in clients:
            try:
                os.killpg(client["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        limit = time.monotonic() + 3
        while any(c["process"].poll() is None for c in clients) and time.monotonic() < limit:
            time.sleep(0.05)
        for client in clients:
            try:
                os.killpg(client["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                client["process"].wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.summary["cleanup_error"] = "A child did not exit after SIGKILL; inspect recorded PIDs"

    def sample(self, directory, mps):
        result = self.command(["nvidia-smi", f"--query-gpu={SAMPLE_FIELDS}",
                               "--format=csv,noheader,nounits"], timeout=3)
        clients = self.mps_clients() if mps else set()
        with (directory / "gpu-samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": utc(), "fields": SAMPLE_FIELDS,
                                     "mps_client_pids": sorted(clients), **result}) + "\n")
        return clients

    def run_stage(self, spec, eligible, mace_steps):
        name, workload, mps, concurrency = spec
        directory = self.output / name
        directory.mkdir()
        stage = {"name": name, "workload": workload, "mps": mps,
                 "per_gpu": concurrency, "gpu_indices": [g["index"] for g in eligible],
                 "started_at": utc(), "clients": [], "status": "running",
                 "all_clients_overlapped": False, "all_mps_clients_observed": False}
        self.summary["stages"].append(stage)
        self.save()
        print(f"{utc()} {name}: {len(eligible) * concurrency} clients on {len(eligible)} GPUs", flush=True)
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < len(eligible) * concurrency:
            raise RuntimeError("Not enough allowed CPUs to bind every client to a distinct CPU")
        steps = 20 if workload == "lj" else mace_steps
        started = time.monotonic()
        deadline = min(self.deadline, started + self.args.stage_timeout)
        self.clients = []
        template = self.output / "inputs" / workload
        try:
            for gpu in eligible:
                for replica in range(concurrency):
                    workdir = directory / f"gpu-{gpu['index']}-client-{replica}"
                    shutil.copytree(template, workdir)
                    cpu = cpus[len(self.clients)]
                    env = dict(self.mps_env if mps else self.env,
                               CUDA_VISIBLE_DEVICES=gpu["uuid"])
                    argv = ["taskset", "-c", str(cpu), str(self.args.wrapper),
                            "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                            "newton", "on", "neigh", "half", "-i", "in.lammps"]
                    record = {"gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"],
                              "replica": replica, "cpu": cpu, "command": argv,
                              "started_at": utc(), "directory": str(workdir.relative_to(self.output))}
                    with (workdir / "stdout.txt").open("w") as out, (workdir / "stderr.txt").open("w") as err:
                        proc = subprocess.Popen(argv, cwd=workdir, env=env, stdout=out,
                                                stderr=err, start_new_session=True)
                    record["pid"] = proc.pid
                    stage["clients"].append(record)
                    self.clients.append({"process": proc, "record": record, "directory": workdir,
                                         "start": time.monotonic()})
            self.save()
            next_sample = 0
            next_heartbeat = started + 30
            observed_by_gpu = set()
            while True:
                now = time.monotonic()
                running = []
                for client in self.clients:
                    proc, record = client["process"], client["record"]
                    if proc.poll() is None:
                        running.append(client)
                    elif "exit_code" not in record:
                        record.update(exit_code=proc.returncode, finished_at=utc(),
                                      elapsed_seconds=round(now - client["start"], 3))
                if not running:
                    break
                if len(running) == len(self.clients):
                    stage["all_clients_overlapped"] = True
                if now >= deadline or self.stop_signal:
                    reason = "interrupted" if self.stop_signal else "timeout"
                    for client in running:
                        client["record"]["failure"] = reason
                    self.stop_clients(running)
                    for client in running:
                        client["record"].update(exit_code=client["process"].poll(), finished_at=utc(),
                                                elapsed_seconds=round(time.monotonic() - client["start"], 3))
                    break
                if now >= next_sample:
                    active = self.sample(directory, mps)
                    expected = {c["process"].pid for c in self.clients}
                    if mps and expected <= active:
                        stage["all_mps_clients_observed"] = True
                    for gpu in eligible:
                        gpu_pids = {c["process"].pid for c in self.clients
                                    if c["record"]["gpu_uuid"] == gpu["uuid"]}
                        if mps and gpu_pids <= active:
                            observed_by_gpu.add(gpu["uuid"])
                    next_sample = time.monotonic() + 5
                if now >= next_heartbeat:
                    print(f"{utc()} {name}: {len(running)} clients active, {now-started:.0f}s elapsed", flush=True)
                    next_heartbeat = now + 30
                    self.save()
                time.sleep(0.1)
            failed_gpus = set()
            for client in self.clients:
                record, workdir = client["record"], client["directory"]
                record.update(check_lammps(workdir, steps))
                text = "\n".join((workdir / filename).read_text(errors="replace")
                                 for filename in ("stderr.txt", "stdout.txt", "log.lammps")
                                 if (workdir / filename).is_file())
                category = failure_category(text)
                failure = record.get("failure") or category
                if not failure and record["exit_code"]:
                    failure = "process_error"
                if not failure and record.get("validation_error"):
                    failure = "invalid_output"
                if not failure and mps and record["gpu_uuid"] not in observed_by_gpu:
                    failure = "mps_overlap_not_observed"
                record["status"] = "failed" if failure else "passed"
                if failure:
                    record["failure"] = failure
                    failed_gpus.add(record["gpu_uuid"])
                write_json(workdir / "result.json", record)
            elapsed = time.monotonic() - started
            passed = sum(c["status"] == "passed" for c in stage["clients"])
            stage.update(finished_at=utc(), elapsed_seconds=round(elapsed, 3),
                         completed_clients=passed,
                         aggregate_md_steps_per_second=round(steps * passed / elapsed, 4),
                         status="passed" if not failed_gpus else "failed")
            # The four-GPU result also requires simultaneous MPS visibility of all clients.
            if mps and not stage["all_mps_clients_observed"] and not failed_gpus:
                stage.update(status="inconclusive", reason="All clients did not overlap in an MPS sample")
            mps_errors = self.new_mps_errors()
            if mps_errors:
                stage.update(status="failed", mps_log_errors=mps_errors)
                # A server-level fault cannot safely be attributed to one client.
                failed_gpus.update(g["uuid"] for g in eligible)
            self.archive_mps(directory / "mps-logs")
            print(f"{utc()} {name}: {stage['status']}; {passed}/{len(self.clients)} clients passed", flush=True)
            self.save()
            return failed_gpus
        finally:
            # Remove remaining descendants even when the launcher itself has exited.
            self.stop_clients(self.clients)
            self.clients = []

    def cleanup(self):
        self.stop_clients(self.clients)
        if self.mps_started:
            result = self.control("quit")
            self.summary["mps_shutdown"] = result
            if result["returncode"]:
                self.summary["cleanup_error"] = "MPS quit failed; local directories retained"
            self.mps_started = False
        self.archive_mps(self.output / "mps-logs")
        if self.mps_dir and not self.summary.get("cleanup_error"):
            shutil.rmtree(self.mps_dir)

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()
        print(f"Results: {self.output}", flush=True)
        try:
            inputs = self.output / "inputs"
            mace_steps = prepare_mace(self.args.input_dir, inputs / "mace", self.args.full)
            (inputs / "lj").mkdir()
            shutil.copy2(Path(__file__).with_name("in.lj"), inputs / "lj" / "in.lammps")
            self.summary.update(mace_steps=mace_steps, input_dir=str(self.args.input_dir),
                                wrapper=str(self.args.wrapper), stage_timeout=self.args.stage_timeout,
                                stage_plan=stage_specs(self.args.max_per_gpu, self.args.full))
            if self.args.prepare_only:
                self.summary["status"] = "prepared_only"
                return 0
            if not os.environ.get("PBS_JOBID"):
                raise RuntimeError("GPU tests require a fresh PBS compute allocation; use --prepare-only for a local check")
            hosts = set(Path(os.environ["PBS_NODEFILE"]).read_text().split())
            if len(hosts) != 1 or socket.gethostname().split(".")[0] != next(iter(hosts)).split(".")[0]:
                raise RuntimeError("Run on the compute node of a one-node allocation")
            for command in ("nvidia-smi", "nvidia-cuda-mps-control", "taskset"):
                if shutil.which(command) is None:
                    raise RuntimeError(f"Missing command: {command}")
            if not os.access(self.args.wrapper, os.X_OK):
                raise RuntimeError(f"LAMMPS wrapper is not executable: {self.args.wrapper}")
            processes = self.command(["ps", "-eo", "pid=,args="])
            write_json(self.output / "initial-processes.json", processes)
            self.gpus = self.snapshot("before")
            self.summary["gpu_inventory"] = self.gpus
            if processes["returncode"] or existing_mps_processes(processes["stdout"]):
                raise RuntimeError("Existing MPS processes prevent a clean no-MPS baseline; no existing daemon was stopped")
            write_json(self.output / "mps-version.json", self.command(["nvidia-cuda-mps-control", "-v"]))
            eligible = list(self.gpus)
            for spec in stage_specs(self.args.max_per_gpu, self.args.full):
                if not eligible or self.stop_signal or time.monotonic() >= self.deadline:
                    self.summary["stop_reason"] = "No eligible GPUs, interrupted, or 55-minute deadline reached"
                    break
                if spec[2] and not self.mps_started:
                    self.start_mps()
                failed = self.run_stage(spec, eligible, mace_steps)
                eligible = [g for g in eligible if g["uuid"] not in failed]
                self.snapshot(spec[0] + "-after")
            stages = self.summary["stages"]
            passed = len(stages) == len(self.summary["stage_plan"]) and all(s["status"] == "passed" for s in stages)
            # A partial-node pass never establishes support for 16 simulations/node.
            passed = passed and all(len(s["gpu_indices"]) == 4 for s in stages)
            self.summary["status"] = "passed" if passed else "failed_or_incomplete"
        except Exception as exc:
            self.summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(self.summary["error"], file=sys.stderr, flush=True)
        finally:
            try:
                self.cleanup()
            except Exception as exc:
                self.summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if self.summary.get("cleanup_error"):
                self.summary["status"] = "failed"
            if self.stop_signal:
                self.summary.update(status="interrupted", signal=self.stop_signal)
            self.summary["highest_verified_clients_per_gpu_on_all_four_gpus"] = max(
                (stage["per_gpu"] for stage in self.summary["stages"]
                 if stage["mps"] and stage["status"] == "passed" and len(stage["gpu_indices"]) == 4),
                default=0)
            self.summary["finished_at"] = utc()
            self.save()
            print(f"Results: {self.output}\nStatus: {self.summary['status']}", flush=True)
        return 0 if self.summary["status"] in {"passed", "prepared_only"} else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--wrapper", type=Path, default=ROOT / "bin/run-lammps-polaris.sh")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-per-gpu", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--full", action="store_true", help="Original minimization/MD lengths; compare no MPS with only the selected MPS concurrency")
    parser.add_argument("--stage-timeout", type=float, help="Seconds per stage (short: 600; full: 1800)")
    parser.add_argument("--prepare-only", action="store_true", help="Copy/validate inputs and write the test plan without running GPU commands")
    args = parser.parse_args(argv)
    args.stage_timeout = args.stage_timeout if args.stage_timeout is not None else (1800 if args.full else 600)
    if not math.isfinite(args.stage_timeout) or args.stage_timeout <= 0:
        parser.error("--stage-timeout must be finite and positive")
    args.input_dir = args.input_dir.resolve()
    args.wrapper = args.wrapper.resolve()
    if args.output is None:
        job = re.sub(r"[^\w.-]", "_", os.environ.get("PBS_JOBID", "local"))
        args.output = ROOT / "run" / f"lammps-diagnostic-{job}-{time.time_ns()}"
    diagnostic = Diagnostic(args)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: setattr(diagnostic, "stop_signal", signum))
    return diagnostic.run()


if __name__ == "__main__":
    sys.exit(main())
