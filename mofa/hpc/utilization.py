"""Utilization tracking"""
import json
import platform
import shutil
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from subprocess import Popen
from time import sleep
from typing import NoReturn

import psutil
import pynvml
from gpustat import GPUStatCollection


def get_utilization() -> dict:
    """Get the system utilization"""

    # Get the CPU and memory utilization
    output = {'time': datetime.utcnow().isoformat(),
              'cpu_use': psutil.cpu_percent(percpu=True),
              'memory_use': psutil.virtual_memory()._asdict(),
              'network': {}}

    # Network utilization
    for nic, stats in psutil.net_io_counters(pernic=True).items():
        output['network'][nic] = stats._asdict()

    # Processes
    output['procs'] = []
    for proc in psutil.process_iter(['username', 'pid', 'name', 'memory_percent', 'cpu_percent', 'io_counters']):
        info = proc.info
        if info.pop('username') != "root":
            output['procs'].append(info)

    # Disk-utilization
    output['disk'] = {}
    for disk, stats in psutil.disk_io_counters(perdisk=True).items():
        if not disk.startswith('loop'):
            output['disk'][disk] = stats._asdict()

    # Temperatures
    output['temperatures'] = {}
    for k, temp_lst in psutil.sensors_temperatures().items():
        temp_lst = [v._asdict() for v in temp_lst]
        output['temperatures'][k] = temp_lst

    # GPU Utilization
    try:
        gpu_util = GPUStatCollection.new_query()
        output['gpu_use'] = gpu_util.jsonify()['gpus']
    except pynvml.NVMLError:
        pass

    return output


def utilization_cli() -> NoReturn:
    """Log the utilization to disk"""

    parser = ArgumentParser()
    parser.add_argument('--frequency', default=30, type=float, help='How often to log utilization. Units: s')
    parser.add_argument('log_path', help='Name of the log file')
    args = parser.parse_args()

    log_path = Path(args.log_path)

    # Launch `xpu-smi` as a subprocess if available
    xpu_smi = shutil.which('xpu-smi')
    if xpu_smi is not None:
        Popen(
            [xpu_smi, 'dump', '-d', '-1', '-m', '0,1,17,18', '-i', str(int(args.frequency))],
            stdout=(log_path / f'{platform.node()}-xpu.csv').open('w')  # Leave open. Will close as Python exits
        )

    # Make my log name
    log_name = log_path / (platform.node() + ".log")
    with open(log_name, 'wt') as fp:
        get_utilization()  # First one is trash (see PSUtil docs: https://psutil.readthedocs.io/en/latest/#psutil.cpu_times_percent)
        while True:
            try:
                utilization = get_utilization()
            except psutil.NoSuchProcess:  # Happens if a process dies while assessing GPU performance
                continue
            print(json.dumps(utilization), file=fp, flush=True)
            sleep(args.frequency)
