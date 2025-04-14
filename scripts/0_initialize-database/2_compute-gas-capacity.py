from functools import partial

from mofa.simulation.graspa_sycl import GRASPASyclRunner
from mofa.utils.conversions import read_from_string
from concurrent.futures import as_completed
from parsl import Config, HighThroughputExecutor, python_app, load
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

if __name__ == "__main__":

    # Load what we've done already
    output_path = Path('capacity.jsonl')
    if output_path.exists():
        done = list(map(tuple, pd.read_json('capacity.jsonl', lines=True)[['mof', 'opt_steps']].values.tolist()))
    else:
        done = set()

    # Make a parsl executor
    config = Config(executors=[HighThroughputExecutor(max_workers_per_node=6, cpu_affinity='list:0:1:2:3:4:5')])
    with load(config):

        # Run what we have not
        runner = GRASPASyclRunner(
            graspa_command=('/home/lward/Software/gRASPA/graspa-sycl/bin/sycl.out',),
            run_dir=Path('gcmc')
        )
        fun = partial(runner.run_gcmc,
                      adsorbate='CO2',
                      temperature=298,
                      pressure=1e4,
                      n_cycle=100000)
        run_app = python_app(fun)
        futures = []
        for _, row in pd.read_json('charges.jsonl', lines=True).iterrows():
            if (row['mof'], row['steps']) in done:
                continue
            if row['charges'] is None:
                print(f'No charges for {row["mof"]}')
                continue

            # Load the structure and attach charges
            future = run_app(row['mof'], Path('cp2k-run-2') / f"{row['mof']}-optimize-pbe")
            future.row = row
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                mean, std = future.result()
                with output_path.open('a') as fp:
                    print(json.dumps({
                        'mof': future.row['mof'],
                        'opt_steps': future.row['steps'],
                        'mean': mean,
                        'std': std
                    }), file=fp)
            except:
                print(f'{future.row["mof"]} failed')
                continue
