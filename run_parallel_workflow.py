"""An example of the workflow which runs all aspects of MOF generation in parallel"""
from functools import partial, update_wrapper
from subprocess import Popen
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from platform import node
from pathlib import Path
import logging
import hashlib
import json
import sys

from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, register_store
from pymongo import MongoClient
from rdkit import RDLogger
from openbabel import openbabel as ob
from more_itertools import batched, make_decorator
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue.redis import RedisQueues

from mofa.assembly.assemble import assemble_many
from mofa.assembly.validate import process_ligands
from mofa.generator import run_generator, train_generator
from mofa.model import NodeDescription, LigandTemplate
from mofa.simulation.cp2k import CP2KRunner, compute_partial_charges
from mofa.simulation.lammps import LAMMPSRunner
from mofa.simulation.raspa import RASPARunner
from mofa.steering import GeneratorConfig, TrainingConfig, MOFAThinker, SimulationConfig
from mofa.hpc.colmena import DiffLinkerInference
from mofa.hpc.config import configs as hpc_configs

RDLogger.DisableLog('rdApp.*')
ob.obErrorLog.SetOutputLevel(0)

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--simulation-budget', type=int, help='Number of simulations to submit before exiting')

    group = parser.add_argument_group(title='MOF Settings', description='Options related to the MOF type being generated')
    group.add_argument('--node-path', required=True, help='Path to a node record')

    group = parser.add_argument_group(title='Generator Settings', description='Options related to how the generation is performed')
    group.add_argument('--ligand-templates', required=True, nargs='+',
                       help='Path to YAML files containing a description of the ligands to be created')
    group.add_argument('--generator-path', required=True,
                       help='Path to the PyTorch files describing model architecture and weights')
    group.add_argument('--molecule-sizes', nargs='+', type=int, default=list(range(6, 21)),
                       help='Sizes of molecules we should generate')
    group.add_argument('--num-samples', type=int, default=16, help='Number of molecules to generate at each size')
    group.add_argument('--gen-batch-size', type=int, default=16, help='Number of ligands to stream per batch')

    group = parser.add_argument_group('Retraining Settings', description='How often to retain, what to train on, etc')
    group.add_argument('--generator-config-path', required=True, help='Path to the generator training configuration')
    group.add_argument('--retrain-freq', type=int, default=8, help='Trigger retraining after these many successful computations')
    group.add_argument('--maximum-train-size', type=int, default=256, help='Maximum number of MOFs to use for retraining')
    group.add_argument('--num-epochs', type=int, default=128, help='Number of training epochs')
    group.add_argument('--best-fraction', type=float, default=0.5, help='What percentile of MOFs to include in training')
    group.add_argument('--maximum-strain', type=float, default=0.5, help='Maximum strain allowed MOF used in training set')

    group = parser.add_argument_group(title='Assembly Settings', description='Options related to MOF assembly')
    group.add_argument('--max-assemble-attempts', default=100,
                       help='Maximum number of attempts to create a MOF')
    group.add_argument('--minimum-ligand-pool', type=int, default=4, help='Minimum number of ligands before MOF assembly')

    group = parser.add_argument_group(title='Simulation Settings Settings', description='Options related to property calculations')
    group.add_argument('--md-timesteps', default=100000, help='Number of timesteps for the UFF MD simulation', type=int)
    group.add_argument('--md-snapshots', default=100, help='Maximum number of snapshots during MD simulation', type=int)
    group.add_argument('--retain-lammps', action='store_true', help='Keep LAMMPS output files after it finishes')
    group.add_argument('--dft-opt-steps', default=8, help='Maximum number of DFT optimization steps', type=int)
    group.add_argument('--raspa-timesteps', default=100000, help='Number of timesteps for GCMC computation', type=int)

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--lammps-on-ramdisk', action='store_true', help='Write LAMMPS outputs to a RAM Disk')
    group.add_argument('--compute-config', default='local', help='Configuration for the HPC system')
    group.add_argument('--ai-fraction', default=0.1, type=float, help='Fraction of workers devoted to AI tasks')
    group.add_argument('--dft-fraction', default=0.1, type=float, help='Fraction of workers devoted to DFT tasks')
    group.add_argument('--redis-host', default=node(), help='Host for the Redis server')
    group.add_argument('--proxy-threshold', default=10000, type=int, help='Size threshold to use proxystore for data (bytes)')

    args = parser.parse_args()

    # Load the example MOF
    # TODO (wardlt): Use Pydantic for JSON I/O
    node_template = NodeDescription(**json.loads(Path(args.node_path).read_text()))

    # Make the run directory
    run_params = args.__dict__.copy()
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    run_dir = Path('run') / f'parallel-{args.compute_config}-{start_time.strftime("%d%b%y%H%M%S")}-{params_hash}'
    run_dir.mkdir(parents=True)

    # Open a proxystore with Redis
    store = Store(name='redis', connector=RedisConnector(hostname=args.redis_host, port=6379), metrics=True)
    register_store(store)

    # Configure to a use Redis queue, which allows streaming results form other nodes
    queues = RedisQueues(
        hostname=args.redis_host,
        topics=['generation', 'lammps', 'cp2k', 'training', 'assembly'],
        proxystore_name='redis',
        proxystore_threshold=args.proxy_threshold
    )

    # Load the ligand descriptions
    templates = []
    for path in args.ligand_templates:
        template = LigandTemplate.from_yaml(path)
        templates.append(template)

    # Load the HPC configuration
    hpc_config = hpc_configs[args.compute_config]()
    hpc_config.ai_fraction = args.ai_fraction
    hpc_config.dft_fraction = args.dft_fraction

    # Make the Parsl configuration
    config = hpc_config.make_parsl_config(run_dir)
    with (run_dir / 'compute-config.json').open('w') as fp:
        json.dump(asdict(hpc_config), fp)

    # Make the generator settings and the function
    generator = GeneratorConfig(
        generator_path=args.generator_path,
        atom_counts=args.molecule_sizes,
        templates=templates,
        min_ligand_candidates=args.minimum_ligand_pool
    )
    gen_func = partial(run_generator, n_samples=args.num_samples, device=hpc_config.torch_device)
    gen_func = make_decorator(batched)(args.gen_batch_size)(gen_func)  # Wraps gen_func in a decorator in one line
    update_wrapper(gen_func, run_generator)
    gen_method = DiffLinkerInference(
        function=gen_func,
        name='run_generator',
        store_return_value=True,
        streaming_queue=queues,
        store=store
    )

    # Make the training function
    trainer = TrainingConfig(
        maximum_train_size=min(args.maximum_train_size, 2048),
        num_epochs=args.num_epochs,
        minimum_train_size=args.retrain_freq,
        best_fraction=args.best_fraction,
        maximum_strain=args.maximum_strain
    )
    train_func = partial(train_generator, config_path=args.generator_config_path,
                         num_epochs=trainer.num_epochs, device=hpc_config.torch_device)
    update_wrapper(train_func, train_generator)

    # Make the LAMMPS function
    lmp_runner = LAMMPSRunner(hpc_config.lammps_cmd,
                              lmp_sims_root_path='/dev/shm/lmp_run' if args.lammps_on_ramdisk else str(run_dir / 'lmp_run'),
                              lammps_environ=hpc_config.lammps_env,
                              delete_finished=not args.retain_lammps)
    md_fun = partial(lmp_runner.run_molecular_dynamics, report_frequency=max(1, args.md_timesteps / args.md_snapshots))
    update_wrapper(md_fun, lmp_runner.run_molecular_dynamics)
    sim_config = SimulationConfig(md_length=(args.md_timesteps,))

    # Make the CP2K function
    cp2k_runner = CP2KRunner(
        cp2k_invocation=hpc_config.cp2k_cmd,
        run_dir=run_dir / 'cp2k-runs'
    )
    cp2k_fun = partial(cp2k_runner.run_optimization, steps=args.dft_opt_steps)  # Optimizes starting from assembled structure
    update_wrapper(cp2k_fun, cp2k_runner.run_optimization)

    # Make the RASPA function
    raspa_runner = RASPARunner(
        raspa_sims_root_path=run_dir / 'raspa-runs'
    )
    raspa_fun = partial(raspa_runner.run_GCMC_single, timesteps=args.raspa_timesteps)
    update_wrapper(raspa_fun, raspa_runner.run_GCMC_single)

    # Launch MongoDB as a subprocess
    mongo_dir = run_dir / 'db'
    mongo_dir.mkdir(parents=True)
    mongo_proc = Popen(
        f'mongod --wiredTigerCacheSizeGB 4 --dbpath {mongo_dir.absolute()} --logpath {(run_dir / "mongo.log").absolute()}'.split(),
        stderr=(run_dir / 'mongo.err').open('w')
    )

    # Make the thinker
    thinker = MOFAThinker(queues,
                          mongo_client=MongoClient(),  # Connect to a local service
                          hpc_config=hpc_config,
                          generator_config=generator,
                          trainer_config=trainer,
                          simulation_budget=args.simulation_budget,
                          node_template=node_template,
                          out_dir=run_dir)

    # Turn on logging
    my_logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'run.log')]
    for logger in [my_logger, thinker.logger]:
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    my_logger.info(f'Running job in {run_dir} on {hpc_config.num_workers} workers')

    # Save the run parameters to disk
    (run_dir / 'params.json').write_text(json.dumps(run_params))

    # Launch the thinker and task server
    doer = ParslTaskServer(
        methods=[
            (gen_method, {'executors': hpc_config.inference_executors}),
            (train_func, {'executors': hpc_config.train_executors}),
            (md_fun, {'executors': hpc_config.lammps_executors}),
            (cp2k_fun, {'executors': hpc_config.cp2k_executors}),
            (compute_partial_charges, {'executors': hpc_config.helper_executors}),
            (process_ligands, {'executors': hpc_config.helper_executors}),
            (raspa_fun, {'executors': hpc_config.helper_executors}),
            (assemble_many, {'executors': hpc_config.helper_executors})
        ],
        queues=queues,
        config=config
    )

    # Launch the utilization logging
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True)
    util_proc = hpc_config.launch_monitor_process(log_dir.absolute())
    if util_proc.poll() is not None:
        raise ValueError('Monitor process failed to run!')
    my_logger.info(f'Launched monitoring process. pid={util_proc.pid}')

    try:
        doer.start()
        my_logger.info(f'Running parsl. pid={doer.pid}')

        with thinker:  # Opens the output files
            thinker.run()
    finally:
        queues.send_kill_signal()

        # Kill the services launched during workflow
        util_proc.terminate()
        mongo_proc.terminate()
        mongo_proc.poll()

        # Close the proxy store
        store.close()
