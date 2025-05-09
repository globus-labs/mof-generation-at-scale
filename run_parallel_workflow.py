"""An example of the workflow which runs all aspects of MOF generation in parallel"""
from functools import partial, update_wrapper
from subprocess import Popen
from argparse import ArgumentParser
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

from mofa.db import initialize_database
from mofa.assembly.assemble import assemble_many
from mofa.assembly.validate import process_ligands
from mofa.finetune.difflinker import DiffLinkerCurriculum
from mofa.generator import run_generator, train_generator
from mofa.model import NodeDescription, LigandTemplate
from mofa.selection.dft import DFTSelector
from mofa.selection.md import MDSelector
from mofa.simulation.dft import compute_partial_charges
from mofa.simulation.mace import MACERunner
from mofa.steering import GeneratorConfig, TrainingConfig, MOFAThinker, SimulationConfig
from mofa.hpc.colmena import DiffLinkerInference
from mofa.hpc.config import LocalConfig
from mofa.utils.config import load_variable

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
    group.add_argument('--maximum-train-size', type=int, default=4096, help='Maximum number of MOFs to use for retraining')
    group.add_argument('--num-epochs', type=int, default=128, help='Number of training epochs')
    group.add_argument('--best-fraction', type=float, default=0.5, help='What percentile of MOFs to include in training')
    group.add_argument('--maximum-strain', type=float, default=0.5, help='Maximum strain allowed MOF used in training set')

    group = parser.add_argument_group(title='Assembly Settings', description='Options related to MOF assembly')
    group.add_argument('--max-assemble-attempts', default=100,
                       help='Maximum number of attempts to create a MOF')
    group.add_argument('--minimum-ligand-pool', type=int, default=4, help='Minimum number of ligands before MOF assembly')

    group = parser.add_argument_group(title='Simulation Settings Settings', description='Options related to property calculations')
    group.add_argument('--mace-model-path', required=True, help='Path to the MACE model, compiled for LAMPS')
    group.add_argument('--md-timesteps', default=3000, help='Number of timesteps per run of the MACE MD simulation', type=int)
    group.add_argument('--md-timesteps-max', default=20000, help='Maximum number of timesteps to run for any MD simulation', type=int)
    group.add_argument('--md-snapshots-freq', default=1000, help='How frequently to write timesteps', type=int)
    group.add_argument('--retain-lammps', action='store_true', help='Keep LAMMPS output files after it finishes')
    group.add_argument('--dft-opt-steps', default=8, help='Maximum number of DFT optimization steps', type=int)
    group.add_argument('--raspa-timesteps', default=100000, help='Number of timesteps for GCMC computation', type=int)

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--lammps-on-ramdisk', action='store_true', help='Write LAMMPS outputs to a RAM Disk')
    group.add_argument('--compute-config', default='local', help='Configuration for the HPC system. Use either "local" for single node, '
                                                                 'or provide the path to a config file containing the config')
    group.add_argument('--ai-fraction', default=0.1, type=float, help='Fraction of workers devoted to AI tasks')
    group.add_argument('--dft-fraction', default=0.1, type=float, help='Fraction of workers devoted to DFT tasks')
    group.add_argument('--redis-host', default=node(), help='Host for the Redis server')
    group.add_argument('--proxy-threshold', default=10000, type=int, help='Size threshold to use proxystore for data (bytes)')

    group = parser.add_argument_group(title='Selector Settings', description='Control how simulation tasks are selected')
    group.add_argument('--md-new-fraction', default=0.5, help='How frequently to start MD on a new MOF')

    args = parser.parse_args()

    # Load the example MOF
    # TODO (wardlt): Use Pydantic for JSON I/O
    node_template = NodeDescription(**json.loads(Path(args.node_path).read_text()))

    # Make the run directory
    run_params = args.__dict__.copy()
    start_time = datetime.now()
    config_name = Path(args.compute_config).with_suffix('').name
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    run_dir = Path('run') / f'parallel-{config_name}-{start_time.strftime("%d%b%y%H%M%S")}-{params_hash}'
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
    if args.compute_config == 'local':
        hpc_config = LocalConfig()
    else:
        hpc_config = load_variable(args.compute_config, 'hpc_config')
    hpc_config.run_dir = run_dir
    hpc_config.ai_fraction = args.ai_fraction
    hpc_config.dft_fraction = args.dft_fraction

    # Make the Parsl configuration
    config = hpc_config.make_parsl_config()
    with (run_dir / 'compute-config.json').open('w') as fp:
        print(hpc_config.model_dump_json(indent=2), file=fp)

    # Launch MongoDB as a subprocess
    mongo_dir = run_dir / 'db'
    mongo_dir.mkdir(parents=True)
    mongo_proc = Popen(
        f'mongod --wiredTigerCacheSizeGB 4 --dbpath {mongo_dir.absolute()} --logpath {(run_dir / "mongo.log").absolute()}'.split(),
        stderr=(run_dir / 'mongo.err').open('w')
    )
    mongo_client = MongoClient()
    mongo_coll = initialize_database(mongo_client)

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
        num_epochs=args.num_epochs,
        curriculum=DiffLinkerCurriculum(
            max_size=args.maximum_train_size,
            collection=mongo_coll,
            min_strain_counts=args.retrain_freq,
            max_strain=args.maximum_strain,
            min_gas_counts=args.retrain_freq,
        )
    )
    train_func = partial(train_generator,
                         config_path=args.generator_config_path,
                         num_epochs=trainer.num_epochs,
                         device=hpc_config.torch_device,
                         node_list=hpc_config.training_nodes)
    update_wrapper(train_func, train_generator)

    # Make the LAMMPS function
    lmp_runner = MACERunner(lammps_cmd=hpc_config.lammps_cmd,
                            model_path=Path(args.mace_model_path).absolute(),
                            run_dir=Path('/dev/shm/lmp_run' if args.lammps_on_ramdisk else run_dir / 'lmp_run'),
                            delete_finished=args.lammps_on_ramdisk)
    md_fun = partial(lmp_runner.run_molecular_dynamics, report_frequency=args.md_snapshots_freq)
    update_wrapper(md_fun, lmp_runner.run_molecular_dynamics)
    sim_config = SimulationConfig(md_length=args.md_timesteps, md_report=args.md_snapshots_freq)

    md_opt_fun = partial(lmp_runner.run_optimization, steps=1024, fmax=0.5)
    md_opt_fun.__name__ = 'run_optimization_ff'

    md_selector = MDSelector(
        collection=mongo_coll,
        new_fraction=args.md_new_fraction,
        max_strain=args.maximum_strain,
        maximum_steps=args.md_timesteps_max
    )

    # Make the CP2K function
    dft_runner = hpc_config.make_dft_runner()
    cp2k_fun = partial(dft_runner.run_optimization, steps=args.dft_opt_steps)  # Optimizes starting from assembled structure
    update_wrapper(cp2k_fun, dft_runner.run_optimization)

    dft_selector = DFTSelector(
        collection=mongo_coll,
        max_strain=args.maximum_strain
    )

    # Make the RASPA function
    raspa_runner = hpc_config.make_raspa_runner()
    raspa_fun = partial(raspa_runner.run_gcmc,
                        adsorbate='CO2',
                        temperature=298,
                        pressure=1e4,
                        cycles=args.raspa_timesteps)
    update_wrapper(raspa_fun, raspa_runner.run_gcmc)

    # Make the thinker
    thinker = MOFAThinker(queues,
                          collection=mongo_coll,  # Connect to a local service
                          hpc_config=hpc_config,
                          generator_config=generator,
                          trainer_config=trainer,
                          simulation_config=sim_config,
                          simulation_budget=args.simulation_budget,
                          dft_selector=dft_selector,
                          md_selector=md_selector,
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
            (md_opt_fun, {'executors': hpc_config.lammps_executors}),
            (cp2k_fun, {'executors': hpc_config.dft_executors}),
            (compute_partial_charges, {'executors': hpc_config.helper_executors}),
            (process_ligands, {'executors': hpc_config.helper_executors}),
            (raspa_fun, {'executors': hpc_config.raspa_executors}),
            (assemble_many, {'executors': hpc_config.helper_executors})
        ],
        queues=queues,
        config=config
    )

    # Launch the utilization logging
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True)
    util_proc = hpc_config.launch_monitor_process()
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
