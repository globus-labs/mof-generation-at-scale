"""An example of the workflow which runs on a single node"""
import json
import logging
import hashlib
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from random import sample
from pathlib import Path
from io import StringIO

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from mofa.assembly.assemble import assemble_mof
from mofa.assembly.preprocess_linkers import clean_linker
from mofa.assembly.validate import validate_xyz
from mofa.generator import run_generator
from mofa.model import MOFRecord, NodeDescription, LigandDescription
from mofa.scoring.geometry import MinimumDistance
from mofa.simulation.lammps import LAMMPSRunner

RDLogger.DisableLog('rdApp.*')

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='MOF Settings', description='Options related to the MOF type being generated')
    group.add_argument('--node-path', required=True, help='Path to a node record')

    group = parser.add_argument_group(title='Generator Settings', description='Options related to how the generation is performed')
    group.add_argument('--fragment-path', required=True,
                       help='Path to an SDF file containing the fragments')
    group.add_argument('--generator-path', required=True,
                       help='Path to the PyTorch files describing model architecture and weights')
    group.add_argument('--molecule-sizes', nargs='+', type=int, default=(10, 11, 12), help='Sizes of molecules we should generate')
    group.add_argument('--num-samples', type=int, default=16, help='Number of molecules to generate at each size')

    group = parser.add_argument_group(title='Assembly Settings', description='Options related to MOF assembly')
    group.add_argument('--num-to-assemble', default=4, type=int, help='Number of MOFs to create from generated ligands')
    group.add_argument('--max-assemble-attempts', default=100,
                       help='Maximum number of attempts to create a MOF')

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--torch-device', default='cpu', help='Device on which to run torch operations')

    args = parser.parse_args()

    # Load the example MOF
    # TODO (wardlt): Use Pydantic for JSON I/O
    node_record = NodeDescription(**json.loads(Path(args.node_path).read_text()))

    # Make the run directory
    run_params = args.__dict__.copy()
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    run_dir = Path('run') / f'single-{start_time.strftime("%d%b%y%H%M%S")}-{params_hash}'
    run_dir.mkdir(parents=True)

    # Turn on logging
    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'run.log')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(f'Running job in {run_dir}')

    # Save the run parameters to disk
    (run_dir / 'params.json').write_text(json.dumps(run_params))

    # Load a pretrained generator from disk and use it to create ligands
    generated_ligand_atoms = []
    for n_atoms in args.molecule_sizes:
        logger.info(f'Generating molecules with {n_atoms} atoms on {args.torch_device}')
        generated_ligand_atoms.extend(run_generator(
            model=args.generator_path,
            n_atoms=n_atoms,
            input_path=args.fragment_path,
            n_samples=args.num_samples,
            device=args.torch_device
        ))
    logger.info(f'Generated {len(generated_ligand_atoms)} ligands')

    # Initial quality checks and post-processing on the generated ligands
    valid_ligands = []
    all_ligands = []
    for generated_atoms in generated_ligand_atoms:
        # Initialize a record for this ligand and put it in the output dictionary
        fp = StringIO()  # TODO (wardlt): Make a utility operation
        generated_atoms.write(fp, format='xyz')
        xyz = fp.getvalue()

        record = {'xyz': xyz, 'valid': False}
        all_ligands.append(record)  # We will still alter the record

        # Ensure that it validates and we can clean it
        try:
            smiles = validate_xyz(generated_atoms)
            record['smiles'] = smiles
            clean_linker(Chem.MolFromSmiles(smiles))
        except (ValueError, KeyError, IndexError):
            continue

        # Store it as a LigandRecord
        record['valid'] = True
        valid_ligands.append(LigandDescription(smiles=smiles))
    logger.info(f'Screened generated ligands. {len(valid_ligands)} pass quality checks')

    # Save the ligands
    pd.DataFrame(all_ligands).to_csv(run_dir / 'all-ligands.csv', index=False)

    # Check if we can proceed
    if len(valid_ligands) < 3:
        raise ValueError('Too few passed quality checks')

    # Combine them with the template MOF to create new MOFs
    new_mofs = []
    attempts = 0
    while len(new_mofs) < args.num_to_assemble and attempts < args.max_assemble_attempts * args.num_to_assemble:
        # TODO (wardlt): Do not hard-code generation options
        attempts += 1
        ligand_choices = sample(valid_ligands, 3)
        try:
            new_mof = assemble_mof(
                nodes=[node_record],
                ligands=ligand_choices,
                topology='pcu'
            )
        except (ValueError, KeyError, IndexError) as e:
            continue
        new_mofs.append(new_mof)
    logger.info(f'Generated {len(new_mofs)} new MOFs after {attempts} attempts')

    if len(new_mofs) == 0:
        raise ValueError('Failed to create any MOFs')

    # Score the MOFs
    scorer = MinimumDistance()  # TODO (wardlt): Add or replace with a CGCNN that predicts absorption
    scores = [scorer.score_mof(new_mof) for new_mof in new_mofs]
    logger.info(f'Scored all {len(new_mofs)} MOFs')

    # Run LAMMPS on the top MOF
    ranked_mofs: list[tuple[float, MOFRecord]] = sorted(zip(scores, new_mofs))
    lmp_runner = LAMMPSRunner('lmp_serial')
    for _, mof in ranked_mofs:
        lmp_runner.run_molecular_dynamics(ranked_mofs[-1][1], 100, 1)
    logger.info('Ran LAMMPS for all MOFs')

    # Save the completed MOFs to disk
    with (run_dir / 'completed-mofs.json').open('w') as fp:
        for _, mof in ranked_mofs:
            print(json.dumps(asdict(mof)), file=fp)
    logger.info('Saved everything do disk. Done!')
