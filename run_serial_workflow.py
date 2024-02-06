"""An example of the workflow which runs on a single node"""
import json
import logging
import hashlib
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from random import choice
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from openbabel import openbabel as ob

from mofa.assembly.assemble import assemble_mof
from mofa.generator import run_generator
from mofa.model import MOFRecord, NodeDescription, LigandTemplate
from mofa.scoring.geometry import MinimumDistance
from mofa.simulation.lammps import LAMMPSRunner
from mofa.utils.xyz import xyz_to_mol

RDLogger.DisableLog('rdApp.*')
ob.obErrorLog.SetOutputLevel(0)


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='MOF Settings', description='Options related to the MOF type being generated')
    group.add_argument('--node-path', required=True, help='Path to a node record')

    group = parser.add_argument_group(title='Generator Settings', description='Options related to how the generation is performed')
    group.add_argument('--ligand-templates', required=True, nargs='+',
                       help='Path to YAML files containing a description of the ligands to be created')
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

    # Load the ligand descriptions
    templates = {}
    for path in args.ligand_templates:
        template = LigandTemplate.from_yaml(path)
        templates[template.anchor_type] = template
    logger.info(f'Loaded {len(templates)} ligand templates: {", ".join(templates.keys())}')

    # Load a pretrained generator from disk and use it to create ligands
    generated_ligands = {}
    for template in templates.values():
        my_ligands = []
        for n_atoms in args.molecule_sizes:
            logger.info(f'Generating molecules with {n_atoms} atoms for {template.anchor_type} on {args.torch_device}')
            my_ligands.extend(run_generator(
                templates=[template],
                model=args.generator_path,
                n_atoms=n_atoms,
                n_samples=args.num_samples,
                device=args.torch_device
            ))
        generated_ligands[template.anchor_type] = my_ligands
        logger.info(f'Generated a total of {len(my_ligands)} ligands for {template.anchor_type}')

    # Initial quality checks and post-processing on the generated ligands
    valid_ligands = {}  # Ligands to be used during assembly
    all_ligands = []  # All ligands which were generated
    for anchor_type, new_ligands in generated_ligands.items():
        valid_ligands[anchor_type] = []
        if anchor_type == "COO":
            continue
        for ligand in new_ligands:
            # Store the ligand information for debugging purposes
            record = {"anchor_type": ligand.anchor_type, "xyz": ligand.xyz,
                      "anchor_atoms": ligand.anchor_atoms, "valid": False}

            # Parse each new ligand, determine whether it is a single molecule
            try:
                mol = xyz_to_mol(ligand.xyz)
            except (ValueError,):
                continue

            # Store the smiles string
            Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol)
            record['smiles'] = smiles

            if len(Chem.GetMolFrags(mol)) > 1:
                continue

            # If passes, save the SMILES string and store the molecules
            ligand.smiles = Chem.MolToSmiles(mol)
            valid_ligands[anchor_type].append(ligand)

            # Update the record
            record['valid'] = True
            all_ligands.append(record)

            # begin of swap cyano for COO
            coo_ligand = ligand.swap_cyano_with_COO()
            coo_record = {"anchor_type": coo_ligand.anchor_type, "xyz": coo_ligand.xyz,
                      "anchor_atoms": coo_ligand.anchor_atoms, "valid": False}

            # Parse each new ligand, determine whether it is a single molecule
            try:
                mol = xyz_to_mol(coo_ligand.xyz)
            except (ValueError,):
                continue

            # Store the smiles string
            Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol)
            coo_record['smiles'] = smiles

            if len(Chem.GetMolFrags(mol)) > 1:
                continue

            # If passes, save the SMILES string and store the molecules
            coo_ligand.smiles = Chem.MolToSmiles(mol)
            valid_ligands["COO"].append(coo_ligand)

            # Update the record
            coo_record['valid'] = True
            all_ligands.append(coo_record)
            # end of swap cyano for COO

        logger.info(f'{len(valid_ligands[anchor_type])} of {len(new_ligands)} for {anchor_type} pass quality checks')
        logger.info(f'{len(valid_ligands["COO"])} of {len(new_ligands)} for COO pass quality checks')

    # Save the ligands
    pd.DataFrame(all_ligands).to_csv(run_dir / 'all-ligands.csv', index=False)

    # Combine them with the template MOF to create new MOFs
    new_mofs = []
    attempts = 0
    while len(new_mofs) < args.num_to_assemble and attempts < args.max_assemble_attempts * args.num_to_assemble:
        attempts += 1
        # TODO (wardlt): Do not hard-code requirements
        requirements = {'COO': 2, 'cyano': 1}
        ligand_choices = {}
        for anchor_type, count in requirements.items():
            ligand_choices[anchor_type] = [choice(valid_ligands[anchor_type])] * count

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
