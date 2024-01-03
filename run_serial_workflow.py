"""An example of the workflow which runs on a single node"""
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from random import sample

from rdkit import Chem
from rdkit import RDLogger

from mofa.assembly.assemble import assemble_mof
from mofa.assembly.preprocess_linkers import clean_linker
from mofa.assembly.validate import validate_xyz
from mofa.generator import run_generator
from mofa.model import MOFRecord, NodeDescription
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

    group = parser.add_argument_group(title='Compute Settings', description='Compute environment configuration')
    group.add_argument('--torch-device', default='cpu', help='Device on which to run torch operations')

    args = parser.parse_args()

    # TODO: Make a run directory

    # Load the example MOF
    # TODO (wardlt): Use Pydantic
    node_record = NodeDescription(**json.loads(Path(args.node_path).read_text()))

    # Turn on logging
    logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Load a pretrained generator from disk and use it to create ligands
    logger.info(f'Running generation on {args.torch_device}')
    generated_ligand_xyzs = []
    for n_atoms in args.molecule_sizes:
        generated_ligand_xyzs.extend(run_generator(
            model=args.generator_path,
            n_atoms=n_atoms,
            input_path=args.fragment_path,
            n_samples=args.num_samples,
            device=args.torch_device
        ))
    logger.info(f'Generated {len(generated_ligand_xyzs)} ligands')

    # Initial quality checks and post-processing on the generated ligands
    ligands = []
    for generated_xyz in generated_ligand_xyzs:
        # Ensure that it validates and we can clean it
        try:
            smiles = validate_xyz(generated_xyz)
            clean_linker(Chem.MolFromSmiles(smiles))
        except (ValueError, KeyError, IndexError):
            continue
    logger.info(f'Screened generated ligands. {len(ligands)} pass quality checks')
    if len(ligands) < 3:
        raise ValueError('Too few passed quality checks')

    # Combine them with the template MOF to create new MOFs
    new_mofs = []
    while len(new_mofs) < args.num_to_assemble:
        # TODO (do not hard-code assembly options)
        ligand_choices = sample(ligands, 3)
        try:
            new_mof = assemble_mof(
                nodes=[node_record],
                ligands=ligand_choices,
                topology='pcu'
            )
        except (ValueError, KeyError, IndexError):
            continue
        new_mofs.append(new_mof)
    logger.info(f'Generated {len(new_mofs)} new MOFs')

    # Score the MOFs
    scorer = MinimumDistance()  # TODO (wardlt): Add or replace with a CGCNN that predicts absorption
    scores = [scorer.score_mof(new_mof) for new_mof in new_mofs]
    logger.info(f'Scored all {len(new_mofs)} MOFs')

    # Run LAMMPS on the top MOF
    ranked_mofs: list[tuple[float, MOFRecord]] = sorted(zip(scores, new_mofs))
    LAMMPSRunner('lmp_serial').run_molecular_dynamics(ranked_mofs[-1][1], 100, 1)
    logger.info('Ran LAMMPS for all MOFs')
