"""An example of the workflow which runs on a single node"""
import logging
import sys
from argparse import ArgumentParser

import torch

from mofa.generator import run_generator
from mofa.model import MOFRecord
from mofa.scoring.geometry import MinimumDistance
from mofa.simulation.lammps import LAMMPSRunner

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='MOF Settings', description='Options related to the MOF type being generatored')
    group.add_argument('--mof-template', default=None, help='Path to a MOF we are going to be altering')

    group = parser.add_argument_group(title='Generator Settings', description='Options related to how the generation is performed')
    group.add_argument('--generator-path', default=None, help='Path to the PyTorch files describing model architecture and weights')
    group.add_argument('--molecule-sizes', nargs='+', dtype=int, default=(10, 11, 12), help='Sizes of molecules we should generate')
    group.add_argument('--num-samples', dtype=int, default=16, help='Number of molecules to generate at each size')

    args = parser.parse_args()

    # TODO: Make a run directory

    # Turn on logging
    logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Load a pretrained generator from disk and use it to create ligands
    template_mof = MOFRecord.from_file(cif_path=args.mof_template)
    model = torch.load(args.generator_path)
    generated_ligand_xyzs = run_generator(
        model,
        fragment_template=template_mof.ligands[0],
        molecule_sizes=args.molecule_sizes,
        num_samples=args.num_samples
    )
    logger.info(f'Generated {len(generated_ligand_xyzs)} ligands')

    # Initial quality checks and post-processing on the generated ligands
    validated_ligand_xyzs = []
    for generated_xyz in generated_ligand_xyzs:
        if False:  # TODO (wardlt): Add checks for detecting fragmented molecules, valency checks, ...
            pass
        validated_ligand_xyzs.append(add_hydrogens_to_ligand(generated_xyz))  # TODO (wardlt): Add a function which adds H's to the XYZ file
    logger.info(f'Screened generated ligands. {validated_ligand_xyzs} pass quality checks')

    # Combine them with the template MOF to create new MOFs
    new_mofs = []
    for new_ligand in validated_ligand_xyzs:
        new_mof = template_mof.replace_ligand(new_ligand)
        new_mofs.append(new_mof)
    logger.info(f'Generated {len(new_mofs)} new MOFs')

    # Score the MOFs
    scorer = MinimumDistance()  # TODO (wardlt): Add or replace with a CGCNN that predicts absorption
    scores = [scorer.score_mof(new_mof) for new_mof in new_mofs]
    logger.info(f'Scored all {len(new_mofs)} MOFs')

    # Run LAMMPS on the top MOF
    ranked_mofs: list[tuple[float, MOFRecord]] = sorted(zip(scores, new_mofs))
    LAMMPSRunner('lmp_serial').run_molecular_dynamics(ranked_mofs[-1][1], 100, 1)
