from pathlib import Path
from glob import glob
from typing import Sequence
import os
import shutil

from mofa.utils.rdkit_conf_parallel import compute_confs_worker
import mofa.utils.prepare_dataset as prep
from mofa.utils.filter_and_merge import run as fm_run


def process_fragments(fragment_dir: Path,
                      nodes: Sequence[str] = ('CuCu',)):
    """

    Args:
        fragment_dir: Directory containing the pre-processed fragment data
        nodes: Which nodes to process
    """
    for node in nodes:
        target_dir = fragment_dir / f'sdf/{node}/'
        input_smiles = fragment_dir / f'fragments_smi/frag_{node}.txt'
        output_template = 'hMOF_frag'
        out_dir = fragment_dir / f'fragments_all/{node}/'
        cores = '0'

        # generate sdf of molecular fragments
        for path in [target_dir, out_dir]:
            path.mkdir(parents=True, exist_ok=True)

        smiles = []
        with open(input_smiles, 'r') as f:
            for line in f:
                smiles.append(line.strip())

        # Generate the SDFs
        sdf_file = out_dir / f"{output_template}.sdf"
        compute_confs_worker(smifile=smiles, sdffile=sdf_file,
                             pid=f"{cores}", verbose=False)
        for sdf in glob('*.sdf'):
            shutil.move(sdf, target_dir)

        # generate sdf for fragment and connection atom
        os.makedirs(out_dir, exist_ok=True)
        sdf_path = os.path.join(out_dir, f'{output_template}.sdf')
        out_mol_path = os.path.join(out_dir, f'{output_template}_mol.sdf')
        out_frag_path = os.path.join(out_dir, f'{output_template}_frag.sdf')
        out_link_path = os.path.join(out_dir, f'{output_template}_link.sdf')
        out_table_path = os.path.join(out_dir, f'{output_template}_table.csv')
        prep.run(table_path=input_smiles, sdf_path=sdf_path, out_mol_path=out_mol_path,
                 out_frag_path=out_frag_path, out_link_path=out_link_path, out_table_path=out_table_path)

        # filter and merge
        fm_run(input_dir=out_dir, output_dir=out_dir, template=output_template)
