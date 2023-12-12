import os
import argparse
import subprocess
from mofa.utils.difflinker_sample_and_analyze import main_run, run_dflk_sample_analyze
from typing import *
from pathlib import Path


def sampler(nodes: List[str] = ['CuCu'], n_atoms_list: List[int] = [8]):
    # nodes = ['CuCu']
    # change to the line below to reproduce paper result
    # nodes = [i.split('_')[1].split('.sdf')[0] for i in os.listdir('data/conformers') if 'conformers' in i]

    for n_atoms in n_atoms_list:
        # change to the line below to reproduce paper result
        # for n_atoms in range(5,11):
        print(f'Sampling {n_atoms} atoms...')
        for node in nodes:
            if node != 'V':
                print(f'Now on node: {node}')
                OUTPUT_DIR = f'mofa/output/n_atoms_{n_atoms}/{node}'
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                # subprocess.run(f'python -W ignore utils/difflinker_sample_and_analyze.py --linker_size {n_atoms} --fragments data/fragments_all/{node}/hMOF_frag.sdf --model models/geom_difflinker.ckpt --output {OUTPUT_DIR} --n_samples 1',shell=True)
                main_run(input_path=f"mofa/data/fragments_all/{node}/hMOF_frag_frag.sdf", model="mofa/models/geom_difflinker.ckpt", linker_size=str(n_atoms),
                         output_dir=OUTPUT_DIR, n_samples=1, n_steps=None, anchors=None)

                # change to the line below to reproduce paper result
                # subprocess.run(f'python -W ignore utils/difflinker_sample_and_analyze.py --linker_size {n_atoms} --fragments data/fragments_all/{node}/hMOF_frag.sdf --model models/geom_difflinker.ckpt --output {OUTPUT_DIR} --n_samples 20',shell=True)


def sample_from_sdf(
        input_path: str | Path,
        output_dir: str | Path,
        node: str = 'CuCu',
        n_atoms: int = 8,
        model: str | Path = "mofa/models/geom_difflinker.ckpt",
        n_samples: int = 1,
        n_steps: int = None
):
    """Run the sampling and write new molecules to an output directory"""
    if node == 'V':
        raise NotAllowedElementError()

    main_run(
        input_path=input_path,
        output_dir=output_dir,
        model=model,
        linker_size=str(n_atoms),
        n_samples=n_samples,
        n_steps=n_steps,
        anchors=None,
        device='cpu'
    )


class NotAllowedElementError(Exception):
    def __init__(self, message="Element V is not allowed!"):
        self.message = message
        super().__init__(self.message)
