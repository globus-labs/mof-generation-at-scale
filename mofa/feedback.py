from typing import Sequence
from pathlib import Path
import itertools
import os
import pandas as pd
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import AllChem

def collect_hp_linkers(database: Union[str, pathlib.Path]):
    # output to sdf
    conformer_sdf_path = output_dir / f'conformers/conformers_{node_name}.sdf'
    if not os.path.isfile(conformer_sdf_path):
        writer = Chem.SDWriter(str(conformer_sdf_path))
        for smile in all_smiles_unique:  # change to tqdm(all_smiles_unique) to reproduce paper result
            try:
                if (mol := Chem.MolFromSmiles(smile)) is None:
                    continue
                mol = Chem.AddHs(mol)
                AllChem.EmbedMultipleConfs(mol, numConfs=1)
                mol.GetConformer(0)
                for cid in range(mol.GetNumConformers()):
                    writer.write(mol, confId=cid)
            except ValueError:
                pass
    
    # generate fragment SMILES
    smiles_file = (output_dir / f'fragments_smi/frag_{node_name}.txt')
    if not smiles_file.exists():
        prepare_sdf(sdf_path=str(conformer_sdf_path), output_path=str(smiles_file), verbose=False)
