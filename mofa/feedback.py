from typing import Sequence
import pathlib
from pathlib import Path
import itertools
import os
import pandas as pd
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import AllChem
import glob
from typing import *
from mofa.utils.prepare_data_from_sdf import prepare_sdf

ROOT_DIR = pathlib.Path(__file__).parent #mofa dir

def collect_hp_linkers(input_path: Union[str, pathlib.Path] = os.path.join(ROOT_DIR, "lammps/pdb/ligands"), 
                       out_path: Union[str, pathlib.Path] = os.path.join(ROOT_DIR, "feedback_output"), 
                       node_name="TEMPNODE"):

                           
    all_graphs_unique: List[str] = glob.glob(os.path.join(input_path, "*.pdb"))
    all_graphs_unique: List = list(map(lambda inp: Chem.MolFromPDBFile(inp), all_graphs_unique))
                           
    pathlib.Path(out_path).mkdir(exist_ok=True)
    # create necessary folders
    for path in ['conformers', 'fragments_smi']:
        pathlib.Path(os.path.join(out_path, path)).mkdir(parents=True, exist_ok=True)
        
    # output to sdf
    conformer_sdf_path = os.path.join(out_path, f"conformers/conformers_{node_name}.sdf")
    if not os.path.isfile(conformer_sdf_path):
        writer = Chem.SDWriter(str(conformer_sdf_path))
        for graph in all_graphs_unique:  # change to tqdm(all_smiles_unique) to reproduce paper result
            try:
                # if (mol := Chem.MolFromSmiles(smile)) is None:
                #     continue
                # mol = Chem.AddHs(mol)
                # AllChem.EmbedMultipleConfs(mol, numConfs=1)
                # mol.GetConformer(0)
                # for cid in range(mol.GetNumConformers()):
                writer.write(graph, confId=0)
            except ValueError:
                pass
    
    # generate fragment SMILES
    smiles_file = pathlib.Path(os.path.join(out_path, f'fragments_smi/frag_{node_name}.txt'))
    if not smiles_file.exists():
        prepare_sdf(sdf_path=str(conformer_sdf_path), output_path=str(smiles_file), verbose=False)

if __name__ == "__main__":
   collect_hp_linkers() 
   out_path = os.path.join(ROOT_DIR, "feedback_output")
   process_fragments(out_path, ("TEMPNODE", ))
