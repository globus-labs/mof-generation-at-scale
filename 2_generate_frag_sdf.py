import os
from glob import glob
import shutil
import subprocess
from subprocess import PIPE
from utils.rdkit_conf_parallel import compute_confs_worker
import utils.prepare_dataset as prep
from utils.filter_and_merge import run as fm_run


nodes = ['CuCu']
# change to the line below to reproduce paper result
# nodes = [i.split('_')[1].split('.sdf')[0] for i in os.listdir('data/conformers') if 'conformers' in i]

# create necessary folders
os.makedirs(f'data/sdf',exist_ok=True)

for node in nodes:
    print(f'Now on node {node}')
    TARGET_DIR = f'data/sdf/{node}/'
    INPUT_SMILES=f'data/fragments_smi/frag_{node}.txt'
    OUTPUT_TEMPLATE=f'hMOF_frag'
    OUT_DIR=f'data/fragments_all/{node}/'
    CORES='0'
    
    # generate sdf of molecular fragments
    print('Generating molecule sdf files...')
    os.makedirs(TARGET_DIR,exist_ok=True)
    
    smiles = []
    with open(INPUT_SMILES, 'r') as f:
        for line in f:
            smiles.append(line.strip())
    
    # subprocess.run([f'python -W ignore utils/rdkit_conf_parallel.py {INPUT_SMILES} {OUTPUT_TEMPLATE} --cores {CORES}'], shell=True, stdout=PIPE, stderr=PIPE)
    compute_confs_worker(smifile=smiles, sdffile=os.path.join(OUT_DIR, f"{OUTPUT_TEMPLATE}" + ".csv"), pid=f"{CORES}")
    for sdf in glob('*.sdf'):
        shutil.move(sdf, TARGET_DIR) 
    
    # generate sdf for fragment and connection atom
    print(f'Generating fragment and connection atom sdf files...')
    os.makedirs(OUT_DIR,exist_ok=True)
    # subprocess.run(f'python -W ignore utils/prepare_dataset_parallel.py --table {INPUT_SMILES} --sdf-dir {TARGET_DIR} --out-dir {OUT_DIR} --template {OUTPUT_TEMPLATE} --cores {CORES}',shell=True)
    sdf_path = os.path.join(OUT_DIR, f'{OUTPUT_TEMPLATE}.sdf')
    out_mol_path = os.path.join(OUT_DIR, f'{OUTPUT_TEMPLATE}_mol.sdf')
    out_frag_path = os.path.join(OUT_DIR, f'{OUTPUT_TEMPLATE}_frag.sdf')
    out_link_path = os.path.join(OUT_DIR, f'{OUTPUT_TEMPLATE}_link.sdf')
    out_table_path = os.path.join(OUT_DIR, f'{OUTPUT_TEMPLATE}_table.csv')
    prep.run(table_path=INPUT_SMILES, sdf_path=sdf_path, out_mol_path=out_mol_path, out_frag_path=out_frag_path, out_link_path=out_link_path, out_table_path=out_table_path)
    
    # filter and merge
    print(f'Filtering and merging ...')
    # subprocess.run(f'python -W ignore utils/filter_and_merge.py --in-dir {OUT_DIR} --out-dir {OUT_DIR} --template {OUTPUT_TEMPLATE} --number-of-files {CORES}',shell=True)
    fm_run(input_dir=OUT_DIR, output_dir=OUT_DIR, template=OUTPUT_TEMPLATE)




