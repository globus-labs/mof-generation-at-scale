import os
import subprocess
import itertools
from tqdm import tqdm
import pandas as pd
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import AllChem
from mofa.utils.prepare_data_from_sdf import prepare_sdf
from typing import *

def fragmentation(nodes: List[str]=["CuCu"]):
    # data cleaning
    df_info = pd.read_csv('mofa/data/hMOF_CO2_info.csv')
    df_info = df_info.dropna() # drop entries containing 'NaN'
    df_info = df_info[df_info.CO2_capacity_001>0] # only keep entries with positive CO2 working capacity
    df_info = df_info[~df_info.MOFid.str.contains('ERROR')] # drop entries with error
    df_info = df_info[~df_info.MOFid.str.contains('NA')] # drop entries with NA
    
    # get node and linker information
    metal_eles = ['Zn', 'Cu', 'Mn', 'Zr', 'Co', 'Ni', 'Fe', 'Cd', 'Pb', 'Al', 'Mg', 'V',
           'Tb', 'Eu', 'Sm', 'Tm', 'Gd', 'Nd', 'Dy', 'La', 'Ba', 'Ga', 'In',
           'Ti', 'Be', 'Ce', 'Li', 'Pd', 'Na', 'Er', 'Ho', 'Yb', 'Ag', 'Pr',
           'Cs', 'Mo', 'Lu', 'Ca', 'Pt', 'Ge', 'Sc', 'Hf', 'Cr', 'Bi', 'Rh',
           'Sn', 'Ir', 'Nb', 'Ru', 'Th', 'As', 'Sr']
    
    # get a list of metal nodes & create a new column named "metal_nodes"
    metal_nodes = []
    organic_linkers = []
    for i,mofid in tqdm(enumerate(df_info.MOFid)):
        sbus = mofid.split()[0].split('.')
        metal_nodes.append([c for c in sbus if any(e in c for e in metal_eles)][0])
        organic_linkers.append([c for c in sbus if any(e in c for e in metal_eles)==False])
    
    df_info['metal_node'] = metal_nodes
    df_info['organic_linker'] = organic_linkers
    
    # get most occuring nodes
    unique_nodes = [n for n in list(df_info['metal_node'].unique()) if len(n)<=30] # node smiles should be shorter then 30 strings
    df_info = df_info[df_info['metal_node'].isin(unique_nodes)] # filter df_info based on unique_nodes
    freq = [df_info['metal_node'].value_counts()[value] for value in list(df_info.metal_node.unique())] # get frequency of unique nodes
    df_freq = pd.DataFrame({'node':list(df_info.metal_node.unique()),'freq':freq})
    print('node occurance:')
    print(df_freq.iloc[:5,:])
    unique_node_select = list(df_freq[df_freq.freq>=5000].node) # select occuring nodes
    df_info_select = df_info[df_info['metal_node'].isin(unique_node_select)] # select df_info with node only in list(unique_node_select)
    
    # create necessary folders
    os.makedirs(f'mofa/data/conformers',exist_ok=True)
    os.makedirs(f'mofa/data/data_by_node',exist_ok=True)
    os.makedirs(f'mofa/data/fragments_smi',exist_ok=True)
    
    # output each node to a separate csv files
    for n in unique_node_select:
        n_name = n.replace('[','').replace(']','').replace('(','').replace(')','')
        df_info_select_node = df_info[df_info.metal_node == n]
        df_info_select_node.to_csv(f'mofa/data/data_by_node/{n_name}.csv',index=False)
    
    # load data
    for node in nodes: # change to ['CuCu','ZnZn','ZnOZnZnZn'] to reproduce paper result
        node_name = node.replace('[','').replace(']','').replace('(','').replace(')','')
        print(f'Now on node {node_name} ... ')
        input_data_path = f'mofa/data/data_by_node/{node_name}.csv' 
    
        df = pd.read_csv(input_data_path)
    
        # MOFs with three linkers
        len_linkers = [len(eval(df['organic_linker'].iloc[i])) for i in range(len(df['organic_linker']))]
        df['len_linkers'] = len_linkers
        df_three_linkers = df[df.len_linkers==3]
    
        # MOFs with high working capactiy at (wc > 2mmol/g @ 0.1 bar)
        df_high_wc = df[df['CO2_capacity_01'] >=2]
    
        # MOFs with high working capacity and three linkers
        len_linkers = [len(eval(df_high_wc['organic_linker'].iloc[i])) for i in range(len(df_high_wc['organic_linker']))]
        df_high_wc['len_linkers'] = len_linkers
        df_high_wc_3_linkers = df_high_wc[df_high_wc.len_linkers==3]
    
        # get list of SMILES for all linkers
        list_smiles = [eval(i) for i in df_high_wc['organic_linker']]
        all_smiles = list(itertools.chain(*list_smiles))
        print(f'number of smiles: {len(all_smiles)}')
        all_smiles_unique = list(pd.Series(all_smiles).unique())
        print(f'number of unique_smiles: {len(all_smiles_unique)}')
    
        # remove the line below to reproduce paper results
        all_smiles_unique = all_smiles_unique[:1000]
    
        # output to sdf
        print('Outputting conformers to sdf ... ')
        conformer_sdf_path = f'mofa/data/conformers/conformers_{node_name}.sdf'
        if not os.path.isfile(conformer_sdf_path):
            writer = Chem.SDWriter(conformer_sdf_path)
            for smile in tqdm(all_smiles_unique): # change to tqdm(all_smiles_unique) to reproduce paper result
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
                    conformers = AllChem.EmbedMultipleConfs(mol, numConfs=1)
                    conformer = mol.GetConformer(0)
                    for cid in range(mol.GetNumConformers()):
                        writer.write(mol, confId=cid)
                except:
                    pass
        if not os.path.isfile(f'mofa/data/fragments_smi/frag_{node_name}.txt'):
            # generate fragment SMILES
            print('Generating SMILES ... ')
            prepare_sdf(sdf_path=f"mofa/data/conformers/conformers_{node_name}.sdf", output_path=f"mofa/data/fragments_smi/frag_{node_name}.txt", verbose=True)
            # subprocess.run(f'python utils/prepare_data_from_sdf.py --sdf_path data/conformers/conformers_{node_name}.sdf --output_path data/fragments_smi/frag_{node_name}.txt --verbose',shell=True)

if __name__ == "__main__":
    fragmentation()
