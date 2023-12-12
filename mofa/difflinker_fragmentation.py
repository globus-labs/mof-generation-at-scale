from typing import Sequence
from pathlib import Path
import itertools
import os

import pandas as pd
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import AllChem

from mofa.utils.prepare_data_from_sdf import prepare_sdf


def fragmentation(
        data_path: Path,
        output_dir: Path,
        nodes: Sequence[str]
):
    """Fragment MOFs from a specific dataset

    Args:
        data_path: Path to a CSV file containing the target MOFs
        output_dir: Directory in which to write fragmented MOFs
    """
    # data cleaning
    df_info = pd.read_csv(data_path)
    df_info = df_info.dropna()  # drop entries containing 'NaN'
    df_info = df_info[df_info.CO2_capacity_001 > 0]  # only keep entries with positive CO2 working capacity
    df_info = df_info[~df_info.MOFid.str.contains('ERROR')]  # drop entries with error
    df_info = df_info[~df_info.MOFid.str.contains('NA')]  # drop entries with NA

    # get node and linker information
    metal_eles = ['Zn', 'Cu', 'Mn', 'Zr', 'Co', 'Ni', 'Fe', 'Cd', 'Pb', 'Al', 'Mg', 'V',
                  'Tb', 'Eu', 'Sm', 'Tm', 'Gd', 'Nd', 'Dy', 'La', 'Ba', 'Ga', 'In',
                  'Ti', 'Be', 'Ce', 'Li', 'Pd', 'Na', 'Er', 'Ho', 'Yb', 'Ag', 'Pr',
                  'Cs', 'Mo', 'Lu', 'Ca', 'Pt', 'Ge', 'Sc', 'Hf', 'Cr', 'Bi', 'Rh',
                  'Sn', 'Ir', 'Nb', 'Ru', 'Th', 'As', 'Sr']

    # get a list of metal nodes & create a new column named "metal_nodes"
    metal_nodes = []
    organic_linkers = []
    for i, mofid in enumerate(df_info.MOFid):
        sbus = mofid.split()[0].split('.')
        metal_nodes.append([c for c in sbus if any(e in c for e in metal_eles)][0])
        organic_linkers.append([c for c in sbus if not any(e in c for e in metal_eles)])

    df_info['metal_node'] = metal_nodes
    df_info['organic_linker'] = organic_linkers

    # get most occurring nodes
    unique_nodes = [n for n in list(df_info['metal_node'].unique()) if len(n) <= 30]  # node smiles should be shorter then 30 strings
    df_info = df_info[df_info['metal_node'].isin(unique_nodes)]  # filter df_info based on unique_nodes
    freq = [df_info['metal_node'].value_counts()[value] for value in list(df_info.metal_node.unique())]  # get frequency of unique nodes
    df_freq = pd.DataFrame({'node': list(df_info.metal_node.unique()), 'freq': freq})
    unique_node_select = list(df_freq[df_freq.freq >= 5000].node)  # select occuring nodes

    # create necessary folders
    for path in ['conformers', 'data_by_node', 'fragments_smi']:
        (output_dir / path).mkdir(parents=True, exist_ok=True)

    # output each node to a separate csv files
    for n in unique_node_select:
        n_name = n.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        df_info_select_node = df_info[df_info.metal_node == n]
        df_info_select_node.to_csv(output_dir / f'data_by_node/{n_name}.csv', index=False)

    # load data
    for node in nodes:  # change to ['CuCu','ZnZn','ZnOZnZnZn'] to reproduce paper result
        node_name = node.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        input_data_path = output_dir / f'data_by_node/{node_name}.csv'

        df = pd.read_csv(input_data_path)

        # MOFs with three linkers
        len_linkers = [len(eval(df['organic_linker'].iloc[i])) for i in range(len(df['organic_linker']))]
        df['len_linkers'] = len_linkers

        # MOFs with high working capactiy at (wc > 2mmol/g @ 0.1 bar)
        df_high_wc = df[df['CO2_capacity_01'] >= 2]

        # MOFs with high working capacity and three linkers
        len_linkers = [len(eval(df_high_wc['organic_linker'].iloc[i])) for i in range(len(df_high_wc['organic_linker']))]
        df_high_wc['len_linkers'] = len_linkers

        # get list of SMILES for all linkers
        list_smiles = [eval(i) for i in df_high_wc['organic_linker']]
        all_smiles = list(itertools.chain(*list_smiles))
        all_smiles_unique = list(pd.Series(all_smiles).unique())

        # remove the line below to reproduce paper results
        all_smiles_unique = all_smiles_unique[:1000]  # TODO (wardlt): remove this?

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
