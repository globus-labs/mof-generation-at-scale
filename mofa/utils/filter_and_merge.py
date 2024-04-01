from mofa.utils.src.datasets import read_sdf
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os


def run(input_dir, output_dir, template):

    os.makedirs(output_dir, exist_ok=True)
    out_table_path = os.path.join(output_dir, f'{template}_table.csv')
    out_mol_path = os.path.join(output_dir, f'{template}_mol.sdf')
    out_frag_path = os.path.join(output_dir, f'{template}_frag.sdf')
    out_link_path = os.path.join(output_dir, f'{template}_link.sdf')

    full_table = []
    full_molecules = []
    full_fragments = []
    full_linkers = []

    # for idx in range(n):
    mol_path = os.path.join(input_dir, f'{template}_mol.sdf')
    frag_path = os.path.join(input_dir, f'{template}_frag.sdf')
    link_path = os.path.join(input_dir, f'{template}_link.sdf')
    table_path = os.path.join(input_dir, f'{template}_table.csv')

    table = pd.read_csv(table_path)
    table['idx'] = table.index
    grouped_table = (
        table
        .groupby(['molecule', 'fragments', 'linker', 'anchor_1', 'anchor_2'])
        .min()
        .reset_index()
        .sort_values(by='idx')
    )
    idx_to_keep = set(grouped_table['idx'].unique())
    table['keep'] = table['idx'].isin(idx_to_keep)

    generator = tqdm(
        zip(table.iterrows(), read_sdf(mol_path), read_sdf(frag_path), read_sdf(link_path)),
        total=len(table),
        desc='Full data',
    )
    try:
        for (_, row), molecule, fragments, linker in generator:
            if row['keep']:
                if molecule.GetProp('_Name') != row['molecule']:
                    print('Molecule _Name:', molecule.GetProp('_Name'), row['molecule'])
                    continue

                full_table.append(row)
                full_molecules.append(molecule)
                full_fragments.append(fragments)
                full_linkers.append(linker)
    except ValueError:
        pass

    full_table = pd.DataFrame(full_table)
    full_table.to_csv(out_table_path, index=False)
    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        for mol in tqdm(full_molecules):
            writer.write(mol)
    with Chem.SDWriter(open(out_frag_path, 'w')) as writer:
        writer.SetKekulize(False)
        for frags in tqdm(full_fragments):
            writer.write(frags)
    with Chem.SDWriter(open(out_link_path, 'w')) as writer:
        writer.SetKekulize(False)
        for linker in tqdm(full_linkers):
            writer.write(linker)
