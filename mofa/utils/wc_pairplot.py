import os
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

df_info = pd.read_csv('../data/hMOF_CO2_info.csv')
df_info = df_info.dropna() # drop entries containing 'NaN'
df_info = df_info[df_info.CO2_wc_001>0] # only keep entries with positive CO2 working capacity
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
df_info['organic_linkers'] = organic_linkers

os.makedirs('../data/hMOF_by_node/',exist_ok=True)

unique_nodes = [n for n in list(df_info['metal_node'].unique()) if len(n)<=30] # node smiles should be shorter then 30 strings
df_info = df_info[df_info['metal_node'].isin(unique_nodes)] # filter df_info based on unique_nodes
freq = [df_info['metal_node'].value_counts()[value] for value in list(df_info.metal_node.unique())] # get frequency of unique nodes
df_freq = pd.DataFrame({'node':list(df_info.metal_node.unique()),'freq':freq})
#print(df_freq)
unique_node_select = ['[Zn][Zn]', '[Cu][Cu]', '[Zn][O]([Zn])([Zn])[Zn]', '[V]'] # select the most occuring nodes
df_info_select = df_info[df_info['metal_node'].isin(unique_node_select)] # select df_info with node only in list(unique_node_select)
df_info_select.to_csv('../data/hMOF_CO2_info_metal_linker.csv',index=False)


# output df for each node  to a separate csv files
for n in unique_node_select:
    df_info_select_node = df_info[df_info.metal_node == n]
    df_info_select_node.to_csv(f'../data/hMOF_by_node/{n}.csv',index=False)


metal_node_name = []
for n in df_info_select.metal_node:
    if n == '[Zn][Zn]':
        metal_node_name.append('Zn paddlewheel-pcu')
    if n == '[Cu][Cu]':
        metal_node_name.append('Cu paddlewheel-pcu')
    if n == '[Zn][O]([Zn])([Zn])[Zn]':
        metal_node_name.append('Zn tetramer-pcu')
    if n == '[V]':
        metal_node_name.append('V-rna')
df_info_select['metal_node_name'] = metal_node_name


df_info_select.columns = ['MOF','MOFid','0.01bar','0.05bar','0.1bar','0.5bar','2.5bar','metal_node','organic_linkers','node-topology'] # rename columns for plotting
sns.pairplot(df_info_select,hue='node-topology',corner=True)
plt.tight_layout()
plt.savefig('../plots/pairplot.png', dpi=300, bbox_inches="tight")