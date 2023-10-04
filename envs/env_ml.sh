#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n dpl310 python=3.10 -y
echo $CONDA_PREFIX
conda activate dpl310
echo $CONDA_PREFIX

#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch -y
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

conda install -c pyg pyg -y

conda install -c conda-forge rdkit -y

conda install -c conda-forge openbabel -y

conda install -c conda-forge lammps -y

conda install -c conda-forge xtb -y

conda install -c conda-forge raspa2 -y

#conda install -c dglteam dgl-cuda11.7 -y

pip install pymatgen plotly pyarrow polars PyCifRW einops curtsies p_tqdm transformers pathlib scikit-image argparse wandb cairosvg h5py pynvml pytorch-lightning jupyter ase fairscale timm fast_ml seaborn captum kmapper umap-learn dash

# for topological data analysis

pip install persim ripser MDAnalysis ray git+https://github.com/bruel-gabrielsson/TopologyLayer.git
