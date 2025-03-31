#! /bin/bash

# Make a virtual environment from the frameworks
module load frameworks/2024.2.1_u1
python3 -m venv ./venv --system-site-packages
source ./venv/bin/activate

# Install the MOFA stuff
export PACKAGE_NAME=pytorch
pip install git+https://github.com/azton/lightning.git
pip install -e .
