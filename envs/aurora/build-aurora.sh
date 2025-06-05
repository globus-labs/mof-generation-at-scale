#! /bin/bash

# Make a virtual environment from the frameworks
module load frameworks/2025
python3 -m venv ./venv --system-site-packages
source ./venv/bin/activate

# Install the MOFA stuff
pip install -e .
# for aurora, take out the pypi install
pip uninstall pytorch-lightning -y
# put in the one with xpu support
export PACKAGE_NAME=pytorch
pip install git+https://github.com/azton/lightning.git
