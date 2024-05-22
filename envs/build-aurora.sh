#! /bin/bash

# Active the modules needed to build libraries later
module reset
module use /soft/modulefiles/
module use /home/ftartagl/graphics-compute-runtime/modulefiles
module load oneapi/release/2023.12.15.001
module load intel_compute_runtime/release/775.20
module load gcc/12.2.0

# Build then activate the environment
conda env create --file envs/environment-aurora.yml --force -p ./env
conda activate ./env

# Build torch_ccl locally
#  Clone from: https://github.com/intel/torch-ccl
cd libs/torch-ccl
COMPUTE_BACKEND=dpcpp pip install -e .

# Now install Corey's stuff
# Clone from: https://github.com/coreyjadams/lightning
cd ../lightning
PACKAGE_NAME=pytorch pip install -e .
