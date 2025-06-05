#!/bin/bash -le
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -N mofa-test
#PBS -A MOFA

hostname
# Change to working directory
cd ${PBS_O_WORKDIR}

# Activate the environment
module load xpu-smi
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
conda deactivate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Change to working directory
cd ${PBS_O_WORKDIR}
pwd

# Run
python run_test.py --config polaris --device xpu $args
echo Python done
