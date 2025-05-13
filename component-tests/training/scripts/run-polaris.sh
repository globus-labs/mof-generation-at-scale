#!/bin/bash -le
#PBS -l select=1:system=polaris
#PBS -l walltime=0:60:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug
#PBS -N train-test
#PBS -A MOFA

hostname

# Change to working directory
cd ${PBS_O_WORKDIR}
pwd

# Activate the environment
conda activate /lus/eagle/projects/MOFA/lward/mof-generation-at-scale/env
which python

# Run
python run_test.py --config polaris --device cuda $args
echo Python done
