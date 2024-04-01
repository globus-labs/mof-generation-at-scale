#!/bin/bash -le
#PBS -l select=64
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -N mofa-test
#PBS -A CSC249ADCD08_CNDA

# Change to working directory
cd ${PBS_O_WORKDIR}

# Activate the environment
module reset
module use /soft/modulefiles/
module use /home/ftartagl/graphics-compute-runtime/modulefiles
module load xpu-smi
module load oneapi/release/2023.12.15.001
module load intel_compute_runtime/release/775.20
module load gcc/12.2.0

source activate /lus/gila/projects/CSC249ADCD08_CNDA/mof-generation-at-scale/env
echo Running on `hostname`

# Start Redis
redis-server --bind 0.0.0.0 --appendonly no --logfile redis.log &
redis_pid=$!
echo launched redis on $redis_pid

# Run
python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path input-files/zn-paddle-pillar/geom_difflinker.ckpt \
      --ligand-templates input-files/zn-paddle-pillar/template_*.yml \
      --num-samples 1024 \
      --gen-batch-size 16 \
      --simulation-budget 100000 \
      --md-timesteps 1000000 \
      --md-snapshots 10 \
      --compute-config sunspot
echo Python done

# Shutdown services
kill $redis_pid
