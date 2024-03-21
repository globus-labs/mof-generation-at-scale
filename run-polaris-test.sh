#!/bin/bash -le
#PBS -l select=10:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug-scaling
#PBS -N test-run
#PBS -A examol

# Change to working directory
cd ${PBS_O_WORKDIR}

# Activate the environment
conda activate /lus/eagle/projects/ExaMol/mofa/mof-generation-at-scale/env-polaris

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
      --simulation-budget 256 \
      --md-timesteps 1000000 \
      --md-snapshots 10 \
      --compute-config polaris
echo Python done

# Shutdown services
kill $redis_pid
