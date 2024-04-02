#!/bin/bash -le
#PBS -l select=2:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug
#PBS -N mofa-test
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
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --maximum-train-size 8192 \
      --retrain-freq 128 \
      --num-epochs 128 \
      --num-samples 1024 \
      --gen-batch-size 16 \
      --simulation-budget 32768 \
      --md-timesteps 1000000 \
      --md-snapshots 10 \
      --compute-config polaris
echo Python done

# Shutdown services
kill $redis_pid
