#!/bin/bash -le
#PBS -l select=32
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -N mofa-prod
#PBS -A MOFA

# Change to working directory
cd ${PBS_O_WORKDIR}

# Activate the environment
module load xpu-smi
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
conda deactivate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Put Redis and MongoDB on the path
export PATH=$PATH:`realpath conda-env/bin/`

# Start Redis
redis-server --bind 0.0.0.0 --appendonly no --logfile redis.log &
redis_pid=$!
echo launched redis on $redis_pid

# Run
python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --retrain-freq 64 \
      --num-epochs 8 \
      --num-samples 1024 \
      --gen-batch-size 64 \
      --simulation-budget -1 \
      --redis-host 127.0.0.1 \
      --compute-config aurora \
      --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
      --md-timesteps 3000 \
      --dft-opt-steps 8

echo Python done

# Shutdown services
kill $redis_pid
