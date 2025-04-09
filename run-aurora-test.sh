#!/bin/bash -le
#PBS -l select=10
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
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
      --ai-fraction 0.2 \
      --retrain-freq 2 \
      --num-epochs 4 \
      --num-samples 128 \
      --gen-batch-size 64 \
      --simulation-budget 0 \
      --compute-config aurora \
      --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
      --md-timesteps 1000 \
      --proxy-threshold 1000 \
      --dft-opt-steps 2

echo Python done

# Shutdown services
kill $redis_pid
