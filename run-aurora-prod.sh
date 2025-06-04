#!/bin/bash -le
#PBS -l select=1024
#PBS -l walltime=3:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -N mofa-prod
#PBS -A MOFA
#PBS -W tolerate_node_failures=all

hostname
# Change to working directory
cd ${PBS_O_WORKDIR}

# Activate the environment
module load xpu-smi
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
conda deactivate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Ensure we don't overload threads
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Put Redis and MongoDB on the path
export PATH=$PATH:`realpath conda-env/bin/`

# Start Redis
redis-server --bind 0.0.0.0 --appendonly no --maxclients 1000000 --logfile redis.log &
redis_pid=$!
echo launched redis on $redis_pid

# Run
python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --retrain-freq 64 \
      --num-epochs 16 \
      --num-samples 8096 \
      --gen-batch-size 512 \
      --ai-fraction 0.05 \
      --dft-fraction 0.6 \
      --simulation-budget -1 \
      --compute-config ./configs/aurora/aurora-pwdft.py \
      --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
      --md-timesteps 3000 \
      --proxy-threshold 100000 \
      --dft-opt-steps 2 \
      --lammps-on-ramdisk

echo Python done

# Shutdown services
kill $redis_pid
