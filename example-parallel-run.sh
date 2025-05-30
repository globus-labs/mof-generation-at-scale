#! /bin/bash
# Run MOFA locally on a system with a single NVIDIA GPU

python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --retrain-freq 2 \
      --num-epochs 4 \
      --num-samples 128 \
      --gen-batch-size 128 \
      --simulation-budget 8 \
      --redis-host 127.0.0.1 \
      --compute-config "local" \
      --mace-model-path ./input-files/mace/mace-mp0_medium-lammps.pt \
      --md-timesteps 1000 \
      --dft-opt-steps 2


# Xiaoli local
# python run_parallel_workflow.py \
#       --node-path input-files/zn-paddle-pillar/node.json \
#       --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
#       --generator-config-path models/geom-300k/config-tf32-a100.yaml \
#       --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
#       --retrain-freq 2 \
#       --num-epochs 4 \
#       --num-samples 32 \
#       --simulation-budget 1 \
#       --redis-host 127.0.0.1 \
#       --compute-config localXY
