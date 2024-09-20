#! /bin/bash

# Extract the launch option if provided
LAUNCH_OPTION=""

# Check for valid launch options and set LAUNCH_OPTION accordingly
if [[ "$1" == "--launch-option both" || "$1" == "--launch-option server" || "$1" == "--launch-option thinker" ]]; then
    LAUNCH_OPTION="$1"
fi

python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --retrain-freq 2 \
      --num-epochs 4 \
      --num-samples 128 \
      --gen-batch-size 64 \
      --simulation-budget 8 \
      --redis-host 127.0.0.1 \
      --dft-opt-steps 0 \
      --compute-config local \
      $LAUNCH_OPTION

