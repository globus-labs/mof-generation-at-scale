#! /bin/bash

python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path input-files/zn-paddle-pillar/geom_difflinker.ckpt \
      --ligand-templates input-files/zn-paddle-pillar/template_*.yml \
      --num-samples 32 \
      --simulation-budget 4 \
      --redis-host 127.0.0.1 \
      --compute-config local
