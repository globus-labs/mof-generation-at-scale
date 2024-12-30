#! /bin/bash

python run_serial_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path input-files/zn-paddle-pillar/geom_difflinker.ckpt \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --num-samples 64 \
      --num-to-assemble 4 \
      --torch-device cuda
