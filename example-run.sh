#! /bin/bash

python run_serial_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path input-files/zn-paddle-pillar/geom_difflinker.ckpt \
      --fragment-path input-files/zn-paddle-pillar/hMOF_frag_frag.sdf \
      --num-samples 256 \
      --torch-device cuda
