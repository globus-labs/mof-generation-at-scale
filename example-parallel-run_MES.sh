#! /bin/bash

#AL SA ONLY
#python run_parallel_workflow_MES.py \
#      --node-path input-files/zn-paddle-pillar/node.json \
#      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
#      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
#      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
#      --retrain-freq 2 \
#      --num-epochs 4 \
#      --num-samples 128 \
#      --gen-batch-size 64 \
#      --best-fraction 0.5 \
#      --maximum-strain 0.5 \
#      --simulation-budget 1000 \
#      --redis-host 127.0.0.1 \
#      --compute-config AL;

#python run_parallel_workflow_MES.py \
#      --node-path input-files/zn-paddle-pillar/node.json \
#      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
#      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
#      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
#      --retrain-freq 2 \
#      --num-epochs 4 \
#      --num-samples 128 \
#      --gen-batch-size 64 \
#      --best-fraction 0.5 \
#      --maximum-strain 0.5 \
#      --simulation-budget 1000 \
#      --redis-host 127.0.0.1 \
#      --compute-config AL;


#AL TANI ONLY
#python run_parallel_workflow_MES2.py \
#      --node-path input-files/zn-paddle-pillar/node.json \
#      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
#      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
#      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
#      --retrain-freq 2 \
#      --num-epochs 4 \
#      --num-samples 128 \
#      --gen-batch-size 64 \
#      --best-fraction 0.5 \
#      --maximum-strain 0.5 \
#      --simulation-budget 1000 \
#      --redis-host 127.0.0.1 \
#      --compute-config AL;

#python run_parallel_workflow_MES2.py \
#      --node-path input-files/zn-paddle-pillar/node.json \
#      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
#      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
#      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
#      --retrain-freq 2 \
#      --num-epochs 4 \
#      --num-samples 128 \
#      --gen-batch-size 64 \
#      --best-fraction 0.5 \
#      --maximum-strain 0.5 \
#      --simulation-budget 1000 \
#      --redis-host 127.0.0.1 \
#      --compute-config AL;


#AL EXPLORE ONLY
#python run_parallel_workflow_MES3.py \
#      --node-path input-files/zn-paddle-pillar/node.json \
#      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
#      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
#      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
#      --retrain-freq 2 \
#      --num-epochs 4 \
#      --num-samples 128 \
#      --gen-batch-size 64 \
#      --best-fraction 0.5 \
#      --maximum-strain 0.5 \
#      --simulation-budget 1000 \
#      --redis-host 127.0.0.1 \
#      --compute-config AL;

python run_parallel_workflow_MES.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --retrain-freq 2 \
      --num-epochs 4 \
      --num-samples 128 \
      --gen-batch-size 64 \
      --best-fraction 0.5 \
      --maximum-strain 0.5 \
      --simulation-budget 1000 \
      --redis-host 127.0.0.1 \
      --compute-config AL;

