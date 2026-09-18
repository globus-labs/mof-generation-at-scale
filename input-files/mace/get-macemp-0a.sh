#! /bin/bash

if [ ! -e mace-mp0_medium ]; then
  wget https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model -O mace-mp0_medium
fi
python create_mliap_model.py mace-mp0_medium --dtype float32
