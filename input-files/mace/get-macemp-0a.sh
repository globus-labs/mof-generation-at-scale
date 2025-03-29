#! /bin/bash

wget https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model -O mace-mp0_medium
mace_create_lammps_model mace-mp0_medium
