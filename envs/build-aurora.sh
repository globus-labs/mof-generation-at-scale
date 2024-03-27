#! /bin/bash

module reset
module use /soft/modulefiles/
module use /home/ftartagl/graphics-compute-runtime/modulefiles
module unload intel_compute_runtime/release/agama-devel-647
module load oneapi/release/2023.12.15.001
module load graphics-compute-runtime/agama-ci-devel-736.25
module load gcc/12.1.0

conda env create --file envs/environment-aurora.yml --force -p ./env
