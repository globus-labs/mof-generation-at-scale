#!/bin/bash -l
# Map node-local MPI ranks across the four Polaris GPUs. This is the helper
# documented in envs/polaris.md and used by the original Polaris test.
num_gpus=4
local_size=$PMI_LOCAL_SIZE
ranks_per_gpu=$(( local_size / num_gpus ))

# GPUs are assigned in reverse order to match the Polaris device topology.
gpu=$(( (local_size - 1 - PMI_LOCAL_RANK) / ranks_per_gpu ))
export CUDA_VISIBLE_DEVICES=$gpu
exec "$@"
