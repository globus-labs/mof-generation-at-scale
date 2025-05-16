#!/usr/bin/env bash

display_help() {
  echo " Will map whole gpu to rank in compact and then round-robin fashion"
  echo " Usage:"
  echo "   mpiexec -np N gpu_dev_compact.sh ./a.out"
  echo
  echo " Example 6 GPU with 3 Ranks:"
  echo "   0 Rank 0"
  echo "   1 Rank 1"
  echo "   2 Rank 2"
  echo 
  echo " Hacked together by apl@anl.gov, please contact if bug found"
  exit 1
}

#This give the exact GPU count i915 knows about and I use udev to only enumerate the devices with physical presence.
num_gpu=12
num_tile=1

if [ "$#" -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$num_gpu" = 0 ] ; then
  display_help
fi

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_RANKID=$MPI_LOCALRANKID 
elif [[ -v PMIX_RANK ]]; then
  _MPI_RANKID=$PMIX_RANK
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_RANKID=$PALS_LOCAL_RANKID
else
  display_help
fi

gpu_id=$(((_MPI_RANKID / num_tile) % num_gpu))

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=$gpu_id

#https://stackoverflow.com/a/28099707/7674852
"$@"

