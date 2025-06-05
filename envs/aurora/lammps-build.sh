#! /bin/bash

module reset
module load frameworks
module load kokkos
module list

export KOKKOS_PATH=/lus/flare/projects/catalyst/world_shared/avazquez/soft/lammps-kokkos/lib/kokkos
export KOKKOS_USE_DEPRECATED_MAKEFILES=1
FPATH=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

cd src
make mpi-stubs
make yes-ml-mace
make -j 8 macesunspotkokkos
