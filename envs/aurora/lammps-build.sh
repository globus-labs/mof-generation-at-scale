#! /bin/bash

FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

cd src
make mpi-stubs
make ml-mace
make -j 16 macesunspotkokkos
