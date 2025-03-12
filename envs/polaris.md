# Installing on Polaris

Polaris is a system with NVIDIA GPUs and AMD Zen3 processors. 
Install MOFA and its dependencies by following these guides.

## Building CP2K

## Building LAMMPS on Polaris

We need a copy of LAMMPS that uses GPUs and not MPI.
While the recommended option from [from ALCF is to build LAMMPS with Kokkos](https://github.com/argonne-lcf/GettingStarted/tree/master/Applications/Polaris/LAMMPS),
our MOF simulations run best with the GPU package.

First download then build LAMMPS following

```bash
#! /bin/bash

# Make the build environment
module reset
module use /soft/modulefiles
module use /soft/modulefiles
module load spack-pe-base cmake
module list

mkdir build-gpu-nompi-mixed
cd build-gpu-nompi-mixed

cmake ../cmake -DCMAKE_BUILD_TYPE=release \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_C_COMPILER=nvc++ \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_CXX_STANDARD=14 \
    -DLAMMPS_MEMALIGN=64 \
    -DLAMMPS_SIZES=smallsmall \
    -DPKG_MOFFF=on \
    -DFFT=KISS \
    -DPKG_QEQ=on \
    -DPKG_REAXFF=on \
    -DPKG_RIGID=on \
    -DPKG_MOLECULE=on \
    -DPKG_EXTRA-MOLECULE=on \
    -DPKG_EXTRA-FIX=on \
    -DPKG_KSPACE=on \
    -DPKG_MANYBODY=on \
    -DPKG_GRANULAR=on \
    -DPKG_GPU=on \
    -DGPU_API=cuda \
    -DGPU_PREC=mixed \
    -DGPU_ARCH=sm_80 \
    -DGPU_DEBUG=no \
    -DCUDA_MPS_SUPPORT=yes \
    -DBUILD_OMP=no \
    -DBUILD_MPI=no \
    -DCUDA_NVCC_FLAGS="-std=c++14 -allow-unsupported-compiler -Xcompiler" \
    -DCMAKE_CXX_FLAGS="-std=c++14 -DCUDA_PROXY"

make -j 16

```
