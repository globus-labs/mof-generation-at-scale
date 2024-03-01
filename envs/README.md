Environment files for different resources

## Building LAMMPS on Polaris

We need a copy of LAMMPS that uses GPUs and not MPI.
The recommended option from [from ALCF is to build LAMMPS with Kokkos](https://github.com/argonne-lcf/GettingStarted/tree/master/Applications/Polaris/LAMMPS).

First download then build LAMMPS following

```bash
#! /bin/bash

# Make the build environment
module use /soft/modulefiles
module load cudatoolkit-standalone/11.8.0
module load kokkos
module list

export KOKKOS_ABSOLUTE_PATH=`realpath $KOKKOS_PATH`
export NVCC_WRAPPER_DEFAULT_COMPILER=nvc++

mkdir build-kokkos-nompi
cd build-kokkos-nompi

cmake ../cmake \
    -DPKG_KOKKOS=on \
    -DKokkos_ARCH_ZEN3=yes \
    -DKokkos_ARCH_AMPERE80=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ENABLE_OPENMP=no \
    -DPKG_MOFFF=on \
    -DFFT=KISS \
    -DPKG_QEQ=on \
    -DPKG_REAXFF=on \
    -DPKG_PTM=on \
    -DPKG_RIGID=on \
    -DPKG_MOLECULE=on \
    -DPKG_EXTRA-MOLECULE=on \
    -DPKG_EXTRA-FIX=on \
    -DPKG_KSPACE=on \
    -DPKG_MANYBODY=on \
    -DPKG_GRANULAR=on \
    -DBUILD_OMP=yes \
    -DBUILD_MPI=no

make -j 16
```

> Logan hasn't figured out OpenMP yet, so there is likely more performance from both CUDA+OpenMP
