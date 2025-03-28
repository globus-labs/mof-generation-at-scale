# Installing on Polaris

Polaris is a system with NVIDIA GPUs and AMD Zen3 processors. 
Install MOFA and its dependencies by following these guides.

## Building CP2K

We use at CP2K 2025.1 in our latest builds and compile a version with ELPA for the distributed eigenvalue solver.

```bash
module reset
module use /soft/modulefiles 
module swap PrgEnv-nvhpc PrgEnv-gnu
module load cray-fftw
module load cudatoolkit-standalone/12.2
module load cray-libsci
module list

export CRAY_LIBSCI_PREFIX_DIR=$CRAY_PE_LIBSCI_PREFIX_DIR

# Debug the environmental variables
echo $NVIDIA_PATH
echo $LD_LIBRARY_PATH

# Make the dependencies
cd tools/toolchain
./install_cp2k_toolchain.sh --gpu-ver=A100 --enable-cuda --target-cpu=znver3 --mpi-mode=mpich --with-elpa=install --with-sirius=no -j 8 | tee install.log
cp install/arch/* ../../arch/
cd ../../

# Make the code
source ./tools/toolchain/install/setup
make -j 4 ARCH=local VERSION="ssmp psmp"
make -j 4 ARCH=local_cuda VERSION="ssmp psmp"
```

> Notes:
>  - The current version of SIRIUS prints a version message to stdout on launch, which breaks the ASE CP2K wrapper. Until that's resolved, we build without SIRIUS

Run CP2K using a version of the [GPU binding script from ALCF](https://docs.alcf.anl.gov/polaris/running-jobs/#binding-mpi-ranks-to-gpus)
modified to map multiple MPI ranks to each GPU:

```bash
#!/bin/bash -l
# Compute the number of ranks per GPU
num_gpus=4
local_size=$PMI_LOCAL_SIZE
ranks_per_gpu=$(( $local_size / $num_gpus ))

# need to assign GPUs in reverse order due to topology
# See Polaris Device Affinity Information https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html
gpu=$(( ($local_size - 1 - ${PMI_LOCAL_RANK}) / $ranks_per_gpu))
export CUDA_VISIBLE_DEVICES=$gpu
#echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}”
exec "$@"

```

An example mpiexec command:

```bash
mpiexec -n 16 --ppn 8 --cpu-bind depth --depth 4 -env OMP_NUM_THREADS=4 \
    /lus/eagle/projects/MOFA/lward/cp2k-2025.1/set_affinity_gpu_polaris.sh \
    /lus/eagle/projects/MOFA/lward/cp2k-2025.1/exe/local_cuda/cp2k.psmp
```

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
