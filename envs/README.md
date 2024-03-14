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

> Please don't add OpenMP or MPI to Pkg GPU or Kokkos: in Pkg GPU multiple OMP threads worsen the performance; in Kokkos multiple OMP threads has no effect on performance. 

The following recipe is for small systems and short simulations (https://docs.lammps.org/Build_settings.html#size-of-lammps-integer-types-and-size-limits) with mixed precision mode (https://docs.lammps.org/Build_extras.html#gpu-package). Tested on ALCF Polaris and NCSA Delta with 1 A100 GPU on 2x2x2 MOFs. 
```bash
#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug-scaling
#PBS -A QuantumDS
#PBS -N gpu-compile
#PBS -M xyan11@anl.gov
#PBS -m abe
#PBS -k doe
#PBS -j oe

lscpu
free -h
ulimit -s unlimited
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
cd build-gpu-nvhpc-mix-nompi-noomp

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

export PATH=/soft/buildtools/cmake/cmake-3.23.2/cmake-3.23.2-linux-x86_64/bin:/opt/anaconda3x/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/opt/cray/pe/bin
export LD_LIBRARY_PATH=
export LIBRARY_PATH=
export MANPATH=/usr/share/lmod/lmod/share/man:/usr/local/man:/usr/share/man:/opt/c3/man:/opt/pbs/share/man:/opt/clmgr/man:/opt/sgi/share/man:/opt/clmgr/share/man:/opt/clmgr/lib/cm-cli/man
export OPAL_PREFIX=
export CPATH=

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nvshmem/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/openmpi4/openmpi-4.0.5/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/extras/qd/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/extras/qd/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/share/llvm/bin:$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/bin:$PATH

export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nvshmem/lib:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/lib:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/openmpi4/openmpi-4.0.5/lib:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/lib64:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/lib:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/extras/qd/lib:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/extras/CUPTI/lib64:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/lib64:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/lib64:$LIBRARY_PATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/lib64/stubs:$LIBRARY_PATH

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nvshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/openmpi4/openmpi-4.0.5/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/lib64/stubs:$LD_LIBRARY_PATH

export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nvshmem/include:$CPATH
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/include:$CPATH
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/openmpi4/openmpi-4.0.5/include:$CPATH
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/extras/qd/include/qd:$CPATH
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/include:$CPATH
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/include:$CPATH
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/include:$CPATH

export OPAL_PREFIX=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/openmpi4/openmpi-4.0.5
export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/man:$MANPATH


cmake ../cmake -DCMAKE_BUILD_TYPE=release \
-DCMAKE_CUDA_COMPILER=nvcc \
-DCMAKE_C_COMPILER=nvc++ \
-DCMAKE_CXX_COMPILER=nvc++ \
-DCMAKE_CXX_STANDARD=14 \
-DLAMMPS_MEMALIGN=64 \
-DLAMMPS_SIZES=smallsmall \
-DPKG_MISC=on \
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
-DCUDA_MPS_SUPPORT=no \
-DBUILD_OMP=no \
-DBUILD_MPI=no \
-DCUDA_NVCC_FLAGS="-std=c++14 -allow-unsupported-compiler -Xcompiler" \
-DCMAKE_CXX_FLAGS="-std=c++14 -DCUDA_PROXY" 


make -j 32
```
