# Installing on Polaris

Polaris is a system with NVIDIA GPUs and AMD Zen3 processors. 
Install MOFA and its dependencies by following these guides.

Start by building an Anaconda environment using the YAML file for polaris:

```bash
conda env create --file envs/environment-polaris.yml -p ./env
```

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

We need a copy of LAMMPS that uses GPUs and not MPI,
and is built with the [ML-IAP module](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html).

Download any recent version of LAMMPS. We cloned the main branch of the LAMMPS repository.

```
git clone git@github.com:lammps/lammps.git
```

Compile LAMMPS using the following build script (editing paths to libtorch appropriately):

```bash
# Make the build environment
module reset
module use /soft/modulefiles
module load spack-pe-base cmake
module load gcc
module load cudatoolkit-standalone/12.8
module load cray-python
module list

# Install python deps
which python
if [ ! -e venv ]; then
  python -m venv ./venv
fi
source venv/bin/activate
pip install -r requirements.txt
which python


mkdir -p build-mliap
cd build-mliap

export LDFLAGS="-Wl,--allow-multiple-definition"
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_C_COMPILEr=NVC++ \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_INSTALL_PREFIX=$(pwd) \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_MPI=OFF \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DCMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
    -DKokkos_ARCH_ZEN3=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-SNAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_PYTHON=ON \
    ../cmake

make -j 8
make install
make install-python
```

The build script installs a Python virtual environment that may have different dependencies than the MOFA environment.
Thus, create a shell script which activates the LAMMPS environment before launching the executable

```bash
#! /bin/bash
module use /soft/modulefiles
module load cudatoolkit-standalone/12.8
home=/lus/eagle/projects/MOFA/lward/lammps-main/build-mliap
source $home/../venv/bin/activate
$home/lmp $@
```

