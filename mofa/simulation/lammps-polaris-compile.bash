module purge

export PATH=/soft/buildtools/cmake/cmake-3.23.2/cmake-3.23.2-linux-x86_64/bin:/opt/anaconda3x/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/dbhome/db2cat/sqllib/bin:/dbhome/db2cat/sqllib/adm:/dbhome/db2cat/sqllib/misc:/dbhome/db2cat/sqllib/gskit/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/opt/cray/pe/bin
export LD_LIBRARY_PATH=/dbhome/db2cat/sqllib/lib64:/dbhome/db2cat/sqllib/lib64/gskit:/dbhome/db2cat/sqllib/lib32
export LIBRARY_PATH=/dbhome/db2cat/sqllib/lib64:/dbhome/db2cat/sqllib/lib64/gskit:/dbhome/db2cat/sqllib/lib32
export MANPATH=/usr/share/lmod/lmod/share/man:/usr/local/man:/usr/share/man:/opt/c3/man:/opt/pbs/share/man:/opt/clmgr/man:/opt/sgi/share/man:/opt/clmgr/share/man:/opt/clmgr/lib/cm-cli/man
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


cd /grand/projects/RAPINS/xiaoliyan/lmp20230802stb/build-gpu-nvhpc
cmake ../cmake -DCMAKE_BUILD_TYPE=release \
-DCMAKE_C_COMPILER=nvc++ \
-DCMAKE_CXX_COMPILER=nvc++ \
-DCMAKE_FORTRAN_COMPILER=nvfortran \
-DCMAKE_CXX_STANDARD=17 \
-DLAMMPS_MEMALIGN=64 \
-DLAMMPS_SIZES=bigbig \
-DPKG_MISC=on \
-DPKG_ML-SNAP=on \
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
-DPKG_GPU=on \
-DGPU_API=cuda \
-DGPU_PREC=double \
-DGPU_ARCH=sm_80 \
-DGPU_DEBUG=yes \
-DCUDA_MPS_SUPPORT=yes \
-DBUILD_OMP=yes \
-DBUILD_MPI=yes \
-DCUDA_NVCC_FLAGS="-std=c++17 -Xcompiler -fopenmp --allow-unsupported-compiler" \
-DCMAKE_CXX_FLAGS="-std=c++17 -DCUDA_PROXY -fopenmp -g -O3 -mp" \
-DCMAKE_C_FLAGS="-std=c++17 -DCUDA_PROXY -fopenmp -g -O3 -mp" \
-DCMAKE_FORTRAN_FLAGS="-std=c++17 -DCUDA_PROXY -fopenmp -g -O3 -mp"

make -j 16

#LMPEXE=/grand/projects/RAPINS/xiaoliyan/lmp20230802stb/build-gpu-nvhpc/lmp
