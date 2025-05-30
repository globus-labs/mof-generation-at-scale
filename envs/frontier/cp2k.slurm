#!/bin/bash
#SBATCH -A mat291
#SBATCH -J test-cp2k
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node 8
#SBATCH --ntasks-per-gpu 1
#SBATCH --mem-per-gpu 64G
#SBATCH --gpu-bind closest

module load PrgEnv-gnu/8.5.0
module load craype-accel-amd-gfx90a
module load rocm/6.0.0
module load openblas
module load cmake
module load boost/1.79.0

source /lustre/orion/mat291/scratch/xyan11/software/cp2k-v2025.1/tools/toolchain/install/setup

ulimit -s unlimited
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=7
srun -N 1 -n 8 --gpus-per-node 8 --gpu-bind=closest /lustre/orion/mat291/scratch/xyan11/software/cp2k-v2025.1/exe/local_hip/cp2k.psmp -i cp2k.inp
