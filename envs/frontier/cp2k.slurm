#!/bin/bash
#SBATCH -A mat291
#SBATCH -J cp2k-scaling
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 64
#SBATCH --gpus-per-node 8
#SBATCH --ntasks-per-gpu 1
#SBATCH --mem-per-gpu 64G
#SBATCH --gpu-bind closest

module reset
module load PrgEnv-gnu/8.5.0
module load craype-accel-amd-gfx90a
module load rocm/6.0.0
module load openblas
module load cmake
module load cray-mpich/8.1.31

export CRAYMPICH=/opt/cray/pe/mpich/8.1.31/ofi/cray/17.0
export LD_LIBRARY_PATH=$CRAYMPICH/lib:$CRAYMPICH/lib-abi-mpich:$LD_LIBRARY_PATH:/ccs/home/xyan11/libsymlink
export LD_RUN_PATH=$CRAYMPICH/lib:$CRAYMPICH/lib-abi-mpich:$LD_RUN_PATH:/ccs/home/xyan11/libsymlink
export LIBRARY_PATH=$CRAYMPICH/lib:$CRAYMPICH/lib-abi-mpich:$LD_LIBRARY_PATH:/ccs/home/xyan11/libsymlink
export CPATH=$CRAYMPICH/include:$CPATH

# only monitors GPUs on the head node, need further dev for all nodes GPU monitoring.
LOG_FILE="gpu_usage.log"
monitor_gpu() {
    echo "===== GPU Monitoring Started at $(date) =====" >> "$LOG_FILE"
    while true; do
        echo "===== GPU Utilization at $(date) =====" >> "$LOG_FILE"
        rocm-smi --showuse --showtemp --showclock >> "$LOG_FILE" 2>&1
        echo "" >> "$LOG_FILE"
        sleep 5
    done
}
monitor_gpu &
MONITOR_PID=$!

source /lustre/orion/mat291/scratch/xyan11/software/cp2k-v2025.1-20250520/tools/toolchain/install/setup
CP2KEXE=/lustre/orion/mat291/scratch/xyan11/software/cp2k-v2025.1-20250520/exe/local_hip/cp2k.psmp

ulimit -s unlimited
export OMP_NUM_THREADS=7
echo "entering `pwd` ..."
srun -N $SLURM_NNODES -n $((SLURM_GPUS_PER_NODE * SLURM_NNODES)) --gpus-per-node $SLURM_GPUS_PER_NODE --gpu-bind=closest $CP2KEXE -i cp2k.inp

kill $MONITOR_PID
wait $MONITOR_PID 2>/dev/null
echo "===== GPU Monitoring Stopped at $(date) =====" >> "$LOG_FILE"
