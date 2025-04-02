#! /bin/bash
#  Wrapper over GNU Parallel to faciliate using it in Parsl

sshfile=$1
exe=`which $2`
args=${@:3}

command=$(cat <<-END
cd $PBS_O_WORKDIR

# General environment variables
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
conda deactivate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Needed for LAMMPS
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

$exe $args
END
)

parallel --env _ --nonall --sshloginfile $sshfile "$command"
