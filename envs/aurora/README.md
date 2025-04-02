# Installing on Aurora

Build MOFA Aurora using environments prepared by others.

## Python Environment

Create a virtual environment based on the
[`frameworks` module on Aurora](https://docs.alcf.anl.gov/aurora/data-science/python/).

This enviornment contains PyTorch versions that are suited for
the Intel GPUs on Aurora and many of the math libraries we require
for MOFA.

```bash
# Make a virtual environment from the frameworks
module load frameworks/2024.2.1_u1
python3 -m venv ./venv --system-site-packages
source ./venv/bin/activate

# Install the MOFA stuff
pip install -e .
```

## Launchers

`mpiexec` on Aurora needs to be handled with care. 
We have a few alternatives to using it for our launching needs:

1. `envs/aurora/parallel.sh` for deploying LAMMPS and helper tasks on 
   to the same node. You otherwise run into problems with ["AC limits"](ihttps://github.com/argonne-lcf/AuroraBugTracking/issues/12).
   Edit the path to the Python environment inside this file before launching

## Databases

Install Redis and MongoDB with Anaconda.

```bash
conda env create --file envs/aurora/environment.yml -p ./conda-env
```

Conda will produce a directory holding the executables
for MongoDB and Redis in `./conda-env/bin/`

## Simulation Codes

### LAMMPS

Build LAMMPS using ALCF's tweak of the
[ML-MACE version of LAMMPS](https://github.com/ACEsuit/lammps/tree/mace)
and the Makefile provided in this folder.

> NOTE: Remove the use of MPI to determine local rank as well. See Logan's version

Include the path to the Torch libraries when running LAMMPS (below)

```
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH
```

Invoke LAMMPS using Kokkos functions.

```
/lus/flare/projects/MOFA/lward/lammps-kokkos/src/lmp_macesunspotkokkos -k on g 1 -sf kk
```

# PWDFT

TBD

