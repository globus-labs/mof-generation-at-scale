# Installing on Aurora

Build MOFA Aurora using environments prepared by others.

## Python Environment

Create a virtual environment based on the
[`frameworks` module on Aurora](https://docs.alcf.anl.gov/aurora/data-science/python/).

This enviornment contains PyTorch versions that are suited for
the Intel GPUs on Aurora and many of the math libraries we require
for MOFA.

## Databases

Install Redis and MongoDB with Anaconda.

```bash
conda env create --file envs/aurora/environment.yml -p ./conda-env
```

Conda will produce a directory holding the executables
for MongoDB and Redis in `./conda-env/bin/`

## Simulation Codes

TBD. We are using CPU-only versions of the codes for now
