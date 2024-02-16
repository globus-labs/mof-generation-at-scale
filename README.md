# MOF Generation on HPC

[![CI](https://github.com/globus-labs/mof-generation-at-scale/actions/workflows/python-package-conda.yml/badge.svg?branch=main)](https://github.com/globus-labs/mof-generation-at-scale/actions/workflows/python-package-conda.yml)
[![Coverage Status](https://coveralls.io/repos/github/globus-labs/mof-generation-at-scale/badge.svg?branch=main)](https://coveralls.io/github/globus-labs/mof-generation-at-scale?branch=main)

Create new MOFs by combining generative AI and simulation on HPC.

## Installation

The requirements for this project are defined using Anaconda. 

Install the environment file appropriate for your system with a command similar to:

```bash
conda env create --file envs/environment-cpu.yml --force
```

## Running MOFA

The `run_serial_workflow.py` script defines a workflow using MOFA. 

Set up the required input files by running `0_assemble-inputs.ipynb` in `input_files/zn-paddle-pillar`.

Then invoke the workflow by running `example-run.sh`

The code will produce a run directory in `run` named using the start time and a hash of the run parameters.

The run directory contains the following files:

- `run.log`: The log messages produced during execution
- `params.json`: The arguments provided to the run script
- `all-ligands.csv`: A CSV file with the geometries of the generated ligands in XYZ format, 
  if they passed all validation screens, and the SMILES string (if available).
 