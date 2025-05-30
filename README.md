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

If solving is slow try updating to the newest version of conda and using the `libmamba` solver:

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create --file envs/environment-cpu.yml
```
## Running MOFA


The `run_parallel_workflow.py` script defines an HPC workflow using MOFA. 

First set up the required input files by running `assemble-inputs.ipynb` in `input_files/zn-paddle-pillar`.
and `get-macemp-0a.sh` in `inputs-files/mace`.

The run scripts available in the root directory include input argument configurations appropriate for different systems
at different scales.
For example, `run-polaris-test.sh` is configured for a short run on Polaris using a small number of nodes.

Each run will produce a run directory in `run` named using the start time and a hash of the run parameters.

The run directory contains the following files:

- `run.log`: The log messages produced during execution
- `params.json`: The arguments provided to the run script
- `all-ligands.csv`: A CSV file with the geometries of the generated ligands in XYZ format, 
  if they passed all validation screens, and the SMILES string (if available).
- `db`: A MongoDB database folder. Convert to JSON format using `./bin/dump_data.sh`
- `*-results.json`: Summaries of different types of computations. See visualizations in `scripts` for examples 
  on reading them.
