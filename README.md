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

For MOFA on Polaris, follow the
[Polaris setup guide](polaris-build/instruction.md#1-configure-external-installations).
CP2K and LAMMPS live in separate external checkouts; MOFA calls their binaries.
The guide includes [installation scripts and source revisions](polaris-build/instruction.md#build-separate-installations)
for building them outside this repository, as well as configuration for existing installations.

If solving is slow try updating to the newest version of conda and using the `libmamba` solver:

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create --file envs/environment-cpu.yml
```
## Running MOFA


The `run_parallel_workflow.py` script defines an HPC workflow using MOFA. 

First set up the required input files by running `assemble-inputs.ipynb` in `input-files/zn-paddle-pillar`.
For Polaris, prepare MACE using the external LAMMPS environment as described in
the [model preparation instructions](polaris-build/instruction.md#2-prepare-the-models).

The run scripts available in the root directory include input argument configurations appropriate for different systems
at different scales.
For the current Polaris setup, submit from the repository root after completing setup:

```bash
qsub run-polaris-local-smoke.sh
# After reviewing the smoke results, request a limited simulation budget:
qsub -v MOFA_SIMULATION_BUDGET=8 run-polaris-repo-test.sh
```

These scripts use one and ten nodes, respectively, and both select `mofa_env_py312`.
See [running simulations and environment overrides](polaris-build/instruction.md#5-run-simulations)
for prerequisites, PBS settings, and the changes from the old `mofa_env`/`deps` setup.
The guide also records the shared LAMMPS access prerequisite and known CP2K limitations.

Each run will produce a run directory in `run` named using the start time and a hash of the run parameters.

The run directory contains the following files:

- `run.log`: The log messages produced during execution
- `params.json`: The arguments provided to the run script
- `all-ligands.csv`: A CSV file with the geometries of the generated ligands in XYZ format, 
  if they passed all validation screens, and the SMILES string (if available).
- `db`: A MongoDB database folder. Convert to JSON format using `./bin/dump_data.sh`
- `*-results.json`: Summaries of different types of computations. See visualizations in `scripts` for examples 
  on reading them.
