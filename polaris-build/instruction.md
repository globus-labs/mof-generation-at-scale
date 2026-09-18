# MOFA on Polaris with external LAMMPS and CP2K

Run setup and submission commands from the MOFA repository root on a Polaris
login node. This guide covers the one-node `run-polaris-local-smoke.sh` and
ten-node `run-polaris-repo-test.sh` workflows. MOFA, LAMMPS, and CP2K live in
separate Git checkouts. This repository includes the Polaris installation
scripts, but CP2K/LAMMPS sources, dependencies, Python environments, and compiled
outputs belong in their external checkouts, outside MOFA and its `deps/` directory.
MOFA calls their binaries through the wrappers in `bin/`.

## 1. Configure external installations

`bin/polaris-paths.sh` supplies the following defaults. Set any overrides
**before** sourcing it or running the installer; use absolute paths visible
from every compute node.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOFA_ENV` | `<repository>/mofa_env_py312` | Interactive MOFA environment; workflow scripts explicitly select this default. |
| `MOFA_CONDA_MODULE` | `conda` | Site Conda module used for installation and MOFA activation. |
| `LAMMPS_ROOT` | `/lus/eagle/projects/ChemGraph/thang/soft/lammps` | External LAMMPS installation. |
| `LAMMPS_VENV` | `$LAMMPS_ROOT/.venv` | Python environment belonging to that LAMMPS build. |
| `CP2K_ROOT` | `/lus/eagle/projects/ChemGraph/thang/soft/cp2k` | External CP2K installation. |
| `CP2K_DATA_DIR` | `$CP2K_ROOT/data` | CP2K basis, pseudopotential, and dispersion data. |

For the existing installations, load the defaults:

```bash
source bin/polaris-paths.sh
```

To use another matching installation, set its roots first. If `LAMMPS_VENV` or
`CP2K_DATA_DIR` was already exported, update it too: changing a root does not
replace an explicitly set dependent path. Submission examples appear below.

### Collaborator access prerequisite

The defaults describe the owner's current setup, not a generally accessible
installation. As inspected on 2026-09-18, `$LAMMPS_VENV/bin/python` resolves
under `/home/tdpham2/.local/share/uv/python/`, and LAMMPS's library search paths
also reference that Python installation. `/home/tdpham2` has mode `700` with
no collaborator ACL. Another account cannot use that interpreter just because
the LAMMPS directory is on Eagle. Access to the ChemGraph project filesystem
is also required.

A collaborator needs an accessible Python runtime and matching LAMMPS build,
then must set `LAMMPS_ROOT`/`LAMMPS_VENV` accordingly. The build recipe below creates
Python inside the external LAMMPS checkout, avoiding a private-home interpreter.
It does not change the existing installation or its permissions unless you
explicitly select that checkout. Verify filesystem access from the collaborator's
account before attempting validation or submitting jobs.

### Build separate installations

Skip this subsection if you already have accessible, compatible installations.
The scripts shipped here are
[build-cp2k-polaris.sh](build-cp2k-polaris.sh) and
[build-lammps-polaris.sh](build-lammps-polaris.sh). They require pre-cloned sources
at the revisions recorded from the September 2026 installations:

| Software | Source revision | Build |
| --- | --- | --- |
| CP2K | [34658f03b9867a3335ace1b4af7b7736c33523bb](https://github.com/harikrishna1410/cp2k/commit/34658f03b9867a3335ace1b4af7b7736c33523bb), `polaris_build_sep_2026` branch of the CP2K fork | CP2K 2025.2 with Polaris/CUDA 13 patches, A100, ELPA, Cray MPI/LibSci. |
| LAMMPS | [d51bbd4983a26e2da6f1550e1b41592690b02a90](https://github.com/lammps/lammps/commit/d51bbd4983a26e2da6f1550e1b41592690b02a90), upstream | Kokkos/CUDA, ML-IAP, EXTRA-MOLECULE, Python, and `BUILD_MPI=OFF`. |

CP2K's fork supplies CUDA 13 compatibility changes for FFT, ELPA, and DBCSR;
an unpatched 2025.2 checkout is not equivalent. Both scripts check the revision
before building. These replace the old CP2K 2025.1/CUDA 12.8 and LAMMPS
`stable_22Jul2025_update5` recipes.

On a login node, choose a writable project directory visible from compute nodes
and accessible to the intended users. Edit the first path below. Keep the MOFA
checkout as your current directory:

```bash
export MOFA_SOFTWARE_ROOT=/lus/eagle/projects/ChemGraph/YOUR_DIRECTORY/soft
export CP2K_ROOT="$MOFA_SOFTWARE_ROOT/cp2k"
export LAMMPS_ROOT="$MOFA_SOFTWARE_ROOT/lammps"
export CP2K_DATA_DIR="$CP2K_ROOT/data"
export LAMMPS_VENV="$LAMMPS_ROOT/.venv"
mkdir -p "$MOFA_SOFTWARE_ROOT"

git clone --branch polaris_build_sep_2026 https://github.com/harikrishna1410/cp2k.git "$CP2K_ROOT"
git -C "$CP2K_ROOT" checkout --detach 34658f03b9867a3335ace1b4af7b7736c33523bb
git -C "$CP2K_ROOT" submodule update --init --recursive

git clone --filter=blob:none --no-checkout https://github.com/lammps/lammps.git "$LAMMPS_ROOT"
git -C "$LAMMPS_ROOT" checkout --detach d51bbd4983a26e2da6f1550e1b41592690b02a90
git -C "$LAMMPS_ROOT" submodule update --init --recursive

qsub -v CP2K_ROOT polaris-build/build-cp2k-polaris.sh
qsub -v LAMMPS_ROOT,LAMMPS_VENV polaris-build/build-lammps-polaris.sh
```

These are independent one-node, one-hour `debug` jobs using allocation `ChemGraph`.
Use `qsub -A YOUR_ALLOCATION` when needed; choose an appropriate queue and walltime
if a fresh dependency build takes longer. Both jobs download packages, so they
need network access to the upstream dependency/package repositories. Do not
start MOFA until both jobs finish successfully. Their merged PBS logs are
`cp2k-build.o<jobid>` and `lammps-build.o<jobid>` in the submission directory.

| Build variable | Default / requirement |
| --- | --- |
| `CP2K_ROOT`, `LAMMPS_ROOT` | Explicit absolute paths to external source checkouts; the build scripts do not select the owner's runtime defaults. Paths inside MOFA are rejected. |
| `LAMMPS_VENV` | `$LAMMPS_ROOT/.venv`; must also be outside MOFA. Pass the same path to runtime jobs if overridden. |
| `MOFA_CONDA_MODULE` | `conda`; used to create LAMMPS's separate Python installation. |
| `CP2K_BUILD_JOBS` | `8` compilation processes. |
| `LAMMPS_BUILD_JOBS` | `32` compilation processes. |
| `CP2K_CLEAN_TOOLCHAIN` | `0` reuses previously built dependencies; `1` moves them aside and rebuilds them. Use `1` after a compiler or CUDA change. |

Forward optional settings with `qsub -v`, for example
`qsub -v CP2K_ROOT,CP2K_BUILD_JOBS=16,CP2K_CLEAN_TOOLCHAIN=1 polaris-build/build-cp2k-polaris.sh`.
`bash polaris-build/build-cp2k-polaris.sh --help` and the corresponding LAMMPS
command work without an allocation or loading modules.

LAMMPS creates a Conda-forge Python 3.12.13 installation in `$LAMMPS_ROOT/.python`,
then a dedicated venv at `$LAMMPS_VENV`. It installs the recorded direct package
versions in [lammps-requirements.txt](lammps-requirements.txt), including Torch
2.5.0, MACE 0.3.13, CuEquivariance, and CuPy. These Python packages use CUDA 12
wheels; LAMMPS itself compiles with the CUDA 13 toolkit. Keep `.python`, the venv,
and the build tree in place together: Python/library paths are embedded in the
build. The script refuses existing Python environments and build directories;
use a fresh external checkout after a failed partial installation. It records
the resolved packages, modules, and source revision in `mofa-lammps-*.txt`.

CP2K builds serial/OpenMP and MPI executables under both `exe/local/` and
`exe/local_cuda/`, including the ASE shell binaries. It preserves old CP2K
object, library, and executable directories with a `.prev-<timestamp>-<jobid>`
suffix. With `CP2K_CLEAN_TOOLCHAIN=1`, it also preserves the previous toolchain
build/install trees. Rebuilding changes the selected installation, so use a new
checkout if existing jobs still depend on it. Toolchain output is recorded in
`$CP2K_ROOT/tools/toolchain/install.log`; modules and source revision are saved
in `mofa-cp2k-*.txt`.

The scripts verify executable/package availability and, for CP2K, runtime
linkage. The recipes were updated from the installed build settings; a fresh
full compilation has not been rerun as part of this repository update. Complete
model preparation, preflight, and the smoke workflow below before scaling.
Continue to pass your selected `CP2K_ROOT`, `CP2K_DATA_DIR`, `LAMMPS_ROOT`, and
`LAMMPS_VENV` with `qsub -v` when submitting MOFA; build-job overrides do not
change the repository's runtime defaults.

### Executables and runtime modules

The LAMMPS wrapper activates `$LAMMPS_VENV/bin/activate` and executes
`$LAMMPS_ROOT/build-mliap-no-mpi/lmp`. The build disables LAMMPS MPI support;
each MOFA worker launches a separate process. Its Python packages and shared
libraries must match the build.

The CP2K wrapper executes `$CP2K_ROOT/exe/local_cuda/$CP2K_BINARY`.
The launch configuration supplies `CP2K_BINARY`; it is not a workflow tuning knob:

- `cp2k_shell.ssmp`: local smoke test.
- `cp2k_shell.psmp`: distributed MOFA tasks (wrapper default).
- `cp2k.psmp`: standalone input-file tests.

Both wrappers load the external builds' runtime modules in child processes:
`PrgEnv-gnu`, `gcc-native/14`, Cray LibSci/FFTW, `craype-accel-nvidia80`, and
`cudatoolkit-standalone/13.0.1`. The installed Cray MPI GPU transport library
requires `libcudart.so.13`. This replaces the older repository-local runtime
module stack. Changing binary paths does not change these modules; another
installation must be compatible with them. The installation scripts above source
this same module configuration from `bin/polaris-simulation-env.sh`.

## 2. Prepare the models

The installer validates models at the end, so prepare the ML-IAP model first.
The DiffLinker checkpoint is supplied under `models/geom-300k/`; the generated
MACE model is ignored by Git and must be prepared in each fresh checkout.
Use the external LAMMPS environment for conversion, so the saved model matches
its ML-IAP runtime. On the login node, with external paths configured:

```bash
(
    set -euo pipefail
    source bin/polaris-simulation-env.sh
    source "$LAMMPS_VENV/bin/activate"
    export PYTHONNOUSERSITE=1
    cd input-files/mace
    if [[ ! -f mace-mp0_medium ]]; then
        wget https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model \
            -O mace-mp0_medium
    fi
    if [[ ! -f mace-mp0_medium-mliap_lammps.pt ]]; then
        python create_lammps_model.py mace-mp0_medium --dtype float32 --format mliap
    fi
)
```

The converter needs the MACE/CuEquivariance packages already installed with
LAMMPS. Do not install those build dependencies into MOFA or overwrite an
existing validated model to follow this guide. The output used by both workflow
scripts is `input-files/mace/mace-mp0_medium-mliap_lammps.pt`.
The `create_mliap_model.py` converter called by `get-macemp-0a.sh` is the older
no-CuEquivariance fallback; the command above is the external-runtime recipe.

## 3. Create the MOFA environment

```bash
module use /soft/modulefiles
module load conda
conda activate base
bash create_conda.sh --prefix "$PWD/mofa_env_py312"
```

The installer uses `MOFA_CONDA_MODULE` (default `conda`). It targets
`<repository>/mofa_env_py312` independently of any inherited `MOFA_ENV`.
`--prefix PATH` selects another installation target; relative paths are resolved
from the calling directory. It refuses every existing target, including partial
installations, and never updates or removes it. `bash create_conda.sh --help`
does not load modules or install packages.

The stack pins Python 3.12.13, Torch 2.5.0+cu124, NumPy 1.26.4, Parsl 2026.7.27,
MACE 0.3.13, Colmena 0.7.2, and redis-py 5.3.1. Python and Parsl match the recorded
August run; the complete historical package set was not recovered. CPU checks
and standalone simulations do not establish full workflow success.

Conda supplies Redis server, MongoDB, ChargeMol, and RASPA2. Pip supplies
`openbabel-wheel==3.1.1.23` as the only Open Babel binding. Do not also install
Conda's `openbabel` or pip's `openbabel`. The CUDA 12.4 Torch wheel uses the
PyTorch index with PyPI available for dependencies. MOFA is installed with
`python -m pip install --no-deps -e .` after the manifest's dependencies.
LAMMPS uses its own Python environment; no additional venv is needed inside MOFA.

Before installation the script disables Python user-site packages, clears
inherited Python paths, resets modules, and unloads XALT in the installer process.
Temporary build files use `/tmp`. Logs are saved under
`.runtime-cache/mofa-env-build.*.log`. The installed environment records
`mofa-environment.yml`, `mofa-conda-explicit.txt`, `mofa-pip-freeze.txt`,
`mofa-modules.txt`, and `mofa-build-log.txt` before validation.

If package installation itself fails, inspect the log and use a new, unused
`--prefix` for another attempt, retaining the failed environment for diagnosis.
If installation completed and only validation failed (for example, an external
path or model was missing), fix the prerequisite and rerun the checks below in
that same environment. Do not rerun the installer on an existing prefix.
Alternative prefixes can be validated interactively; the supplied workflow
scripts still force `mofa_env_py312` and do not select an alternative via PBS.

## 4. Check the installation

For initial checks, or to retry validation without reinstalling:

```bash
# Select the actual installed prefix for interactive checks.
export MOFA_ENV="$PWD/mofa_env_py312"
source bin/activate-mofa-polaris.sh
python bin/check-polaris.py --models --external
python run_parallel_workflow.py --help
python tests/test_polaris_environment.py
python tests/test_polaris_launchers.py
python tests/test_mace_import_order.py
```

Preflight checks the interpreter path, exact Python/package pins, `pip check`,
MOFA imports, native commands, MPI linkage, external paths, and model inputs.
`--models` loads DiffLinker on CPU, populates MACE's foundation-model cache,
and tests MACE energy/forces in fresh processes with both Open Babel and
`mofa.model` imported first. This covers the native import-order crash seen in
the previous Python 3.10/Torch 2.1 environment without changing application
imports. CuPy belongs to the external LAMMPS environment.

On compute nodes, `--gpu` also performs a Torch CUDA calculation; both job
scripts run `--gpu --external` before starting services. Do not run GPU CP2K
on a login node, even with `--version`.

Preflight validates the runtime. Full workflow validation requires running the
smoke simulation below and inspecting its results.

Sample inputs and standalone tests are also provided in
[cp2k-test](../cp2k-test/README.md) and [lammps-test](../lammps-test/README.md).
They use the same external runtime wrappers. The LAMMPS examples include their
structure data and use the MACE model prepared above. Test outputs stay in
ignored local run directories; only inputs, test code, and usage are shared.

## 5. Run simulations

Once the prerequisites and login-node checks pass, no manual environment
activation or exports are needed for default submissions. Submit from the
repository root so PBS supplies the correct `PBS_O_WORKDIR`:

```bash
qsub run-polaris-local-smoke.sh
# Inspect the smoke's simulation results before scaling:
qsub -v MOFA_SIMULATION_BUDGET=8 run-polaris-repo-test.sh
```

| Setting | `run-polaris-local-smoke.sh` | `run-polaris-repo-test.sh` |
| --- | --- | --- |
| Nodes / queue | 1 / `debug` | 10 / `debug-scaling` |
| Walltime / allocation | 1 hour / `ChemGraph` | 1 hour / `ChemGraph` |
| Default simulation budget | 8 | `-1` (unlimited) |
| Samples / generation batch | 8 / 8 | 128 / 64 |
| MD timesteps | 100 | 1,000 |
| DFT optimization steps | 1 | 2 |
| RASPA cycles | 100 | 10,000 |
| CP2K | Serial shell on the GPU executor | Two nodes/task, four ranks/node, eight threads/rank |
| LAMMPS | One GPU worker | Two clients/GPU, with MPS |

`MOFA_SIMULATION_BUDGET` overrides the budget in either script. Without an
override, `qsub run-polaris-repo-test.sh` runs with an unlimited budget until
another termination condition, such as walltime. Budget 8 is a workflow
scheduling limit, not a guarantee of eight completed MOFs or an exact count of
all attempts. Inspect each stage's results.

### Environment changes and PBS overrides

Compared with the previous repository-local instructions:

- `mofa_env` (Python 3.10/Torch 2.1) is replaced by `mofa_env_py312`
  (Python 3.12.13/Torch 2.5.0+cu124). Both scripts **overwrite `MOFA_ENV`**
  before activation, even if supplied with `qsub -v MOFA_ENV=...` or inherited
  via `qsub -V`. Submit these scripts without a `MOFA_ENV` override.
- LAMMPS and CP2K now use the external roots in section 1 instead of `deps/`.
  `LAMMPS_ROOT`, `LAMMPS_VENV`, `CP2K_ROOT`, `CP2K_DATA_DIR`, and
  `MOFA_CONDA_MODULE` remain supported overrides. Distributed workers receive
  the same selected paths as the driver, and RASPA uses the selected MOFA prefix.
- `MOFA_SIMULATION_BUDGET` is now available instead of editing each script's
  `--simulation-budget` argument. It defaults to 8 for smoke and -1 for scaling.

Exports in the submission shell alone are not forwarded to PBS. Use `qsub -v`
for selected overrides. For example, replace these illustrative paths with
accessible installations matching the runtime modules:

```bash
qsub -v LAMMPS_ROOT=/shared/lammps,CP2K_ROOT=/shared/cp2k \
    run-polaris-local-smoke.sh
# Separate LAMMPS Python environment and CP2K data locations:
qsub -v MOFA_SIMULATION_BUDGET=8,LAMMPS_ROOT=/shared/lammps,LAMMPS_VENV=/shared/lammps-env,CP2K_ROOT=/shared/cp2k,CP2K_DATA_DIR=/shared/cp2k-data \
    run-polaris-repo-test.sh
```

The first example derives `.venv` and `data` from the supplied roots. Alternatively,
export all selected variables and pass their names with `qsub -v NAME,...`.
Use `qsub -A YOUR_ALLOCATION ...` to override the script's allocation when needed.

### Variables managed by the scripts

`PBS_O_WORKDIR`, `PBS_JOBID`, and `PBS_NODEFILE` come from PBS; do not set them
manually to launch a workflow on the login node. Activation loads the selected
Conda module and `cray-mpich-abi`, sets `PYTHONNOUSERSITE=1` and
`PYTHONFAULTHANDLER=1`, and clears `PYTHONHOME`/`PYTHONPATH`. It sets
`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS`, and `OMP_NUM_THREADS` to 1;
the distributed CP2K launcher separately sets eight OpenMP threads per rank.
`TMPDIR=/tmp` keeps multiprocessing socket paths short. `MPLCONFIGDIR` and
`XDG_CACHE_HOME` point into `.runtime-cache/$PBS_JOBID/`.

The wrappers select CUDA/MPI runtime variables and GPU affinity. Users do not
need to export `CUDA_VISIBLE_DEVICES`, `CP2K_BINARY`, or thread settings for
these submissions. The scripts start and stop Redis; the scaling script also
manages MPS. Redis persistence is disabled and its working directory is
`.runtime-cache/$PBS_JOBID/redis`, avoiding an old repository `dump.rdb`.

### Production entrypoint

After validating the workflow, `qsub run-polaris-repo.sh` requests **16 nodes**
in `prod` for **3 hours** using allocation **IQC**, with an unlimited default
simulation budget. It has the same environment selection and override rules.
This differs from the ChemGraph smoke and repository-test jobs; override the
allocation with `qsub -A` if needed.

## 6. Inspect results and known limitations

PBS stdout/stderr use job names `mofa-smoke` and `mofa-test`. Redis logs are
`redis-$PBS_JOBID.log`. Workflow results are under `run/parallel-local-smoke-*`
and `run/parallel-polaris-repo-*`, including `run.log`, `params.json`, Parsl logs,
and task result JSON files. Inspect successful generation and MACE relaxation,
completed MD trajectories, CP2K and ChargeMol results, and stored RASPA uptake
before scaling. `MOFAThinker completed` alone is not proof of successful science.

Local testing of the external CP2K/ELPA build reproduced a structure-dependent
GPU solver crash. Block size 32 worked for the tested crash cases but has not
been applied to production MOFA inputs. It did not resolve every SCF convergence
failure. Because `IGNORE_CONVERGENCE_FAILURE` can let CP2K return normally with
unconverged SCF, inspect convergence explicitly. The production workaround and
stricter scientific acceptance remain follow-up work.
