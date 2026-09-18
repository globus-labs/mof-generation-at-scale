# Standalone LAMMPS test on Polaris

The bundled inputs are `in.lj` (a small Lennard-Jones GPU baseline) and
`in.lammps`/`data.lmp` (a 976-atom MACE example). The MACE input refers to
`input-files/mace/mace-mp0_medium-mliap_lammps.pt` relative to this directory.
Prepare that model and the external LAMMPS installation using the
[Polaris setup guide](../polaris-build/instruction.md). Model weights are not
included here.

Submit from the MOFA repository root:

```bash
qsub lammps-test/run-polaris.sh
# Select your external installation when using different paths:
qsub -v LAMMPS_ROOT=/shared/lammps,LAMMPS_VENV=/shared/lammps/.venv \
    lammps-test/run-polaris.sh
```

The job requests one node for one hour in `debug`, allocation `ChemGraph`. It
uses the external LAMMPS Python environment to run the controller, and invokes
the binary through `bin/run-lammps-polaris.sh`. The default inputs are bundled
in this directory; no files from the external installation's `test/` directory
are needed.

The test runs the LJ baseline and MACE without MPS, then MACE with one through
four clients per GPU under MPS. Short MACE copies use five minimization
iterations and 20 MD steps. Source files are preserved. Checks cover completion,
finite thermodynamic output, GPU health, and actual overlapping MPS clients.
Use a fresh allocation: the test refuses to change an existing MPS service.

`LAMMPS_TEST_MAX_PER_GPU` limits concurrency (default 4). Set
`LAMMPS_TEST_FULL=1` to use the sample's original 100 minimization iterations and
100 MD steps, checking only the selected MPS concurrency after the baselines.
`LAMMPS_TEST_DIR` selects another directory containing compatible `in.lammps`
and `data.lmp` files. Forward these overrides with `qsub -v`.

Results, logs, and copied inputs are written under ignored
`run/lammps-diagnostic-<job>-<timestamp>/` directories. Inspect `summary.json`
and per-client logs locally; generated results are not included in Git.

Without an allocation, validate the prepared sample and run offline checks:

```bash
mofa_env_py312/bin/python lammps-test/diagnose.py --prepare-only
mofa_env_py312/bin/python tests/test_lammps_diagnostic.py
```

`--prepare-only` requires the prepared model but performs no GPU operations.
Passing these standalone tests does not validate the complete MOFA workflow.
