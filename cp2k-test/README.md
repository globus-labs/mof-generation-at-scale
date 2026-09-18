# Standalone CP2K test on Polaris

`cp2k.inp` contains a 112-atom sample structure. The test scripts call the
external CP2K installation through `bin/run-cp2k-polaris.sh`; build and configure
it using the [Polaris setup guide](../polaris-build/instruction.md).

From the MOFA repository root, submit the executable/ASE-shell test:

```bash
qsub cp2k-test/run-diagnostic-polaris.sh
# Select your external installation when using different paths:
qsub -v CP2K_ROOT=/shared/cp2k,CP2K_DATA_DIR=/shared/cp2k/data \
    cp2k-test/run-diagnostic-polaris.sh
```

This requests two nodes for one hour in `debug`, allocation `ChemGraph`. The
controller uses `mofa_env_py312` (or an explicit `MOFA_ENV`). It checks the CP2K
executable and ASE shell with MPS off and on, using eight MPI ranks and eight
OpenMP threads per rank. It requires converged SCF and finite outputs; the ASE
test evaluates energy, forces, and stress twice through the same shell process.
Use a fresh allocation: existing MPS daemons cause the test to stop.

For direct executable checks inside an existing interactive allocation:

```bash
bash cp2k-test/run-polaris.sh        # one node, four MPI ranks
bash cp2k-test/run-polaris-2node.sh  # two nodes, eight MPI ranks
```

The direct scripts check normal program termination only; inspect SCF convergence
separately. The sample input includes `IGNORE_CONVERGENCE_FAILURE` and block size
32. The diagnostic removes the convergence override from its direct-run copy;
its ASE stages use MOFA's production template, including its numerical settings.
Neither runner changes the tracked input or production configuration.

Outputs stay in ignored local directories: `cp2k-test/run-*/` for direct checks
and `run/cp2k-diagnostic-*/` for the executable/ASE-shell test. Inspect `cp2k.out`,
captured stderr, and the diagnostic's `summary.json` locally. Generated outputs,
replay collections, and result reports are not included in Git.

Offline checks for the test code:

```bash
mofa_env_py312/bin/python tests/test_cp2k_diagnostic.py
```
