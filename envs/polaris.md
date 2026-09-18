# Installing and running on Polaris

Follow the [Polaris setup guide](../polaris-build/instruction.md#1-configure-external-installations)
to configure external CP2K/LAMMPS paths, prepare models, and create `mofa_env_py312`.
The existing external LAMMPS installation depends on Python under a private home
directory; collaborators need an accessible matching runtime before executing jobs.
The [external installation recipes](../polaris-build/instruction.md#build-separate-installations)
use separate CP2K and LAMMPS source checkouts outside MOFA. The build scripts live
in `polaris-build/`; the LAMMPS recipe keeps its Python runtime in that external
checkout so it can be shared with collaborators who have filesystem access.

After setup and login-node checks, submit from the repository root:

```bash
qsub run-polaris-local-smoke.sh
# Inspect successful simulation results before trying the ten-node test:
qsub -v MOFA_SIMULATION_BUDGET=8 run-polaris-repo-test.sh
```

Both scripts activate `<repository>/mofa_env_py312` automatically and override
any inherited `MOFA_ENV`, including one supplied with `qsub -v`. The smoke job
defaults to budget 8; the repository test defaults to unlimited (`-1`) without
the override above. The budget is a workflow scheduling limit, not a requested
number of fully characterized MOFs.

Use the guide's [submission and environment-variable reference](../polaris-build/instruction.md#5-run-simulations)
for external-path overrides, node counts, queues, output locations, and changes
from the previous `mofa_env`/repository-local dependency setup. Pass optional
overrides with `qsub -v`; exporting variables in the login shell alone does not
forward them to PBS jobs. Manual activation is needed only for interactive checks.

The installer refuses existing targets. If installation completed but validation
failed, fix the prerequisite and [rerun validation](../polaris-build/instruction.md#4-check-the-installation)
in that environment instead of rebuilding it. Alternative `--prefix` environments
can be checked interactively, but the workflow scripts still select `mofa_env_py312`.
