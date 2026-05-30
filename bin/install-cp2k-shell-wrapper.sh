#!/usr/bin/env bash
# Install a cp2k_shell.ssmp wrapper into the active conda env's bin/.
#
# Why this is needed (conda-forge cp2k + ASE on a cloud VM):
#   1. ASE's CP2K calculator asserts the launch command contains the substring
#      "cp2k_shell" (ase/calculators/cp2k.py). conda-forge ships only
#      `cp2k.ssmp`, so MOFA's `cp2k.ssmp --shell` command fails that assert
#      with a bare AssertionError before CP2K ever runs.
#   2. conda-forge does not set CP2K_DATA_DIR, so even once the shell launches
#      CP2K aborts: "basis set <DZVP-MOLOPT-SR-GTH> ... not found in
#      BASIS_MOLOPT".
# The wrapper's name satisfies (1); it exports CP2K_DATA_DIR for (2). Point
# MOFA_CP2K_BIN at it (bin/run-cloud-vm.sh defaults to `cp2k_shell.ssmp`).
#
# Usage: conda activate <env> && bin/install-cp2k-shell-wrapper.sh
set -euo pipefail

PREFIX="${CONDA_PREFIX:?activate the mofa-stream env first (CONDA_PREFIX unset)}"
command -v cp2k.ssmp >/dev/null || {
    echo "cp2k.ssmp not on PATH in $PREFIX — is the env activated / cp2k installed?" >&2
    exit 1
}
[[ -f "$PREFIX/share/cp2k/data/BASIS_MOLOPT" ]] || {
    echo "warning: $PREFIX/share/cp2k/data/BASIS_MOLOPT not found — CP2K_DATA_DIR guess may be wrong" >&2
}

WRAP="$PREFIX/bin/cp2k_shell.ssmp"
cat > "$WRAP" <<'EOF'
#!/usr/bin/env bash
# Wrapper around conda-forge cp2k.ssmp for ASE — see
# bin/install-cp2k-shell-wrapper.sh and §9 of envs/chameleon-stream.md.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CP2K_DATA_DIR="${CP2K_DATA_DIR:-$here/../share/cp2k/data}"
# CP2K runs in its own parsl worker subprocess, so OpenMP threads here do NOT
# steal librdkafka's broker threads in the main (octopus) process. The global
# OMP_NUM_THREADS=1 cap (§0) otherwise leaves cp2k.ssmp single-threaded, which
# makes one MOF SCF take tens of minutes on CPU. Give CP2K real threads while
# keeping nested BLAS single-threaded (MKL/OPENBLAS caps are inherited) to
# avoid OMP×BLAS oversubscription. Override with MOFA_CP2K_OMP.
export OMP_NUM_THREADS="${MOFA_CP2K_OMP:-8}"
exec cp2k.ssmp --shell "$@"
EOF
chmod +x "$WRAP"
echo "installed $WRAP (OMP_NUM_THREADS=${MOFA_CP2K_OMP:-8})"
