#!/bin/bash
# Default to ASE's shell protocol. Standalone tests select cp2k.psmp instead.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "${script_dir}/polaris-paths.sh"
case "${CP2K_BINARY:-cp2k_shell.psmp}" in
    cp2k_shell.ssmp|cp2k_shell.psmp|cp2k.ssmp|cp2k.psmp)
        cp2k_exe="${CP2K_ROOT}/exe/local_cuda/${CP2K_BINARY:-cp2k_shell.psmp}" ;;
    *) echo "Unsupported CP2K_BINARY: ${CP2K_BINARY}" >&2; exit 2 ;;
esac
if [[ ! -x "${cp2k_exe}" || ! -f "${CP2K_DATA_DIR}/BASIS_MOLOPT" || ! -f "${CP2K_DATA_DIR}/GTH_POTENTIALS" ]]; then
    echo "Missing external CP2K executable or data: ${cp2k_exe}, ${CP2K_DATA_DIR}" >&2
    exit 2
fi
# ASE communicates over stdout; module diagnostics must stay on stderr.
source "${script_dir}/polaris-simulation-env.sh" >&2
exec "${cp2k_exe}" "$@"
