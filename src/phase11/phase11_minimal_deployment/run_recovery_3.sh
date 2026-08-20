#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
: "${PHASE11_K100_IMAGE_ID:?host launcher must attest the locked image digest}"
export PATH="${SCRIPT_DIR}/bin:${PATH}"
exec python3 "${SCRIPT_DIR}/acceptance_k100.py" recovery "$@"
