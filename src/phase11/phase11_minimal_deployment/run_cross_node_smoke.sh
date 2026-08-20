#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
: "${PHASE11_K100_IMAGE_ID:?host launcher must attest the locked image digest}"
: "${PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION:?formal smoke must inject PATH at docker run time}"
if [[ "${PATH%%:*}" != "${SCRIPT_DIR}/bin" ]]; then
  echo "formal cross-node smoke requires bundle/tools/bin first on PATH before entrypoint" >&2
  exit 64
fi
exec python3 "${SCRIPT_DIR}/acceptance_k100.py" cross-node-smoke "$@"
