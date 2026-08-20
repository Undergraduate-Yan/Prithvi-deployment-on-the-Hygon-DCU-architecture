#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100"
IMAGE="prithvi-k100:20260807-dtk2504"

if [[ ! -f "${PROJECT_ROOT}/evaluate_full_test_k100_fp32.py" ]]; then
  echo "Missing script: ${PROJECT_ROOT}/evaluate_full_test_k100_fp32.py" >&2
  exit 1
fi

# The vendor K100 runtime initializes as container root. Retaining the host
# user's primary GID plus umask 0002 keeps generated evidence editable in VS Code.
docker run --rm \
  --user "0:$(id -g)" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 109 \
  --ipc=host \
  --shm-size=4g \
  -v /opt/dtk-25.04.4:/opt/dtk-25.04.4:ro \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "${PROJECT_ROOT}:/workspace" \
  -e HOME=/tmp/prithvi-home \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e PRITHVI_PROJECT_ROOT=/workspace \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint bash \
  "${IMAGE}" \
  -lc 'umask 0002; exec python /workspace/evaluate_full_test_k100_fp32.py'
