#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100"
IMAGE="prithvi-k100:20260807-dtk2504"
EXPECTED_HOST="machine2"
NODE_LABEL="K100-2"
LOCK_DIR="/tmp/prithvi_k100_fp32_fp16_accuracy.lock"

if [[ "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Run this accuracy gate on K100-2 (${EXPECTED_HOST}, SSH port 8002)." >&2
  exit 1
fi
if [[ "$(cat /sys/class/drm/card1/device/gpu_busy_percent)" != "0" ]]; then
  echo "K100-2 is busy; refusing to start the FP32/FP16 accuracy gate." >&2
  exit 2
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another FP32/FP16 accuracy job holds ${LOCK_DIR}." >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_PARENT="${PROJECT_ROOT}/outputs/k100_full_test_fp32_fp16"
OUTPUT_DIR="${OUTPUT_PARENT}/${RUN_ID}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
HOST_GID="$(id -g)"

docker run --rm \
  --user "0:${HOST_GID}" \
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
  -e BENCHMARK_NODE="${NODE_LABEL}" \
  -e PRITHVI_IMAGE_TAG="${IMAGE}" \
  -e PRITHVI_IMAGE_ID="${IMAGE_ID}" \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "umask 0002; exec python /workspace/evaluate_full_test_k100_fp32_fp16.py --output-dir /workspace/outputs/k100_full_test_fp32_fp16/${RUN_ID}"

mkdir -p "${OUTPUT_PARENT}"
printf '%s\n' "${OUTPUT_DIR}" >"${OUTPUT_PARENT}/LATEST"
echo "FP32/FP16 full-test accuracy gate complete: ${OUTPUT_DIR}"
