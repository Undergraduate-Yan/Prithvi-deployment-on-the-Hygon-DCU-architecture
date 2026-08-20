#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100"
IMAGE="prithvi-k100:20260807-dtk2504"
EXPECTED_HOST="machine2"
NODE_LABEL="K100-2"
SAMPLE_INDEX="${1:-0}"

if [[ "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "Run this FP16 gate on K100-2 (${EXPECTED_HOST}, SSH port 8002)." >&2
  exit 1
fi
if [[ ! "${SAMPLE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "sample index must be a non-negative integer" >&2
  exit 1
fi
if [[ "$(cat /sys/class/drm/card1/device/gpu_busy_percent)" != "0" ]]; then
  echo "K100-2 is currently busy; refusing to start the FP16 gate." >&2
  exit 2
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/k100_fp32_fp16_real_sample/sample_$(printf '%03d' "${SAMPLE_INDEX}")_${RUN_ID}"
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
  -lc "umask 0002; exec python /workspace/verify_real_sample_fp32_fp16_k100.py --sample-index ${SAMPLE_INDEX} --output-dir /workspace/outputs/k100_fp32_fp16_real_sample/sample_$(printf '%03d' "${SAMPLE_INDEX}")_${RUN_ID}"

printf '%s\n' "${OUTPUT_DIR}" >"${PROJECT_ROOT}/outputs/k100_fp32_fp16_real_sample/LATEST"
echo "FP32/FP16 real-sample gate complete: ${OUTPUT_DIR}"
