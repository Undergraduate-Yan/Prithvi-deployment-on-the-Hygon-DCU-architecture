#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100"
IMAGE="prithvi-k100:20260807-dtk2504"
EXPECTED_HOST="machine2"
NODE_LABEL="K100-2"
SMI="/opt/dtk-25.04.4/.hyhal/bin/hy-smi"
SMI_LIBRARY_PATH="/opt/dtk-25.04.4/lib:/opt/dtk-25.04.4/.hyhal/lib"
LOCK_DIR="/tmp/prithvi_k100_fp32_benchmark.lock"

if [[ "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "This formal benchmark must run on K100-2 (${EXPECTED_HOST}, SSH port 8002)." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/benchmark_k100_fp32.py" ]]; then
  echo "Missing script: ${PROJECT_ROOT}/benchmark_k100_fp32.py" >&2
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Missing Docker image on K100-2: ${IMAGE}" >&2
  exit 1
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another Prithvi K100 benchmark holds ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_PARENT="${PROJECT_ROOT}/outputs/k100_fp32_benchmark"
RUN_ROOT="${OUTPUT_PARENT}/${RUN_ID}"
mkdir -p "${OUTPUT_PARENT}"
mkdir "${RUN_ROOT}"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
HOST_GID="$(id -g)"

capture_snapshot() {
  local output_file="$1"
  {
    date --iso-8601=seconds
    hostname
    uname -a
    uptime
    printf 'loadavg='
    cat /proc/loadavg
    printf 'gpu_busy_percent='
    cat /sys/class/drm/card1/device/gpu_busy_percent
    printf 'vram_used_bytes='
    cat /sys/class/drm/card1/device/mem_info_vram_used
    printf 'vram_total_bytes='
    cat /sys/class/drm/card1/device/mem_info_vram_total
    LD_LIBRARY_PATH="${SMI_LIBRARY_PATH}" "${SMI}" \
      --showuse --showtemp --showpower --showmemuse --showpids --showperflevel
    docker ps --format '{{.ID}} {{.Names}} {{.Status}} {{.Image}} {{.Command}}'
  } >"${output_file}" 2>&1
}

docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
docker version >"${RUN_ROOT}/docker_version.txt"
capture_snapshot "${RUN_ROOT}/host_initial.txt"

INITIAL_BUSY="$(cat /sys/class/drm/card1/device/gpu_busy_percent)"
if [[ "${INITIAL_BUSY}" != "0" ]]; then
  echo "K100-2 is busy (${INITIAL_BUSY}%). Benchmark aborted before timing." >&2
  exit 2
fi

echo "Benchmark run root: ${RUN_ROOT}"
echo "Image ID: ${IMAGE_ID}"

for ROUND_INDEX in 1 2 3; do
  ROUND_NAME="$(printf 'round_%02d' "${ROUND_INDEX}")"
  capture_snapshot "${RUN_ROOT}/${ROUND_NAME}_pre_smi.txt"
  PRE_BUSY="$(cat /sys/class/drm/card1/device/gpu_busy_percent)"
  if [[ "${PRE_BUSY}" != "0" ]]; then
    echo "K100-2 became busy before ${ROUND_NAME} (${PRE_BUSY}%)." >&2
    exit 2
  fi

  echo "Starting ${ROUND_NAME}..."
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
    -e PRITHVI_IMAGE_TAG="${IMAGE}" \
    -e PRITHVI_IMAGE_ID="${IMAGE_ID}" \
    -e BENCHMARK_NODE="${NODE_LABEL}" \
    -e BENCHMARK_HOSTNAME="${EXPECTED_HOST}" \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e NO_ALBUMENTATIONS_UPDATE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "umask 0002; exec python /workspace/benchmark_k100_fp32.py --round-index ${ROUND_INDEX} --output-dir /workspace/outputs/k100_fp32_benchmark/${RUN_ID}/${ROUND_NAME}" \
    2>&1 | tee "${RUN_ROOT}/${ROUND_NAME}.log"

  capture_snapshot "${RUN_ROOT}/${ROUND_NAME}_post_smi.txt"
  sleep 5
done

echo "Aggregating the three independent rounds..."
set +e
docker run --rm \
  --user "0:${HOST_GID}" \
  -v "${PROJECT_ROOT}:/workspace" \
  -e HOME=/tmp/prithvi-home \
  -e PRITHVI_PROJECT_ROOT=/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "umask 0002; exec python /workspace/benchmark_k100_fp32.py --aggregate-dir /workspace/outputs/k100_fp32_benchmark/${RUN_ID}" \
  2>&1 | tee "${RUN_ROOT}/aggregate.log"
AGGREGATE_STATUS="${PIPESTATUS[0]}"
set -e

capture_snapshot "${RUN_ROOT}/host_final.txt"
printf '%s\n' "${RUN_ROOT}" >"${OUTPUT_PARENT}/LATEST"
if [[ "${AGGREGATE_STATUS}" -ne 0 ]]; then
  echo "Benchmark data complete, aggregate status=${AGGREGATE_STATUS}: ${RUN_ROOT}" >&2
  exit "${AGGREGATE_STATUS}"
fi
echo "Benchmark complete: ${RUN_ROOT}"
