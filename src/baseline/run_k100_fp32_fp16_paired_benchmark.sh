#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100"
IMAGE="prithvi-k100:20260807-dtk2504"
EXPECTED_HOST="machine2"
NODE_LABEL="K100-2"
SMI="/opt/dtk-25.04.4/.hyhal/bin/hy-smi"
SMI_LIBRARY_PATH="/opt/dtk-25.04.4/lib:/opt/dtk-25.04.4/.hyhal/lib"
LOCK_DIR="/tmp/prithvi_k100_fp32_fp16_paired_benchmark.lock"
MAX_START_ATTEMPTS=40

if [[ "$(hostname)" != "${EXPECTED_HOST}" ]]; then
  echo "This paired benchmark must run on K100-2 (${EXPECTED_HOST}, SSH port 8002)." >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/benchmark_k100_fp32_fp16_paired.py" ]]; then
  echo "Missing benchmark script." >&2
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Missing image: ${IMAGE}" >&2
  exit 1
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another paired benchmark holds ${LOCK_DIR}." >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

ACCURACY_LATEST="${PROJECT_ROOT}/outputs/k100_full_test_fp32_fp16/LATEST"
if [[ ! -f "${ACCURACY_LATEST}" ]]; then
  echo "Missing full-test FP16 accuracy LATEST pointer." >&2
  exit 1
fi
ACCURACY_ROOT="$(cat "${ACCURACY_LATEST}")"
ACCURACY_RESULT="${ACCURACY_ROOT}/result.json"
if [[ ! -f "${ACCURACY_RESULT}" ]]; then
  echo "Missing accuracy result: ${ACCURACY_RESULT}" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_PARENT="${PROJECT_ROOT}/outputs/k100_fp32_fp16_paired"
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

read_temperatures() {
  local temperature_output edge_temperature memory_temperature
  temperature_output="$(
    LD_LIBRARY_PATH="${SMI_LIBRARY_PATH}" "${SMI}" --showtemp 2>/dev/null
  )"
  edge_temperature="$(
    awk '/Temperature \(Sensor edge\)/ {print $NF; exit}' <<<"${temperature_output}"
  )"
  memory_temperature="$(
    awk '/Temperature \(Sensor mem\)/ {print $NF; exit}' <<<"${temperature_output}"
  )"
  printf '%s %s\n' "${edge_temperature}" "${memory_temperature}"
}

wait_for_start_condition() {
  local label="$1" attempt second busy temperatures edge_temperature memory_temperature
  for ((attempt = 1; attempt <= MAX_START_ATTEMPTS; attempt++)); do
    busy=0
    for ((second = 1; second <= 15; second++)); do
      if [[ "$(cat /sys/class/drm/card1/device/gpu_busy_percent)" != "0" ]]; then
        busy=1
        break
      fi
      sleep 1
    done
    if [[ "${busy}" -ne 0 ]]; then
      echo "${label}: K100 was active during the 15-second idle gate; retry ${attempt}."
      sleep 10
      continue
    fi
    temperatures="$(read_temperatures)"
    edge_temperature="${temperatures%% *}"
    memory_temperature="${temperatures##* }"
    if awk -v edge="${edge_temperature}" -v mem="${memory_temperature}" \
      'BEGIN {exit !((edge + 0) <= 50.0 && (mem + 0) <= 72.0)}'; then
      echo "${label}: start gate passed; edge=${edge_temperature}C mem=${memory_temperature}C."
      return 0
    fi
    echo "${label}: waiting for temperature gate; edge=${edge_temperature}C mem=${memory_temperature}C."
    sleep 15
  done
  echo "${label}: start conditions not reached after ${MAX_START_ATTEMPTS} attempts." >&2
  return 2
}

docker_args=(
  --rm
  --user "0:${HOST_GID}"
  --device=/dev/kfd
  --device=/dev/dri
  --group-add 44
  --group-add 109
  --ipc=host
  --shm-size=4g
  --cpuset-cpus=0-7
  -v /opt/dtk-25.04.4:/opt/dtk-25.04.4:ro
  -v /opt/hyhal:/opt/hyhal:ro
  -v "${PROJECT_ROOT}:/workspace"
  -e HOME=/tmp/prithvi-home
  -e MPLCONFIGDIR=/tmp/matplotlib
  -e PRITHVI_PROJECT_ROOT=/workspace
  -e PRITHVI_IMAGE_TAG="${IMAGE}"
  -e PRITHVI_IMAGE_ID="${IMAGE_ID}"
  -e BENCHMARK_NODE="${NODE_LABEL}"
  -e BENCHMARK_HOSTNAME="${EXPECTED_HOST}"
  -e OMP_NUM_THREADS=4
  -e MKL_NUM_THREADS=4
  -e PYTHONHASHSEED=0
  -e PYTHONDONTWRITEBYTECODE=1
  -e NO_ALBUMENTATIONS_UPDATE=1
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  --entrypoint bash
)

docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
docker version >"${RUN_ROOT}/docker_version.txt"
printf '%s\n' "${ACCURACY_RESULT}" >"${RUN_ROOT}/accuracy_result_path.txt"
sha256sum "${ACCURACY_RESULT}" >"${RUN_ROOT}/accuracy_result_sha256.txt"
capture_snapshot "${RUN_ROOT}/host_initial.txt"

docker run "${docker_args[@]}" "${IMAGE_ID}" -lc \
  "umask 0002; exec python /workspace/benchmark_k100_fp32_fp16_paired.py --write-protocol /workspace/outputs/k100_fp32_fp16_paired/${RUN_ID}/protocol.json --accuracy-result /workspace/outputs/k100_full_test_fp32_fp16/${ACCURACY_ROOT##*/}/result.json --image-id '${IMAGE_ID}'" \
  2>&1 | tee "${RUN_ROOT}/protocol.log"
sha256sum "${RUN_ROOT}/protocol.json" >"${RUN_ROOT}/protocol.sha256"

echo "Paired benchmark run root: ${RUN_ROOT}"
echo "Frozen protocol SHA-256: $(cut -d' ' -f1 "${RUN_ROOT}/protocol.sha256")"

for ROUND_INDEX in 1 2 3; do
  ROUND_NAME="$(printf 'round_%02d' "${ROUND_INDEX}")"
  CURRENT_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
  if [[ "${CURRENT_IMAGE_ID}" != "${IMAGE_ID}" ]]; then
    echo "Container image ID changed after protocol freeze." >&2
    exit 2
  fi
  wait_for_start_condition "${ROUND_NAME}"
  capture_snapshot "${RUN_ROOT}/${ROUND_NAME}_pre_smi.txt"
  echo "Starting paired ${ROUND_NAME}..."
  docker run "${docker_args[@]}" "${IMAGE_ID}" -lc \
    "umask 0002; exec python /workspace/benchmark_k100_fp32_fp16_paired.py --round-index ${ROUND_INDEX} --protocol /workspace/outputs/k100_fp32_fp16_paired/${RUN_ID}/protocol.json --output-dir /workspace/outputs/k100_fp32_fp16_paired/${RUN_ID}/${ROUND_NAME}" \
    2>&1 | tee "${RUN_ROOT}/${ROUND_NAME}.log"
  capture_snapshot "${RUN_ROOT}/${ROUND_NAME}_post_smi.txt"
  sleep 15
done

echo "Aggregating three independent paired rounds..."
CURRENT_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
if [[ "${CURRENT_IMAGE_ID}" != "${IMAGE_ID}" ]]; then
  echo "Container image ID changed before aggregation." >&2
  exit 2
fi
set +e
docker run "${docker_args[@]}" "${IMAGE_ID}" -lc \
  "umask 0002; exec python /workspace/benchmark_k100_fp32_fp16_paired.py --aggregate-dir /workspace/outputs/k100_fp32_fp16_paired/${RUN_ID}" \
  2>&1 | tee "${RUN_ROOT}/aggregate.log"
AGGREGATE_STATUS="${PIPESTATUS[0]}"
set -e

capture_snapshot "${RUN_ROOT}/host_final.txt"
printf '%s\n' "${RUN_ROOT}" >"${OUTPUT_PARENT}/LATEST"
if [[ "${AGGREGATE_STATUS}" -ne 0 ]]; then
  echo "Paired data complete, aggregate status=${AGGREGATE_STATUS}: ${RUN_ROOT}" >&2
  exit "${AGGREGATE_STATUS}"
fi
echo "Paired benchmark complete: ${RUN_ROOT}"
