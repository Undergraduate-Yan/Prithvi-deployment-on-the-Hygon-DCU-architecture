#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 FP32_ROOT FP32_CANDIDATE_ROOT SAMPLE RUN_ROOT" >&2
  exit 64
fi

IMAGE='sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291'
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FP32_ROOT="$(realpath -e -- "$1")"
FP32_CANDIDATE_ROOT="$(realpath -e -- "$2")"
SAMPLE="$(realpath -e -- "$3")"
RUN_PARENT="$(realpath -m -- "$(dirname -- "$4")")"
RUN_ROOT="${RUN_PARENT}/$(basename -- "$4")"
FINGERPRINTER="${TOOLS}/fingerprint_phase11_mixed_precision_performance_runtime.py"
WRAPPER="${TOOLS}/measure_phase11_k100_vram_around_command.py"
FP32_BENCHMARK="${TOOLS}/benchmark_phase11_fp32_segment25_headbarrier_all_resident_trial.py"
VRAM_COUNTER='/sys/class/drm/card1/device/mem_info_vram_used'
GPU_BUSY_COUNTER='/sys/class/drm/card1/device/gpu_busy_percent'

exec 9>/var/tmp/phase11-fp32-segment25-same-scope-vram-node3.lock
flock -n 9 || exit 75

test "$(hostname)" = machine3
test -z "$(docker ps -q)"
test -r "${VRAM_COUNTER}"
test ! -e "${RUN_ROOT}"
mkdir -p -- "${RUN_PARENT}"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${FINGERPRINTER}")" = 5516
test "$(sha256sum "${FINGERPRINTER}" | awk '{print $1}')" = f38e24ea72c02a164ec311ce4521f119da1e0f397fc796ea380dc62bb2167577
test "$(stat -c %s "${WRAPPER}")" = 9459
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = 402c43803e80defaa32c110e28b1ce9c3cd360889b55d3d1c2c3abca72b93428
test "$(stat -c %s "${FP32_BENCHMARK}")" = 25591
test "$(sha256sum "${FP32_BENCHMARK}" | awk '{print $1}')" = 31af48acd99b1ddbc614854bbf99721ca0f1d86bc984b21605883740a7f743dd
test "$(stat -c %s "${FP32_ROOT}/tools/lsmod")" = 819664
test "$(sha256sum "${FP32_ROOT}/tools/lsmod" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12

if [[ -r "${GPU_BUSY_COUNTER}" ]]; then
  for _ in 1 2 3 4 5; do
    test "$(tr -d '\r\n' <"${GPU_BUSY_COUNTER}")" = 0
    sleep 0.2
  done
fi

mkdir -- "${RUN_ROOT}"
docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
docker run --rm --name phase11-fp32-vram-fingerprint-node3 --entrypoint python3 \
  -e PATH=/work/root/tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro -v "${FP32_ROOT}:/work/root:ro" \
  -v "${TOOLS}:/work/tools:ro" -w /work \
  "${IMAGE}" /work/tools/fingerprint_phase11_mixed_precision_performance_runtime.py \
  >"${RUN_ROOT}/runtime_fingerprint.json" 2>"${RUN_ROOT}/runtime_fingerprint.stderr"

CONTAINER='phase11-fp32-segment25-vram-node3'
CID="$(docker create --name "${CONTAINER}" --entrypoint python3 \
  --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
  -e PATH=/work/root/tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${TOOLS}:/work/tools:ro" \
  -v "${FP32_ROOT}:/work/root:ro" -v "${FP32_CANDIDATE_ROOT}:/work/candidate:ro" \
  -v "${SAMPLE}:/work/sample:ro" -v "${RUN_ROOT}:/work/run:rw" -w /work \
  "${IMAGE}" /work/tools/measure_phase11_k100_vram_around_command.py \
  --label fp32_25segment_same_protocol --container-image-id "${IMAGE}" \
  --runtime-fingerprint /work/run/runtime_fingerprint.json \
  --vram-counter "${VRAM_COUNTER}" --sampling-interval-ms 10 \
  --output-dir /work/run/output -- \
  python3 /work/tools/benchmark_phase11_fp32_segment25_headbarrier_all_resident_trial.py \
  --root /work/root --candidate-root /work/candidate --sample /work/sample \
  --trial-index 1 --output-dir /work/run/wrapped_benchmark_output)"
printf '%s\n' "${CID}" >"${RUN_ROOT}/container_id.txt"
docker start "${CID}" >/dev/null
docker inspect "${CID}" >"${RUN_ROOT}/container_inspect_started.json"
RC="$(docker wait "${CID}")"
docker logs "${CID}" >"${RUN_ROOT}/container.log" 2>&1 || true
docker inspect "${CID}" >"${RUN_ROOT}/container_inspect_exited.json"
printf '%s\n' "${RC}" >"${RUN_ROOT}/container.exit"
docker rm "${CID}" >/dev/null
if [[ "${RC}" -eq 0 ]]; then
  printf 'completed_passed_latency_outputs_invalid\n' >"${RUN_ROOT}/RUN_STATUS"
else
  printf 'failed\n' >"${RUN_ROOT}/RUN_STATUS"
fi
test ! -f "${RUN_ROOT}/output/result.json" || cat "${RUN_ROOT}/output/result.json"
exit "${RC}"
