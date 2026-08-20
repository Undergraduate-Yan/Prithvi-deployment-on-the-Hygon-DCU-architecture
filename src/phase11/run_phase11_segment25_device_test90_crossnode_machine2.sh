#!/usr/bin/env bash
set -euo pipefail

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
BUNDLE='/var/tmp/phase11_int8_backbone_segment25_evidence_bundle_20260817'
PROJECT='/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100'
RUN_ROOT='/var/tmp/20260817-phase11-segment25-crossnode-machine2'
RUN="${RUN_ROOT}/device_resident_test90_crossnode_v1"
CONTAINER='phase11-segment25-device-test90-crossnode-machine2-v1'
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
VALIDATOR="${SCRIPT_DIR}/validate_phase11_segment25_crossnode_test90.py"
EVALUATOR="${BUNDLE}/tools/evaluate_phase11_int8_backbone_segment25_iobinding_90.py"
LSMOD="${BUNDLE}/tools/lsmod"
BUSY_PATH='/sys/class/drm/card1/device/gpu_busy_percent'
VRAM_USED_PATH='/sys/class/drm/card1/device/mem_info_vram_used'
DEFAULT_PORTABILITY_RESULT="${RUN_ROOT}/cache_portability_smoke_v2/cache_portability_validation_v2.json"
case "$#" in
  1)
    PORTABILITY_INPUT="${DEFAULT_PORTABILITY_RESULT}"
    PORTABILITY_SHA256="$1"
    ;;
  2)
    PORTABILITY_INPUT="$1"
    PORTABILITY_SHA256="$2"
    ;;
  *)
    echo "usage: $0 [PORTABILITY_VALIDATION_JSON] EXPECTED_SHA256" >&2
    exit 64
    ;;
esac

exec 9>/var/tmp/phase11-segment25-device-test90-crossnode-machine2-v1.lock
flock -n 9 || exit 75
test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test -r "${BUSY_PATH}"
test "$(tr -d '[:space:]' <"${BUSY_PATH}")" = 0 || {
  echo "K100-2 is busy; refusing to start test90" >&2
  exit 75
}
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${BUNDLE}/BUNDLE_MANIFEST.json")" = 126255
test "$(sha256sum "${BUNDLE}/BUNDLE_MANIFEST.json" | awk '{print $1}')" = 0abd5a99c3fa9ebcd4236bf6f57090f9fce102777535402313e081777eb9c6cc
test "$(stat -c %s "${EVALUATOR}")" = 18982
test "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = b0d4e876c6cd3845fad10775f803591c558621cb81af9977353152c73fdefe79
test "$(stat -c %s "${LSMOD}")" = 819664
test "$(sha256sum "${LSMOD}" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test "$(stat -c %s "${VALIDATOR}")" = 17529
test "$(sha256sum "${VALIDATOR}" | awk '{print $1}')" = 18fad4540e183fcb00a1a098bc4d424ec2b9c1acd2d10a28dd20cdc626364cbb
test "$(stat -c %s "${PROJECT}/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py")" = 16657
test "$(sha256sum "${PROJECT}/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py" | awk '{print $1}')" = a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f
test "$(stat -c %s "${PROJECT}/phase11_frozen_test90/frozen_fp32_test90_inputs.npz")" = 144553732
test "$(sha256sum "${PROJECT}/phase11_frozen_test90/frozen_fp32_test90_inputs.npz" | awk '{print $1}')" = 6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86
test "$(stat -c %s "${PROJECT}/phase11_frozen_test90/manifest.json")" = 28349
test "$(sha256sum "${PROJECT}/phase11_frozen_test90/manifest.json" | awk '{print $1}')" = cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5
test -f "${PORTABILITY_INPUT}"
test ! -L "${PORTABILITY_INPUT}"
PORTABILITY_RESULT="$(readlink -f -- "${PORTABILITY_INPUT}")"
case "${PORTABILITY_RESULT}" in
  "${RUN_ROOT}"/*) ;;
  *) echo "portability validation must be under ${RUN_ROOT}" >&2; exit 65 ;;
esac
python3 "${VALIDATOR}" preflight-portability \
  --portability-result "${PORTABILITY_RESULT}" \
  --portability-sha256 "${PORTABILITY_SHA256}" >/dev/null
test ! -e "${RUN}"

mkdir -p "${RUN}"
cp --reflink=auto --preserve=mode,timestamps -- "${PORTABILITY_RESULT}" "${RUN}/cache_portability_validation.json"
test "$(sha256sum "${RUN}/cache_portability_validation.json" | awk '{print $1}')" = "${PORTABILITY_SHA256}"
chmod a-w "${RUN}/cache_portability_validation.json"
sha256sum "$0" "${VALIDATOR}" "${EVALUATOR}" >"${RUN}/launcher_and_tools.sha256"
sha256sum "${RUN}/cache_portability_validation.json" >"${RUN}/cache_portability_validation.sha256"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'scope=cross_node_25_segment_device_resident_iobinding_test90\n'
  printf 'cache_portability_prerequisite=%s\n' "${PORTABILITY_RESULT}"
  printf 'cache_portability_sha256=%s\n' "${PORTABILITY_SHA256}"
  printf 'exact_image_identity=false; critical_runtime_content_match=true\n'
  printf 'bundle_mount=read_only; project_evidence_mount=read_only; cpu_fallback=disabled\n'
  printf 'container_memory_limit=16g; rationale=one_cached_session_at_a_time_plus_90_device_OrtValues; no_compile_or_all_resident_sessions\n'
  printf 'gpu_busy_percent=' && cat "${BUSY_PATH}"
  if [[ -r "${VRAM_USED_PATH}" ]]; then
    printf 'vram_used_bytes=' && cat "${VRAM_USED_PATH}"
  fi
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${RUN}/host_pre.txt" 2>&1

(
  while true; do
    date -Ins
    docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' "${CONTAINER}" 2>/dev/null || true
    grep '^MemAvailable:' /proc/meminfo || true
    if [[ -r "${BUSY_PATH}" ]]; then
      printf 'gpu_busy_percent=' && cat "${BUSY_PATH}"
    fi
    if [[ -r "${VRAM_USED_PATH}" ]]; then
      printf 'vram_used_bytes=' && cat "${VRAM_USED_PATH}"
    fi
    sleep 5
  done
) >"${RUN}/telemetry.log" 2>&1 &
TELEMETRY_PID=$!
stop_telemetry() {
  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
    TELEMETRY_PID=''
  fi
}
trap stop_telemetry EXIT INT TERM

# Re-check exclusivity immediately before Docker claims the K100 devices.
test -z "$(docker ps -q)"
test "$(tr -d '[:space:]' <"${BUSY_PATH}")" = 0 || {
  echo "K100-2 became busy during preflight; refusing to start test90" >&2
  exit 75
}

set +e
docker run --rm --name "${CONTAINER}" --entrypoint /usr/bin/python3 \
  --ulimit stack=-1:-1 --memory=16g --pids-limit=1024 --network=none \
  -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "${BUNDLE}:/work/root:ro" -v "${PROJECT}:/workspace:ro" -v "${RUN}:/work/run:rw" \
  -w /work "${IMAGE}" /work/root/tools/evaluate_phase11_int8_backbone_segment25_iobinding_90.py \
  --root /work/root \
  --device-single /work/root/segment25_iobinding_end_to_end_single/output/result.json \
  --host90-result /work/root/segment25_cached_end_to_end_test90_v2/output/result.json \
  --host90-tensors /work/root/segment25_cached_end_to_end_test90_v2/output/predictions_and_targets.npz \
  --base-evaluator /workspace/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py \
  --helper /workspace/evaluate_full_test_k100_onnx_int8.py \
  --fp32-result /workspace/outputs/k100_full_test_fp32/20260807-171433/result.json \
  --fp32-csv /workspace/outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv \
  --fp32-tensors /workspace/outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt \
  --frozen-inputs /workspace/phase11_frozen_test90/frozen_fp32_test90_inputs.npz \
  --frozen-input-manifest /workspace/phase11_frozen_test90/manifest.json \
  --output-dir /work/run/output >"${RUN}/evaluation.log" 2>&1
EVALUATION_RC=$?
set -e

stop_telemetry
trap - EXIT INT TERM
printf '%s\n' "${EVALUATION_RC}" >"${RUN}/evaluation.exit"
{
  date -Ins
  printf 'gpu_busy_percent=' && cat "${BUSY_PATH}"
  if [[ -r "${VRAM_USED_PATH}" ]]; then
    printf 'vram_used_bytes=' && cat "${VRAM_USED_PATH}"
  fi
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${RUN}/host_post.txt" 2>&1

POST_RC=66
if [[ -f "${RUN}/output/result.json" ]]; then
  set +e
  python3 "${VALIDATOR}" validate-result \
    --portability-result "${RUN}/cache_portability_validation.json" \
    --portability-sha256 "${PORTABILITY_SHA256}" \
    --bundle-manifest "${BUNDLE}/BUNDLE_MANIFEST.json" \
    --evaluator "${EVALUATOR}" --lsmod "${LSMOD}" \
    --image-inspect "${RUN}/image_inspect.json" \
    --test90-result "${RUN}/output/result.json" --output-dir "${RUN}/output" \
    --output "${RUN}/post_validation.json" >"${RUN}/post_validation.log" 2>&1
  POST_RC=$?
  set -e
fi
printf '%s\n' "${POST_RC}" >"${RUN}/post_validation.exit"
if [[ -f "${RUN}/post_validation.json" ]]; then
  cat "${RUN}/post_validation.json"
fi

if [[ "${EVALUATION_RC}" != 0 ]]; then
  exit "${EVALUATION_RC}"
fi
exit "${POST_RC}"
