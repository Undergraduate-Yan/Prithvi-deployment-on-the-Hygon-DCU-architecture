#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE='sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291'
PROJECT='/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100'
ROOT='/var/tmp/20260817-phase11-fp32-backbone-node3'
TOOLS='/var/tmp/phase11_fp32_segment25_headbarrier_tools_v1'
CANDIDATE_ROOT='/var/tmp/20260818-phase11-fp32-segment25-headbarrier-node3'
SAMPLE="${PROJECT}/outputs/k100_real_sample_parity/sample_000_20260807-162919/sample_and_logits.pt"
BUILDER="${TOOLS}/build_phase11_fp32_segment25_head_fpn4_barrier.py"
SINGLE_EVALUATOR="${TOOLS}/evaluate_phase11_fp32_segment25_headbarrier_single.py"
TEST90_EVALUATOR="${TOOLS}/evaluate_phase11_fp32_segment25_headbarrier_test90.py"
CANDIDATE="${CANDIDATE_ROOT}/24_upernet_decoder_head_deconv_fpn4barrier.onnx"
BUILD_REPORT="${CANDIDATE_ROOT}/build_report.json"
SINGLE_ROOT="${CANDIDATE_ROOT}/single_v1"
HEAD_CACHE="${SINGLE_ROOT}/segment_24_fpn4barrier.mxr"
TEST90_ROOT="${CANDIDATE_ROOT}/test90_v1"

exec 9>/var/tmp/phase11-fp32-segment25-headbarrier-admission-node3.lock
flock -n 9 || exit 75

ACTIVE_CONTAINER=''
TELEMETRY_PID=''
cleanup() {
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ACTIVE_CONTAINER}" ]]; then
    docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

test "$(hostname)" = machine3
test -z "$(docker ps -q)"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test -d /opt/hyhal
test -d "${ROOT}"
test -d "${PROJECT}"
test -d "${TOOLS}"
test ! -e "${CANDIDATE_ROOT}"
AVAILABLE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
test "${AVAILABLE_KB}" -ge 41943040

# Immutable tools shipped for this run.
test "$(stat -c %s "${BUILDER}")" = 7440
test "$(sha256sum "${BUILDER}" | awk '{print $1}')" = 77fe9aeb2560e49bae1cf4746609a875d37a1e3faed9b314e65fed4e3f3b1f6e
test "$(stat -c %s "${SINGLE_EVALUATOR}")" = 18865
test "$(sha256sum "${SINGLE_EVALUATOR}" | awk '{print $1}')" = f78750b02006270efeef781cffd53dfd48b255dd75c33f729e691fece92da71e
test "$(stat -c %s "${TEST90_EVALUATOR}")" = 19253
test "$(sha256sum "${TEST90_EVALUATOR}" | awk '{print $1}')" = 05bb42b4765d072cce7f16b32bf5a2387301b19011173133d56f41635cbe739b

# Frozen FP32 25-segment source evidence. ROOT is mounted read-only in every container.
test "$(stat -c %s "${ROOT}/build_ln/segment25_report_deconv.json")" = 24959
test "$(sha256sum "${ROOT}/build_ln/segment25_report_deconv.json" | awk '{print $1}')" = c94692d130a0eb924290b3d56f790e166535fa3049fd9f3e72ef1fe16085c90f
test "$(stat -c %s "${ROOT}/build_ln/cpu_sequential_parity_deconv.json")" = 14840
test "$(sha256sum "${ROOT}/build_ln/cpu_sequential_parity_deconv.json" | awk '{print $1}')" = 58c076e974ea7116a3f943bce6ca653fb0803993ac77af18c0b4cdfb837c2e72
test "$(stat -c %s "${ROOT}/build_ln/models/24_upernet_decoder_head_deconv.onnx")" = 60551311
test "$(sha256sum "${ROOT}/build_ln/models/24_upernet_decoder_head_deconv.onnx" | awk '{print $1}')" = 2644c864768639151a062efcd0c69e657cdb0b7455c9b3561f08411397af2f6e
test "$(stat -c %s "${ROOT}/cache_ln_deconv_head/output/cpu_and_migraphx_output.npz")" = 3653541
test "$(sha256sum "${ROOT}/cache_ln_deconv_head/output/cpu_and_migraphx_output.npz" | awk '{print $1}')" = 98fe1142953a05944244f8934141b2500d787bfe9d549a2f3c34d7c643b49ab9
test "$(cat "${ROOT}/cache_ln_remaining/launcher.exit")" = 0
test "$(cat "${ROOT}/cache_ln_remaining/LATEST_COMPLETED_SEGMENT")" = 24
test "$(stat -c %s "${SAMPLE}")" = 4820344
test "$(sha256sum "${SAMPLE}" | awk '{print $1}')" = 4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62

# Frozen task90 sources; deeper FP32-reference identities are checked by the locked helper.
test "$(stat -c %s "${PROJECT}/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py")" = 16657
test "$(sha256sum "${PROJECT}/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py" | awk '{print $1}')" = a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f
test "$(stat -c %s "${PROJECT}/evaluate_full_test_k100_onnx_int8.py")" = 59386
test "$(sha256sum "${PROJECT}/evaluate_full_test_k100_onnx_int8.py" | awk '{print $1}')" = cd746e45ac45e8da148e6481e318c3701473088c2bba2397bbf85e6917517435
test "$(stat -c %s "${PROJECT}/phase11_frozen_test90/frozen_fp32_test90_inputs.npz")" = 144553732
test "$(sha256sum "${PROJECT}/phase11_frozen_test90/frozen_fp32_test90_inputs.npz" | awk '{print $1}')" = 6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86
test "$(stat -c %s "${PROJECT}/phase11_frozen_test90/manifest.json")" = 28349
test "$(sha256sum "${PROJECT}/phase11_frozen_test90/manifest.json" | awk '{print $1}')" = cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5
test -s "${PROJECT}/outputs/k100_full_test_fp32/20260807-171433/result.json"
test -s "${PROJECT}/outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv"
test -s "${PROJECT}/outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt"

mkdir "${CANDIDATE_ROOT}"
docker image inspect "${IMAGE}" >"${CANDIDATE_ROOT}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'scope=fp32_segment25_head_fpn4barrier_build_single_test90\n'
  printf 'old_root_read_only=true\nperformance_timing=false\n'
  printf 'mem_available_kb=%s\n' "${AVAILABLE_KB}"
  sha256sum "${BUILDER}" "${SINGLE_EVALUATOR}" "${TEST90_EVALUATOR}"
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${CANDIDATE_ROOT}/host_pre.txt" 2>&1
printf 'building\n' >"${CANDIDATE_ROOT}/RUN_STATUS"

ACTIVE_CONTAINER='phase11-fp32-segment25-headbarrier-build-node3'
set +e
docker run --rm --name "${ACTIVE_CONTAINER}" --entrypoint python3 \
  --memory=16g --pids-limit=1024 --ipc=host --shm-size=4g \
  -v "${ROOT}:/work/root:ro" -v "${TOOLS}:/work/tools:ro" \
  -v "${CANDIDATE_ROOT}:/work/candidate:rw" \
  -w /work "${IMAGE}" /work/tools/build_phase11_fp32_segment25_head_fpn4_barrier.py \
  --source /work/root/build_ln/models/24_upernet_decoder_head_deconv.onnx \
  --segment-report /work/root/build_ln/segment25_report_deconv.json \
  --paired-inputs /work/root/cache_ln_deconv_head/output/cpu_and_migraphx_output.npz \
  --candidate /work/candidate/24_upernet_decoder_head_deconv_fpn4barrier.onnx \
  --report /work/candidate/build_report.json \
  >"${CANDIDATE_ROOT}/build.log" 2>&1
BUILD_RC=$?
set -e
ACTIVE_CONTAINER=''
printf '%s\n' "${BUILD_RC}" >"${CANDIDATE_ROOT}/build.exit"
if [[ "${BUILD_RC}" -ne 0 ]]; then
  printf 'build_failed\n' >"${CANDIDATE_ROOT}/RUN_STATUS"
  exit "${BUILD_RC}"
fi
test -s "${CANDIDATE}"
test -s "${BUILD_REPORT}"
mkdir "${SINGLE_ROOT}"
printf 'single_running\n' >"${CANDIDATE_ROOT}/RUN_STATUS"

(
  while true; do
    date -Ins
    docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' \
      phase11-fp32-segment25-headbarrier-single-node3 2>/dev/null || true
    grep '^MemAvailable:' /proc/meminfo || true
    sleep 15
  done
) >"${SINGLE_ROOT}/telemetry.log" 2>&1 &
TELEMETRY_PID=$!
ACTIVE_CONTAINER='phase11-fp32-segment25-headbarrier-single-node3'
set +e
docker run --rm --name "${ACTIVE_CONTAINER}" --entrypoint python3 \
  --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
  -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${ROOT}:/work/root:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${CANDIDATE_ROOT}:/work/candidate:rw" \
  -v "${SAMPLE}:/work/sample.pt:ro" -w /work "${IMAGE}" \
  /work/tools/evaluate_phase11_fp32_segment25_headbarrier_single.py \
  --root /work/root --sample /work/sample.pt \
  --candidate /work/candidate/24_upernet_decoder_head_deconv_fpn4barrier.onnx \
  --build-report /work/candidate/build_report.json \
  --head-cache /work/candidate/single_v1/segment_24_fpn4barrier.mxr \
  --output-dir /work/candidate/single_v1/output \
  >"${SINGLE_ROOT}/evaluation.log" 2>&1
SINGLE_RC=$?
set -e
ACTIVE_CONTAINER=''
kill "${TELEMETRY_PID}" 2>/dev/null || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
TELEMETRY_PID=''
printf '%s\n' "${SINGLE_RC}" >"${SINGLE_ROOT}/evaluation.exit"
if [[ "${SINGLE_RC}" -ne 0 ]]; then
  printf 'single_failed\n' >"${CANDIDATE_ROOT}/RUN_STATUS"
  exit "${SINGLE_RC}"
fi
python3 - "${SINGLE_ROOT}/output/result.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("status") != "diagnostic_completed" or not data.get("claims", {}).get("task90_eligible"):
    raise SystemExit(91)
PY

mkdir "${TEST90_ROOT}"
printf 'test90_running\n' >"${CANDIDATE_ROOT}/RUN_STATUS"
(
  while true; do
    date -Ins
    docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' \
      phase11-fp32-segment25-headbarrier-test90-node3 2>/dev/null || true
    grep '^MemAvailable:' /proc/meminfo || true
    sleep 15
  done
) >"${TEST90_ROOT}/telemetry.log" 2>&1 &
TELEMETRY_PID=$!
ACTIVE_CONTAINER='phase11-fp32-segment25-headbarrier-test90-node3'
set +e
docker run --rm --name "${ACTIVE_CONTAINER}" --entrypoint python3 \
  --memory=20g --pids-limit=1024 \
  -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${ROOT}:/work/root:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${CANDIDATE_ROOT}:/work/candidate:rw" \
  -v "${PROJECT}:/workspace:ro" -w /work "${IMAGE}" \
  /work/tools/evaluate_phase11_fp32_segment25_headbarrier_test90.py \
  --root /work/root \
  --candidate /work/candidate/24_upernet_decoder_head_deconv_fpn4barrier.onnx \
  --build-report /work/candidate/build_report.json \
  --head-cache /work/candidate/single_v1/segment_24_fpn4barrier.mxr \
  --single-result /work/candidate/single_v1/output/result.json \
  --base-evaluator /workspace/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py \
  --helper /workspace/evaluate_full_test_k100_onnx_int8.py \
  --fp32-result /workspace/outputs/k100_full_test_fp32/20260807-171433/result.json \
  --fp32-csv /workspace/outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv \
  --fp32-tensors /workspace/outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt \
  --frozen-inputs /workspace/phase11_frozen_test90/frozen_fp32_test90_inputs.npz \
  --frozen-manifest /workspace/phase11_frozen_test90/manifest.json \
  --output-dir /work/candidate/test90_v1/output \
  >"${TEST90_ROOT}/evaluation.log" 2>&1
TEST90_RC=$?
set -e
ACTIVE_CONTAINER=''
kill "${TELEMETRY_PID}" 2>/dev/null || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
TELEMETRY_PID=''
printf '%s\n' "${TEST90_RC}" >"${TEST90_ROOT}/evaluation.exit"

{
  date -Ins
  printf 'build_rc=%s\nsingle_rc=%s\ntest90_rc=%s\n' "${BUILD_RC}" "${SINGLE_RC}" "${TEST90_RC}"
  free -h
  /usr/local/hyhal/bin/hy-smi || true
  find "${CANDIDATE_ROOT}" -maxdepth 3 -type f -print0 | sort -z | xargs -0 sha256sum
} >"${CANDIDATE_ROOT}/host_post.txt" 2>&1
if [[ "${TEST90_RC}" -eq 0 ]]; then
  printf 'completed_task90_passed\n' >"${CANDIDATE_ROOT}/RUN_STATUS"
else
  printf 'completed_task90_failed\n' >"${CANDIDATE_ROOT}/RUN_STATUS"
fi
test ! -f "${TEST90_ROOT}/output/result.json" || cat "${TEST90_ROOT}/output/result.json"
exit "${TEST90_RC}"
