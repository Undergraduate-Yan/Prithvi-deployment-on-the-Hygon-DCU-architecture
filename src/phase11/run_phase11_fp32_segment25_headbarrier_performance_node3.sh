#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE='sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291'
PROJECT='/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100'
ROOT='/var/tmp/20260817-phase11-fp32-backbone-node3'
TOOLS='/var/tmp/phase11_fp32_segment25_headbarrier_tools_v1'
CANDIDATE_ROOT='/var/tmp/20260818-phase11-fp32-segment25-headbarrier-node3'
SAMPLE="${PROJECT}/outputs/k100_real_sample_parity/sample_000_20260807-162919/sample_and_logits.pt"
TRIAL_SCRIPT="${TOOLS}/benchmark_phase11_fp32_segment25_headbarrier_all_resident_trial.py"
AGGREGATOR="${TOOLS}/aggregate_phase11_fp32_segment25_headbarrier_performance.py"
RUN="${CANDIDATE_ROOT}/performance_25segment_all_resident_v2"

exec 9>/var/tmp/phase11-fp32-segment25-headbarrier-performance-node3.lock
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
test -d "${CANDIDATE_ROOT}"
test -d "${TOOLS}"
test "$(tr -d '\r\n' <"${CANDIDATE_ROOT}/RUN_STATUS")" = completed_task90_passed
test ! -e "${RUN}"
AVAILABLE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
test "${AVAILABLE_KB}" -ge 41943040

# Immutable benchmark programs.
test "$(stat -c %s "${TRIAL_SCRIPT}")" = 25591
test "$(sha256sum "${TRIAL_SCRIPT}" | awk '{print $1}')" = 31af48acd99b1ddbc614854bbf99721ca0f1d86bc984b21605883740a7f743dd
test "$(stat -c %s "${AGGREGATOR}")" = 13052
test "$(sha256sum "${AGGREGATOR}" | awk '{print $1}')" = 25bfeaefa2982c33e304bacae188fd4e23b30097b46c97f158d21acab2df9706
test "$(stat -c %s "${ROOT}/tools/lsmod")" = 819664
test "$(sha256sum "${ROOT}/tools/lsmod" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12

# Frozen source, candidate, cache, and admission evidence.
test "$(stat -c %s "${ROOT}/build_ln/segment25_report_deconv.json")" = 24959
test "$(sha256sum "${ROOT}/build_ln/segment25_report_deconv.json" | awk '{print $1}')" = c94692d130a0eb924290b3d56f790e166535fa3049fd9f3e72ef1fe16085c90f
test "$(cat "${ROOT}/cache_ln_remaining/launcher.exit")" = 0
test "$(cat "${ROOT}/cache_ln_remaining/LATEST_COMPLETED_SEGMENT")" = 24
test "$(stat -c %s "${CANDIDATE_ROOT}/24_upernet_decoder_head_deconv_fpn4barrier.onnx")" = 60551388
test "$(sha256sum "${CANDIDATE_ROOT}/24_upernet_decoder_head_deconv_fpn4barrier.onnx" | awk '{print $1}')" = 593db691abcb2e07dd8df2f1fabc8315eeab08380161c93199778bc1963f299f
test "$(stat -c %s "${CANDIDATE_ROOT}/build_report.json")" = 2511
test "$(sha256sum "${CANDIDATE_ROOT}/build_report.json" | awk '{print $1}')" = 1b288841074cda9134ff91a5e763b8116e3935a2e3124f6b69813f2def6b86d9
test "$(stat -c %s "${CANDIDATE_ROOT}/single_v1/segment_24_fpn4barrier.mxr")" = 61093641
test "$(sha256sum "${CANDIDATE_ROOT}/single_v1/segment_24_fpn4barrier.mxr" | awk '{print $1}')" = e0c43b934aa235c1710468cfb76060e834e54672bf939e1c0a2a7a7096a6cb72
test "$(stat -c %s "${CANDIDATE_ROOT}/single_v1/output/result.json")" = 33109
test "$(sha256sum "${CANDIDATE_ROOT}/single_v1/output/result.json" | awk '{print $1}')" = 240479bbf29fef32165b908166c0550711384018d33e9a1a8ff75fdae3837655
test "$(stat -c %s "${CANDIDATE_ROOT}/test90_v1/output/result.json")" = 39291
test "$(sha256sum "${CANDIDATE_ROOT}/test90_v1/output/result.json" | awk '{print $1}')" = 0904da34078801106b6b414dcc17d1b612b448a997460563fc7055c4e6988b3b
test "$(stat -c %s "${SAMPLE}")" = 4820344
test "$(sha256sum "${SAMPLE}" | awk '{print $1}')" = 4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62

mkdir "${RUN}"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'track=same_protocol_fp32_reference\n'
  printf 'batch=1\nwarmup_per_scope=30\nmeasured_per_scope=100\nfresh_process_trials=3\n'
  printf 'primary_scope=model_only_25_resident_fixed_device_OrtValues\n'
  printf 'secondary_scope=host_input_H2D_25_resident_device_OrtValues_D2H_logits\n'
  printf 'session_load_excluded=true\ntrial_median_cv_gate_percent=5\n'
  printf 'candidate_sha256=593db691abcb2e07dd8df2f1fabc8315eeab08380161c93199778bc1963f299f\n'
  printf 'test90_result_sha256=0904da34078801106b6b414dcc17d1b612b448a997460563fc7055c4e6988b3b\n'
  printf 'mem_available_kb=%s\n' "${AVAILABLE_KB}"
} >"${RUN}/protocol.txt"
printf 'running\n' >"${RUN}/RUN_STATUS"

for TRIAL in 1 2 3; do
  TRIAL_ROOT="${RUN}/trial_${TRIAL}"
  CONTAINER="phase11-fp32-headbarrier-perf-${TRIAL}-node3"
  mkdir "${TRIAL_ROOT}"
  {
    date -Ins
    free -h
    /usr/local/hyhal/bin/hy-smi || true
    /usr/local/hyhal/bin/hy-smi --showmeminfo vram || true
  } >"${TRIAL_ROOT}/host_pre.txt" 2>&1

  (
    while true; do
      date -Ins
      docker stats --no-stream \
        --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' \
        "${CONTAINER}" 2>/dev/null || true
      /usr/local/hyhal/bin/hy-smi --showmeminfo vram 2>/dev/null || true
      sleep 3
    done
  ) >"${TRIAL_ROOT}/telemetry.log" 2>&1 &
  TELEMETRY_PID=$!

  ACTIVE_CONTAINER="${CONTAINER}"
  set +e
  docker run --rm --name "${CONTAINER}" --entrypoint python3 \
    --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
    -e PATH=/work/root/tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${ROOT}:/work/root:ro" \
    -v "${TOOLS}:/work/tools:ro" -v "${CANDIDATE_ROOT}:/work/candidate:ro" \
    -v "${SAMPLE}:/work/sample.pt:ro" -v "${TRIAL_ROOT}:/work/trial:rw" \
    -w /work "${IMAGE}" \
    /work/tools/benchmark_phase11_fp32_segment25_headbarrier_all_resident_trial.py \
    --root /work/root --candidate-root /work/candidate --sample /work/sample.pt \
    --trial-index "${TRIAL}" --output-dir /work/trial/output \
    >"${TRIAL_ROOT}/benchmark.log" 2>&1
  RC=$?
  set -e
  ACTIVE_CONTAINER=''
  kill "${TELEMETRY_PID}" 2>/dev/null || true
  wait "${TELEMETRY_PID}" 2>/dev/null || true
  TELEMETRY_PID=''
  printf '%s\n' "${RC}" >"${TRIAL_ROOT}/benchmark.exit"
  {
    date -Ins
    free -h
    /usr/local/hyhal/bin/hy-smi || true
    /usr/local/hyhal/bin/hy-smi --showmeminfo vram || true
  } >"${TRIAL_ROOT}/host_post.txt" 2>&1
  if [[ "${RC}" -ne 0 ]]; then
    printf 'trial_%s_failed\n' "${TRIAL}" >"${RUN}/RUN_STATUS"
    test ! -f "${TRIAL_ROOT}/output/result.json" || cat "${TRIAL_ROOT}/output/result.json"
    exit "${RC}"
  fi
done

ACTIVE_CONTAINER='phase11-fp32-headbarrier-perf-aggregate-node3'
set +e
docker run --rm --name "${ACTIVE_CONTAINER}" --entrypoint python3 \
  -v "${TOOLS}:/work/tools:ro" -v "${RUN}:/work/run:rw" -w /work "${IMAGE}" \
  /work/tools/aggregate_phase11_fp32_segment25_headbarrier_performance.py \
  --run-root /work/run >"${RUN}/aggregate.log" 2>&1
RC=$?
set -e
ACTIVE_CONTAINER=''
printf '%s\n' "${RC}" >"${RUN}/aggregate.exit"
if [[ "${RC}" -eq 0 ]]; then
  printf 'completed_passed\n' >"${RUN}/RUN_STATUS"
else
  printf 'completed_unstable_or_failed\n' >"${RUN}/RUN_STATUS"
fi
{
  date -Ins
  free -h
  /usr/local/hyhal/bin/hy-smi || true
  find "${RUN}" -maxdepth 3 -type f -print0 | sort -z | xargs -0 sha256sum
} >"${RUN}/host_post.txt" 2>&1
test ! -f "${RUN}/summary.json" || cat "${RUN}/summary.json"
exit "${RC}"
