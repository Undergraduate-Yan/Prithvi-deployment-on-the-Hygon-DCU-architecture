#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -lt 6 || "$#" -gt 7 ]]; then
  echo "usage: $0 BUNDLE_ROOT CANDIDATE_ROOT TEST90_ROOT SAMPLE FP32_SUMMARY RUN_ROOT [M0,M1,...]" >&2
  exit 64
fi

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BUNDLE_ROOT="$(realpath -e -- "$1")"
CANDIDATE_ROOT="$(realpath -e -- "$2")"
TEST90_ROOT="$(realpath -e -- "$3")"
SAMPLE="$(realpath -e -- "$4")"
FP32_SUMMARY="$(realpath -e -- "$5")"
RUN_PARENT="$(realpath -m -- "$(dirname -- "$6")")"
RUN_ROOT="${RUN_PARENT}/$(basename -- "$6")"
SCHEDULE="${7:-M0,M1,M2,M3,M4}"
FP32_VRAM_RESULT_HOST="${FP32_VRAM_RESULT:-}"
if [[ -n "${FP32_VRAM_RESULT_HOST}" ]]; then
  FP32_VRAM_RESULT_HOST="$(realpath -e -- "${FP32_VRAM_RESULT_HOST}")"
fi

COMMON="${TOOLS}/phase11_mixed_precision_common.py"
TEST90_AGGREGATOR="${TOOLS}/aggregate_phase11_mixed_precision_test90.py"
FINGERPRINTER="${TOOLS}/fingerprint_phase11_mixed_precision_performance_runtime.py"
BENCHMARK="${TOOLS}/benchmark_phase11_mixed_precision_task_equivalence_trial.py"
PERF_AGGREGATOR="${TOOLS}/aggregate_phase11_mixed_precision_task_equivalence.py"
VRAM_MEASURER="${TOOLS}/measure_phase11_mixed_precision_k100_vram.py"
PARETO_SUMMARIZER="${TOOLS}/summarize_phase11_mixed_precision_capacity_pareto.py"
PROTOCOL="${TOOLS}/PHASE11_MIXED_PRECISION_TASK_EQUIVALENCE_PROTOCOL_V1.json"
VRAM_COUNTER='/sys/class/drm/card1/device/mem_info_vram_used'
GPU_BUSY_COUNTER='/sys/class/drm/card1/device/gpu_busy_percent'
RUNTIME_TOOLS_HOST='/var/tmp/phase11_int8_backbone_segment25_evidence_bundle_20260817/tools'
LSMOD_SHIM="${RUNTIME_TOOLS_HOST}/lsmod"

exec 9>/var/tmp/phase11-mixed-precision-formal-performance-node2.lock
flock -n 9 || exit 75

ACTIVE_CONTAINER=''
cleanup() {
  if [[ -n "${ACTIVE_CONTAINER}" ]]; then
    docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test -d /opt/hyhal
test -r "${VRAM_COUNTER}"
test "$(stat -c %s -- "${LSMOD_SHIM}")" = 819664
test "$(sha256sum -- "${LSMOD_SHIM}" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
mkdir -p -- "${RUN_PARENT}"
test ! -e "${RUN_ROOT}"

case "${CANDIDATE_ROOT}/" in
  "${BUNDLE_ROOT}/"*) ;;
  *) echo 'CANDIDATE_ROOT must be contained by BUNDLE_ROOT' >&2; exit 64 ;;
esac
CANDIDATE_REL="$(realpath --relative-to="${BUNDLE_ROOT}" "${CANDIDATE_ROOT}")"

IFS=',' read -r -a CANDIDATES <<<"${SCHEDULE}"
test "${#CANDIDATES[@]}" -ge 1
test "${#CANDIDATES[@]}" -le 6
PREVIOUS=-1
for CANDIDATE in "${CANDIDATES[@]}"; do
  [[ "${CANDIDATE}" =~ ^M[0-5]$ ]]
  INDEX="${CANDIDATE#M}"
  test "${INDEX}" -gt "${PREVIOUS}"
  PREVIOUS="${INDEX}"
  test -f "${CANDIDATE_ROOT}/${CANDIDATE}/mixed_precision_manifest.final.json"
  test -f "${TEST90_ROOT}/${CANDIDATE}/three_run_summary/summary.json"
done

lock_file() {
  local path="$1" expected_size="$2" expected_sha="$3"
  test "$(stat -c %s -- "${path}")" = "${expected_size}"
  test "$(sha256sum -- "${path}" | awk '{print $1}')" = "${expected_sha}"
}

lock_file "${COMMON}" 23030 f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8
lock_file "${TEST90_AGGREGATOR}" 9628 717aa6c5e10b889c082acb4626105ad708c9006acf70380793f0b93c71fbd159
lock_file "${FINGERPRINTER}" 5516 f38e24ea72c02a164ec311ce4521f119da1e0f397fc796ea380dc62bb2167577
lock_file "${BENCHMARK}" 18067 836b890a6fecbaf39658578dd3b0d7bf22c6b1d60ee5cacc6a0a4c3d514fd113
lock_file "${PERF_AGGREGATOR}" 17844 8ef4d2ecf23052d5e49398311a374a14a3468aa7b2c32baf66cc74902cc0c80a
lock_file "${VRAM_MEASURER}" 16702 4a554b78a67055e869d3663e87b80ab27ae8ffead209f18531c41075a3bf8a50
lock_file "${PARETO_SUMMARIZER}" 21316 b80f6cd2728fce0c1395fa0fb9bfb7cb5c1a6eaf0e37ffaaa96d3ed4c6dfe778
lock_file "${PROTOCOL}" 6754 d9515fe2537014ac13428b6fab3024dd16510823ee7a77278cb9a6d715a2f5b4

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
test "${IMAGE_ID}" = "${IMAGE}"

mkdir -- "${RUN_ROOT}"
docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
cp -- "${PROTOCOL}" "${RUN_ROOT}/protocol.json"
{
  date -Ins
  hostname
  printf 'image_id=%s\n' "${IMAGE_ID}"
  printf 'schedule=%s\n' "${SCHEDULE}"
  printf 'bundle_root=%s\n' "${BUNDLE_ROOT}"
  printf 'candidate_root=%s\n' "${CANDIDATE_ROOT}"
  printf 'test90_root=%s\n' "${TEST90_ROOT}"
  printf 'sample=%s\n' "${SAMPLE}"
  printf 'fp32_summary=%s\n' "${FP32_SUMMARY}"
  printf 'fp32_vram_result=%s\n' "${FP32_VRAM_RESULT_HOST:-unavailable_not_supplied}"
  printf 'vram_counter=%s\n' "${VRAM_COUNTER}"
  printf 'latency_and_vram_processes_separate=true\n'
} >"${RUN_ROOT}/host_protocol.txt"
printf 'runtime_fingerprint\n' >"${RUN_ROOT}/RUN_STATUS"

set +e
docker run --rm --name phase11-mixed-runtime-fingerprint-node2 \
  --entrypoint python3 \
  -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -w /work \
  "${IMAGE}" /work/tools/fingerprint_phase11_mixed_precision_performance_runtime.py \
  >"${RUN_ROOT}/runtime_fingerprint.json" 2>"${RUN_ROOT}/runtime_fingerprint.stderr"
RC=$?
set -e
printf '%s\n' "${RC}" >"${RUN_ROOT}/runtime_fingerprint.exit"
test "${RC}" -eq 0

run_created_container() {
  local name="$1" evidence_dir="$2"
  shift 2
  mkdir -p -- "${evidence_dir}"
  ACTIVE_CONTAINER="${name}"
  local cid rc
  cid="$(docker create --name "${name}" "$@")"
  printf '%s\n' "${cid}" >"${evidence_dir}/container_id.txt"
  docker start "${cid}" >/dev/null
  docker inspect "${cid}" >"${evidence_dir}/container_inspect_started.json"
  rc="$(docker wait "${cid}")"
  docker logs "${cid}" >"${evidence_dir}/container.log" 2>&1 || true
  docker inspect "${cid}" >"${evidence_dir}/container_inspect_exited.json"
  printf '%s\n' "${rc}" >"${evidence_dir}/container.exit"
  docker rm "${cid}" >/dev/null
  ACTIVE_CONTAINER=''
  return "${rc}"
}

printf 'timed_trials\n' >"${RUN_ROOT}/RUN_STATUS"
COUNT="${#CANDIDATES[@]}"
for TRIAL in 1 2 3; do
  OFFSET=$((TRIAL - 1))
  for ((POSITION=1; POSITION<=COUNT; POSITION++)); do
    BASE_INDEX=$(((POSITION - 1 + OFFSET) % COUNT))
    CANDIDATE="${CANDIDATES[BASE_INDEX]}"
    EVIDENCE="${RUN_ROOT}/trial_$(printf '%02d' "${TRIAL}")/position_$(printf '%02d' "${POSITION}")_${CANDIDATE}"
    mkdir -p -- "${EVIDENCE}"
    CONTAINER="phase11-mixed-perf-t${TRIAL}-p${POSITION}-${CANDIDATE}-node2"
    set +e
    run_created_container "${CONTAINER}" "${EVIDENCE}" \
      --entrypoint python3 --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
      -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
      -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
      -v "${TOOLS}:/work/tools:ro" \
      -v "${BUNDLE_ROOT}:/work/bundle:ro" -v "${TEST90_ROOT}:/work/test90:ro" \
      -v "${SAMPLE}:/work/sample.pt:ro" -v "${RUN_ROOT}:/work/run:rw" -w /work \
      "${IMAGE}" /work/tools/benchmark_phase11_mixed_precision_task_equivalence_trial.py \
      --bundle /work/bundle \
      --manifest "/work/bundle/${CANDIDATE_REL}/${CANDIDATE}/mixed_precision_manifest.final.json" \
      --test90-three-run-summary "/work/test90/${CANDIDATE}/three_run_summary/summary.json" \
      --sample /work/sample.pt --trial-index "${TRIAL}" --candidate-position "${POSITION}" \
      --candidate-schedule "${SCHEDULE}" --container-image-id "${IMAGE_ID}" \
      --runtime-fingerprint /work/run/runtime_fingerprint.json \
      --output-dir "/work/run/trial_$(printf '%02d' "${TRIAL}")/position_$(printf '%02d' "${POSITION}")_${CANDIDATE}/output"
    RC=$?
    set -e
    if [[ "${RC}" -ne 0 ]]; then
      printf 'timed_trial_failed trial=%s position=%s candidate=%s rc=%s\n' \
        "${TRIAL}" "${POSITION}" "${CANDIDATE}" "${RC}" >"${RUN_ROOT}/RUN_STATUS"
      exit "${RC}"
    fi
  done
done

printf 'performance_aggregate\n' >"${RUN_ROOT}/RUN_STATUS"
set +e
run_created_container phase11-mixed-perf-aggregate-node2 "${RUN_ROOT}/performance_aggregate_container" \
  --entrypoint python3 -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" -v "${TOOLS}:/work/tools:ro" \
  -v "${RUN_ROOT}:/work/run:rw" \
  -v "${FP32_SUMMARY}:/work/fp32_summary.json:ro" -w /work \
  "${IMAGE}" /work/tools/aggregate_phase11_mixed_precision_task_equivalence.py \
  --run-root /work/run --candidate-schedule "${SCHEDULE}" \
  --fp32-summary /work/fp32_summary.json
AGGREGATE_RC=$?
set -e
if [[ "${AGGREGATE_RC}" -ne 0 && "${AGGREGATE_RC}" -ne 2 ]]; then
  printf 'performance_aggregate_failed rc=%s\n' "${AGGREGATE_RC}" >"${RUN_ROOT}/RUN_STATUS"
  exit "${AGGREGATE_RC}"
fi
test -f "${RUN_ROOT}/summary.json"

wait_for_idle_k100() {
  test -z "$(docker ps -q)"
  if [[ -r "${GPU_BUSY_COUNTER}" ]]; then
    for _ in 1 2 3 4 5; do
      test "$(tr -d '\r\n' <"${GPU_BUSY_COUNTER}")" = 0
      sleep 0.2
    done
  fi
}

printf 'isolated_vram\n' >"${RUN_ROOT}/RUN_STATUS"
for CANDIDATE in "${CANDIDATES[@]}"; do
  wait_for_idle_k100
  EVIDENCE="${RUN_ROOT}/vram/${CANDIDATE}"
  mkdir -p -- "${EVIDENCE}"
  CONTAINER="phase11-mixed-vram-${CANDIDATE}-node2"
  set +e
  run_created_container "${CONTAINER}" "${EVIDENCE}" \
    --entrypoint python3 --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
    -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
    -v "${TOOLS}:/work/tools:ro" \
    -v "${BUNDLE_ROOT}:/work/bundle:ro" -v "${TEST90_ROOT}:/work/test90:ro" \
    -v "${SAMPLE}:/work/sample.pt:ro" -v "${RUN_ROOT}:/work/run:rw" -w /work \
    "${IMAGE}" /work/tools/measure_phase11_mixed_precision_k100_vram.py \
    --bundle /work/bundle \
    --manifest "/work/bundle/${CANDIDATE_REL}/${CANDIDATE}/mixed_precision_manifest.final.json" \
    --test90-three-run-summary "/work/test90/${CANDIDATE}/three_run_summary/summary.json" \
    --sample /work/sample.pt --container-image-id "${IMAGE_ID}" \
    --runtime-fingerprint /work/run/runtime_fingerprint.json \
    --vram-counter "${VRAM_COUNTER}" --output-dir "/work/run/vram/${CANDIDATE}/output"
  RC=$?
  set -e
  if [[ "${RC}" -ne 0 ]]; then
    printf 'vram_failed candidate=%s rc=%s\n' "${CANDIDATE}" "${RC}" >"${RUN_ROOT}/RUN_STATUS"
    exit "${RC}"
  fi
done

PARETO_ARGS=(
  --bundle /work/bundle
  --performance-summary /work/run/summary.json
  --fp32-performance-summary /work/fp32_summary.json
  --output-dir /work/run/pareto
)
FP32_VRAM_MOUNT=()
if [[ -n "${FP32_VRAM_RESULT_HOST}" ]]; then
  FP32_VRAM_MOUNT=(-v "${FP32_VRAM_RESULT_HOST}:/work/fp32_vram_result.json:ro")
  PARETO_ARGS+=(--fp32-vram-result /work/fp32_vram_result.json)
fi
for CANDIDATE in "${CANDIDATES[@]}"; do
  PARETO_ARGS+=(
    --manifest "${CANDIDATE}=/work/bundle/${CANDIDATE_REL}/${CANDIDATE}/mixed_precision_manifest.final.json"
    --test90-summary "${CANDIDATE}=/work/test90/${CANDIDATE}/three_run_summary/summary.json"
    --vram-result "${CANDIDATE}=/work/run/vram/${CANDIDATE}/output/result.json"
  )
done

printf 'pareto_summary\n' >"${RUN_ROOT}/RUN_STATUS"
set +e
run_created_container phase11-mixed-pareto-node2 "${RUN_ROOT}/pareto_container" \
  --entrypoint python3 -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" -v "${TOOLS}:/work/tools:ro" \
  -v "${BUNDLE_ROOT}:/work/bundle:ro" \
  -v "${TEST90_ROOT}:/work/test90:ro" -v "${RUN_ROOT}:/work/run:rw" \
  -v "${FP32_SUMMARY}:/work/fp32_summary.json:ro" "${FP32_VRAM_MOUNT[@]}" -w /work \
  "${IMAGE}" /work/tools/summarize_phase11_mixed_precision_capacity_pareto.py \
  "${PARETO_ARGS[@]}"
RC=$?
set -e
if [[ "${RC}" -ne 0 ]]; then
  printf 'pareto_failed rc=%s\n' "${RC}" >"${RUN_ROOT}/RUN_STATUS"
  exit "${RC}"
fi

if [[ "${AGGREGATE_RC}" -eq 0 ]]; then
  printf 'completed_all_candidates_stable\n' >"${RUN_ROOT}/RUN_STATUS"
else
  printf 'completed_unstable_candidates_excluded_from_pareto\n' >"${RUN_ROOT}/RUN_STATUS"
fi
{
  date -Ins
  hostname
  /usr/local/hyhal/bin/hy-smi || true
  find "${RUN_ROOT}" -type f -print0 | sort -z | xargs -0 sha256sum
} >"${RUN_ROOT}/host_post.txt" 2>&1
cat "${RUN_ROOT}/pareto/pareto_summary.json"
exit "${AGGREGATE_RC}"
