#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -lt 4 || "$#" -gt 6 ]]; then
  echo "usage: $0 BUNDLE_ROOT CANDIDATE_ROOT SAMPLE RUN_ROOT [M0,M1,M2,M3,M4,M5] [REPETITIONS]" >&2
  exit 64
fi

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LAUNCHER="$(realpath -e -- "${BASH_SOURCE[0]}")"
BUNDLE_ROOT="$(realpath -e -- "$1")"
CANDIDATE_ROOT="$(realpath -e -- "$2")"
SAMPLE="$(realpath -e -- "$3")"
RUN_PARENT="$(realpath -m -- "$(dirname -- "$4")")"
RUN_ROOT="${RUN_PARENT}/$(basename -- "$4")"
SCHEDULE="${5:-M0,M1,M2,M3,M4,M5}"
REPETITIONS="${6:-20}"
TASK_LIMIT="${PHASE11_KERNEL_TASK_LIMIT:-0}"
CONTAINER_TIMEOUT_SECONDS="${PHASE11_KERNEL_CONTAINER_TIMEOUT_SECONDS:-1800}"
M5_CANDIDATE_ROOT="${PHASE11_M5_CANDIDATE_ROOT:-}"
if [[ -n "${M5_CANDIDATE_ROOT}" ]]; then
  M5_CANDIDATE_ROOT="$(realpath -e -- "${M5_CANDIDATE_ROOT}")"
fi

COMMON="${TOOLS}/phase11_mixed_precision_common.py"
FINGERPRINTER="${TOOLS}/fingerprint_phase11_mixed_precision_performance_runtime.py"
PLANNER="${TOOLS}/plan_phase11_mixed_precision_kernel_evidence.py"
PREPASS="${TOOLS}/materialize_phase11_mixed_precision_block_input.py"
TARGET="${TOOLS}/profile_phase11_mixed_precision_block.py"
SUMMARIZER="${TOOLS}/summarize_phase11_mixed_precision_kernel_evidence.py"
PROTOCOL="${TOOLS}/PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_PROTOCOL_V3.json"
RUNTIME_TOOLS_HOST='/var/tmp/phase11_int8_backbone_segment25_evidence_bundle_20260817/tools'
LSMOD_SHIM="${RUNTIME_TOOLS_HOST}/lsmod"

case "${SAMPLE,,}" in
  *.npy) SAMPLE_CONTAINER='/work/sample.npy' ;;
  *.npz) SAMPLE_CONTAINER='/work/sample.npz' ;;
  *.pt) SAMPLE_CONTAINER='/work/sample.pt' ;;
  *.pth) SAMPLE_CONTAINER='/work/sample.pth' ;;
  *) echo 'SAMPLE must have a .npy, .npz, .pt, or .pth suffix' >&2; exit 64 ;;
esac

exec 9>/var/tmp/phase11-mixed-precision-kernel-evidence-m0-m5-node2.lock
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
command -v timeout >/dev/null
command -v sha256sum >/dev/null
command -v flock >/dev/null
mkdir -p -- "${RUN_PARENT}"
test ! -e "${RUN_ROOT}"
[[ "${REPETITIONS}" =~ ^[0-9]+$ ]]
test "${REPETITIONS}" -ge 1
test "${REPETITIONS}" -le 1000
[[ "${TASK_LIMIT}" =~ ^[0-9]+$ ]]
test "${TASK_LIMIT}" -le 33
[[ "${CONTAINER_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]
test "${CONTAINER_TIMEOUT_SECONDS}" -ge 60
test "${CONTAINER_TIMEOUT_SECONDS}" -le 7200
case "${CANDIDATE_ROOT}/" in
  "${BUNDLE_ROOT}/"*) ;;
  *) echo 'CANDIDATE_ROOT must be contained by BUNDLE_ROOT' >&2; exit 64 ;;
esac
if [[ -n "${M5_CANDIDATE_ROOT}" ]]; then
  case "${M5_CANDIDATE_ROOT}/" in
    "${BUNDLE_ROOT}/"*) ;;
    *) echo 'PHASE11_M5_CANDIDATE_ROOT must be contained by BUNDLE_ROOT' >&2; exit 64 ;;
  esac
fi

lock_file() {
  local path="$1" expected_size="$2" expected_sha="$3"
  test "$(stat -c %s -- "${path}")" = "${expected_size}"
  test "$(sha256sum -- "${path}" | awk '{print $1}')" = "${expected_sha}"
}

lock_file "${COMMON}" 23030 f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8
lock_file "${FINGERPRINTER}" 5516 f38e24ea72c02a164ec311ce4521f119da1e0f397fc796ea380dc62bb2167577
lock_file "${PREPASS}" 11960 b28d9e9ea7851f877c835095ab0b48900607c5ff7d8164ec20042ef37f45b3e6
lock_file "${TARGET}" 16603 58893c10fc21c10a09116579e39a0b9e6ad3cae3ab86528cbd107d067f3cba71
lock_file "${PLANNER}" 13211 1cf9dd90a40b5a6a6f2c8311870c9e498ed4f56974cc5b3f8e704b2a5a27f9bc
lock_file "${SUMMARIZER}" 45123 794b0abb7c3ff3eab6230c66db6cb8b30c6ed2a5cd9edc8556f818ed4eb77e6b
lock_file "${PROTOCOL}" 5886 ffb2342e61409aa6da68c0c8de2be950bd86fd6ab3aab8574d7c287c11c2f349
lock_file "${LSMOD_SHIM}" 819664 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test -x "${LSMOD_SHIM}"
test -f "${SAMPLE}"

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
test "${IMAGE_ID}" = "${IMAGE}"
docker run --rm --entrypoint test "${IMAGE}" -x /opt/dtk/bin/hipprof

IFS=',' read -r -a CANDIDATES <<<"${SCHEDULE}"
test "${#CANDIDATES[@]}" -ge 1
test "${#CANDIDATES[@]}" -le 6
PREVIOUS=-1
MANIFEST_ARGS=()
for CANDIDATE in "${CANDIDATES[@]}"; do
  [[ "${CANDIDATE}" =~ ^M[0-5]$ ]]
  INDEX="${CANDIDATE#M}"
  test "${INDEX}" -gt "${PREVIOUS}"
  PREVIOUS="${INDEX}"
  SELECTED_CANDIDATE_ROOT="${CANDIDATE_ROOT}"
  if [[ "${CANDIDATE}" = M5 && -n "${M5_CANDIDATE_ROOT}" ]]; then
    SELECTED_CANDIDATE_ROOT="${M5_CANDIDATE_ROOT}"
  fi
  MANIFEST="${SELECTED_CANDIDATE_ROOT}/${CANDIDATE}/mixed_precision_manifest.final.json"
  test -f "${MANIFEST}"
  RELATIVE="$(realpath --relative-to="${BUNDLE_ROOT}" "${MANIFEST}")"
  MANIFEST_ARGS+=(--manifest "${CANDIDATE}=/work/bundle/${RELATIVE}")
done

mkdir -- "${RUN_ROOT}"
mkdir -- "${RUN_ROOT}/profiles" "${RUN_ROOT}/tool_snapshot"
docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
cp -- "${PROTOCOL}" "${RUN_ROOT}/protocol.json"
cp -- "${COMMON}" "${FINGERPRINTER}" "${PLANNER}" "${PREPASS}" "${TARGET}" "${SUMMARIZER}" \
  "${PROTOCOL}" "${LAUNCHER}" "${RUN_ROOT}/tool_snapshot/"
cp -- "${LSMOD_SHIM}" "${RUN_ROOT}/tool_snapshot/lsmod"
{
  date -Ins
  hostname
  printf 'image_id=%s\n' "${IMAGE_ID}"
  printf 'schedule=%s\n' "${SCHEDULE}"
  printf 'repetitions=%s\n' "${REPETITIONS}"
  printf 'task_limit=%s\n' "${TASK_LIMIT}"
  printf 'container_timeout_seconds=%s\n' "${CONTAINER_TIMEOUT_SECONDS}"
  printf 'bundle_root=%s\n' "${BUNDLE_ROOT}"
  printf 'candidate_root=%s\n' "${CANDIDATE_ROOT}"
  printf 'm5_candidate_root=%s\n' "${M5_CANDIDATE_ROOT:-${CANDIDATE_ROOT}}"
  printf 'sample=%s\n' "${SAMPLE}"
  printf 'sample_container=%s\n' "${SAMPLE_CONTAINER}"
  printf 'static_lsmod_shim=%s\n' "${LSMOD_SHIM}"
  printf 'static_lsmod_sha256=9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12\n'
  printf 'profile_isolation=one_untracked_prepass_container_plus_one_fresh_direct_hipprof_container_per_unique_model_cache_identity\n'
  printf 'trace_mode=direct_outer_hipprof_no_trace_off_no_session_no_dynamic_control\n'
  printf 'trace_scope=target_segment_only_upstream_segments_zero\n'
  printf 'provider_placement_is_kernel_proof=false\n'
} >"${RUN_ROOT}/host_protocol.txt"
printf 'runtime_fingerprint\n' >"${RUN_ROOT}/RUN_STATUS"

docker run --rm --name phase11-mixed-kernel-fingerprint-node2 \
  --entrypoint /usr/bin/python3 \
  -e PATH=/work/runtime_tools:/work/tools:/opt/dtk/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -w /work \
  "${IMAGE}" /work/tools/fingerprint_phase11_mixed_precision_performance_runtime.py \
  >"${RUN_ROOT}/runtime_fingerprint.json" 2>"${RUN_ROOT}/runtime_fingerprint.stderr"

PLAN_REQUIRE=(--no-require-all-m0-m5)
if [[ "${SCHEDULE}" = M0,M1,M2,M3,M4,M5 ]]; then
  PLAN_REQUIRE=(--require-all-m0-m5)
fi
PLAN_LIMIT=()
if [[ "${TASK_LIMIT}" -gt 0 ]]; then
  PLAN_LIMIT=(--max-profile-tasks "${TASK_LIMIT}")
fi
printf 'identity_locked_plan\n' >"${RUN_ROOT}/RUN_STATUS"
docker run --rm --name phase11-mixed-kernel-plan-node2 \
  --entrypoint /usr/bin/python3 \
  -e PATH=/work/runtime_tools:/work/tools:/opt/dtk/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${BUNDLE_ROOT}:/work/bundle:ro" \
  -v "${RUN_ROOT}:/work/run:rw" -w /work \
  "${IMAGE}" /work/tools/plan_phase11_mixed_precision_kernel_evidence.py \
  --bundle /work/bundle "${MANIFEST_ARGS[@]}" "${PLAN_REQUIRE[@]}" "${PLAN_LIMIT[@]}" \
  --output-dir /work/run/plan \
  >"${RUN_ROOT}/plan_container.log" 2>&1

test -s "${RUN_ROOT}/plan/profile_plan.json"
test -s "${RUN_ROOT}/plan/profile_plan.tsv"

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
  if rc="$(timeout --foreground "${CONTAINER_TIMEOUT_SECONDS}" docker wait "${cid}")"; then
    :
  else
    local wait_rc=$?
    docker logs "${cid}" >"${evidence_dir}/container.log" 2>&1 || true
    docker inspect "${cid}" >"${evidence_dir}/container_inspect_timeout.json" || true
    printf '%s\n' "${wait_rc}" >"${evidence_dir}/container.exit"
    printf 'docker_wait_failed_or_timed_out rc=%s timeout_seconds=%s\n' \
      "${wait_rc}" "${CONTAINER_TIMEOUT_SECONDS}" >"${evidence_dir}/wait_failure.txt"
    docker rm -f "${cid}" >/dev/null 2>&1 || true
    ACTIVE_CONTAINER=''
    return "${wait_rc}"
  fi
  docker logs "${cid}" >"${evidence_dir}/container.log" 2>&1 || true
  docker inspect "${cid}" >"${evidence_dir}/container_inspect_exited.json"
  printf '%s\n' "${rc}" >"${evidence_dir}/container.exit"
  docker rm "${cid}" >/dev/null
  ACTIVE_CONTAINER=''
  return "${rc}"
}

printf 'unique_block_profiles\n' >"${RUN_ROOT}/RUN_STATUS"
while IFS=$'\t' read -r ORDINAL EVIDENCE_KEY CANDIDATE MANIFEST_RELATIVE SEGMENT PRECISION MODEL_SHA CACHE_SHA; do
  [[ "${EVIDENCE_KEY}" =~ ^[a-z0-9_]+$ ]]
  [[ "${CANDIDATE}" =~ ^M[0-5]$ ]]
  [[ "${SEGMENT}" =~ ^([0-9]|1[0-9]|2[0-3])$ ]]
  [[ "${PRECISION}" = int8_qdq || "${PRECISION}" = fp16 ]]
  [[ "${MODEL_SHA}" =~ ^[0-9a-f]{64}$ ]]
  [[ "${CACHE_SHA}" =~ ^[0-9a-f]{64}$ ]]
  EVIDENCE_DIR="${RUN_ROOT}/profiles/${EVIDENCE_KEY}"
  PREPASS_CONTAINER="phase11-mixed-prepass-${ORDINAL}-${CANDIDATE}-b${SEGMENT}-node2"
  TRACE_CONTAINER="phase11-mixed-kernel-v3-${ORDINAL}-${CANDIDATE}-b${SEGMENT}-node2"
  printf 'boundary_prepass ordinal=%s key=%s candidate=%s block=%s\n' \
    "${ORDINAL}" "${EVIDENCE_KEY}" "${CANDIDATE}" "${SEGMENT}" >"${RUN_ROOT}/RUN_STATUS"
  set +e
  run_created_container "${PREPASS_CONTAINER}" "${EVIDENCE_DIR}/prepass_container" \
    --entrypoint /usr/bin/python3 --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
    -e PATH=/work/runtime_tools:/work/tools:/opt/dtk/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
    -v "${TOOLS}:/work/tools:ro" -v "${BUNDLE_ROOT}:/work/bundle:ro" \
    -v "${SAMPLE}:${SAMPLE_CONTAINER}:ro" \
    -v "${RUN_ROOT}:/work/run:rw" -w /work \
    "${IMAGE}" /work/tools/materialize_phase11_mixed_precision_block_input.py \
    --bundle /work/bundle --manifest "/work/bundle/${MANIFEST_RELATIVE}" \
    --candidate-id "${CANDIDATE}" --segment-index "${SEGMENT}" \
    --expected-precision "${PRECISION}" --expected-model-sha256 "${MODEL_SHA}" \
    --expected-cache-sha256 "${CACHE_SHA}" --sample "${SAMPLE_CONTAINER}" \
    --runtime-fingerprint /work/run/runtime_fingerprint.json --container-image-id "${IMAGE_ID}" \
    --output-dir "/work/run/profiles/${EVIDENCE_KEY}/prepass"
  PREPASS_RC=$?
  set -e
  if [[ "${PREPASS_RC}" -ne 0 ]]; then
    printf 'prepass_failed ordinal=%s key=%s candidate=%s block=%s rc=%s\n' \
      "${ORDINAL}" "${EVIDENCE_KEY}" "${CANDIDATE}" "${SEGMENT}" "${PREPASS_RC}" \
      >"${RUN_ROOT}/RUN_STATUS"
    exit "${PREPASS_RC}"
  fi

  BOUNDARY_INPUT="${EVIDENCE_DIR}/prepass/boundary_input.npy"
  PREPASS_RESULT="${EVIDENCE_DIR}/prepass/result.json"
  test -s "${BOUNDARY_INPUT}"
  test -s "${PREPASS_RESULT}"
  BOUNDARY_SIZE="$(stat -c %s -- "${BOUNDARY_INPUT}")"
  BOUNDARY_SHA="$(sha256sum -- "${BOUNDARY_INPUT}" | awk '{print $1}')"
  PREPASS_SIZE="$(stat -c %s -- "${PREPASS_RESULT}")"
  PREPASS_SHA="$(sha256sum -- "${PREPASS_RESULT}" | awk '{print $1}')"
  [[ "${BOUNDARY_SHA}" =~ ^[0-9a-f]{64}$ ]]
  [[ "${PREPASS_SHA}" =~ ^[0-9a-f]{64}$ ]]
  {
    printf 'boundary_input_size_bytes=%s\n' "${BOUNDARY_SIZE}"
    printf 'boundary_input_sha256=%s\n' "${BOUNDARY_SHA}"
    printf 'prepass_result_size_bytes=%s\n' "${PREPASS_SIZE}"
    printf 'prepass_result_sha256=%s\n' "${PREPASS_SHA}"
  } >"${EVIDENCE_DIR}/boundary_receipt.txt"

  printf 'direct_outer_hipprof ordinal=%s key=%s candidate=%s block=%s\n' \
    "${ORDINAL}" "${EVIDENCE_KEY}" "${CANDIDATE}" "${SEGMENT}" >"${RUN_ROOT}/RUN_STATUS"
  set +e
  run_created_container "${TRACE_CONTAINER}" "${EVIDENCE_DIR}" \
    --entrypoint /opt/dtk/bin/hipprof --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
    -e PATH=/work/runtime_tools:/work/tools:/opt/dtk/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
    -v "${TOOLS}:/work/tools:ro" -v "${BUNDLE_ROOT}:/work/bundle:ro" \
    -v "${RUN_ROOT}:/work/run:rw" -w /work \
    "${IMAGE}" --hip-trace --hsa-trace --hiptx-trace \
    -o "/work/run/profiles/${EVIDENCE_KEY}/hipprof.json" \
    /usr/bin/python3 /work/tools/profile_phase11_mixed_precision_block.py \
    --bundle /work/bundle --manifest "/work/bundle/${MANIFEST_RELATIVE}" \
    --candidate-id "${CANDIDATE}" --segment-index "${SEGMENT}" \
    --expected-precision "${PRECISION}" --expected-model-sha256 "${MODEL_SHA}" \
    --expected-cache-sha256 "${CACHE_SHA}" \
    --boundary-input "/work/run/profiles/${EVIDENCE_KEY}/prepass/boundary_input.npy" \
    --boundary-prepass-result "/work/run/profiles/${EVIDENCE_KEY}/prepass/result.json" \
    --expected-boundary-input-size "${BOUNDARY_SIZE}" \
    --expected-boundary-input-sha256 "${BOUNDARY_SHA}" \
    --expected-prepass-result-size "${PREPASS_SIZE}" \
    --expected-prepass-result-sha256 "${PREPASS_SHA}" \
    --runtime-fingerprint /work/run/runtime_fingerprint.json --container-image-id "${IMAGE_ID}" \
    --repetitions "${REPETITIONS}" --output-dir "/work/run/profiles/${EVIDENCE_KEY}/target"
  RC=$?
  set -e
  if [[ "${RC}" -ne 0 ]]; then
    printf 'profile_failed ordinal=%s key=%s candidate=%s block=%s rc=%s\n' \
      "${ORDINAL}" "${EVIDENCE_KEY}" "${CANDIDATE}" "${SEGMENT}" "${RC}" \
      >"${RUN_ROOT}/RUN_STATUS"
    exit "${RC}"
  fi
done <"${RUN_ROOT}/plan/profile_plan.tsv"

printf 'kernel_evidence_summary\n' >"${RUN_ROOT}/RUN_STATUS"
set +e
run_created_container phase11-mixed-kernel-summary-node2 "${RUN_ROOT}/summary_container" \
  --entrypoint /usr/bin/python3 \
  -e PATH=/work/runtime_tools:/work/tools:/opt/dtk/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${RUN_ROOT}:/work/run:rw" -w /work \
  "${IMAGE}" /work/tools/summarize_phase11_mixed_precision_kernel_evidence.py \
  --plan /work/run/plan/profile_plan.json --profiles-root /work/run/profiles \
  --runtime-fingerprint /work/run/runtime_fingerprint.json --container-image-id "${IMAGE_ID}" \
  --output-dir /work/run/summary
SUMMARY_RC=$?
set -e
if [[ "${SUMMARY_RC}" -ne 0 && "${SUMMARY_RC}" -ne 2 ]]; then
  printf 'summary_failed rc=%s\n' "${SUMMARY_RC}" >"${RUN_ROOT}/RUN_STATUS"
  exit "${SUMMARY_RC}"
fi

if [[ "${SUMMARY_RC}" -eq 0 && "${TASK_LIMIT}" -gt 0 ]]; then
  printf 'completed_selected_task_smoke_kernel_gates_passed_no_full_mapping_claim\n' \
    >"${RUN_ROOT}/RUN_STATUS"
elif [[ "${SUMMARY_RC}" -eq 0 ]]; then
  printf 'completed_all_required_kernel_gates_passed\n' >"${RUN_ROOT}/RUN_STATUS"
else
  printf 'completed_evidence_preserved_one_or_more_kernel_gates_failed\n' >"${RUN_ROOT}/RUN_STATUS"
fi
{
  date -Ins
  hostname
  /usr/local/hyhal/bin/hy-smi || true
  find "${RUN_ROOT}" -type f ! -path "${RUN_ROOT}/host_post.txt" -print0 \
    | sort -z | xargs -0 sha256sum
} >"${RUN_ROOT}/host_post.txt" 2>&1
cat "${RUN_ROOT}/summary/kernel_evidence_summary.json"
exit "${SUMMARY_RC}"
