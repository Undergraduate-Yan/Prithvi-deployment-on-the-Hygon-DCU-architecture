#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 M5_BUILD_ROOT SAMPLE PROJECT_ROOT NEW_ADMISSION_ROOT" >&2
  exit 64
fi

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
M5_ROOT="$(realpath -e -- "$1")"
SAMPLE="$(realpath -e -- "$2")"
PROJECT="$(realpath -e -- "$3")"
RUN_PARENT="$(realpath -m -- "$(dirname -- "$4")")"
RUN_NAME="$(basename -- "$4")"
RUN_ROOT="${RUN_PARENT}/${RUN_NAME}"
RUNTIME_TOOLS_HOST='/var/tmp/phase11_int8_backbone_segment25_evidence_bundle_20260817/tools'
LSMOD_SHIM="${RUNTIME_TOOLS_HOST}/lsmod"
case "${SAMPLE}" in
  *.npy) SAMPLE_SUFFIX='.npy' ;;
  *.npz) SAMPLE_SUFFIX='.npz' ;;
  *.pt) SAMPLE_SUFFIX='.pt' ;;
  *.pth) SAMPLE_SUFFIX='.pth' ;;
  *) echo "sample extension must be .npy, .npz, .pt, or .pth" >&2; exit 64 ;;
esac
SAMPLE_CONTAINER="/work/sample${SAMPLE_SUFFIX}"

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test -d /opt/hyhal
test "$(stat -c %s -- "${LSMOD_SHIM}")" = 819664
test "$(sha256sum -- "${LSMOD_SHIM}" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test -f "${M5_ROOT}/M5/mixed_precision_manifest.final.json"
test ! -e "${RUN_ROOT}"
mkdir -p -- "${RUN_PARENT}"

exec 9>/var/tmp/phase11-mixed-precision-formal-performance-node2.lock
flock -n 9 || exit 75
test -z "$(docker ps -q)"

COMMON="${TOOLS}/phase11_mixed_precision_common.py"
SINGLE="${TOOLS}/evaluate_phase11_mixed_precision_single.py"
TEST90="${TOOLS}/evaluate_phase11_mixed_precision_test90.py"
AGGREGATOR="${TOOLS}/aggregate_phase11_mixed_precision_test90.py"

lock_file() {
  local path="$1" expected_size="$2" expected_sha="$3"
  test "$(stat -c %s -- "${path}")" = "${expected_size}"
  test "$(sha256sum -- "${path}" | awk '{print $1}')" = "${expected_sha}"
}

lock_file "${COMMON}" 23030 f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8
lock_file "${SINGLE}" 8872 e69b2870303a92d8d2bb91dcfe950256ad66268c82beb7a11a34f08d03119db0
lock_file "${TEST90}" 17348 8bc54ffe04719cce53d22648b0ffc2fd1e8daaac8e2e4acb61c12910c61d4091
lock_file "${AGGREGATOR}" 9628 717aa6c5e10b889c082acb4626105ad708c9006acf70380793f0b93c71fbd159

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
test "${IMAGE_ID}" = "${IMAGE}"
mkdir -- "${RUN_ROOT}"
docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'image_id=%s\n' "${IMAGE_ID}"
  printf 'm5_root=%s\n' "${M5_ROOT}"
  printf 'sample=%s\n' "${SAMPLE}"
  printf 'sample_container_path=%s\n' "${SAMPLE_CONTAINER}"
  printf 'project=%s\n' "${PROJECT}"
  printf 'static_lsmod_shim=%s\n' "${LSMOD_SHIM}"
  printf 'static_lsmod_sha256=9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12\n'
  printf 'single_process_count=1\n'
  printf 'fixed90_fresh_process_count=3\n'
} >"${RUN_ROOT}/host_protocol.txt"

ACTIVE_CONTAINER=''
cleanup() {
  if [[ -n "${ACTIVE_CONTAINER}" ]]; then
    docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

run_container() {
  local name="$1" log="$2"
  shift 2
  ACTIVE_CONTAINER="${name}"
  set +e
  docker run --rm --name "${name}" "$@" >"${log}" 2>&1
  local rc=$?
  set -e
  ACTIVE_CONTAINER=''
  if [[ "${rc}" -ne 0 ]]; then
    sed -n '1,240p' "${log}" >&2 || true
    return "${rc}"
  fi
}

printf 'single_sample\n' >"${RUN_ROOT}/RUN_STATUS"
mkdir -p -- "${RUN_ROOT}/M5/single"
run_container phase11-m5-single-node2 "${RUN_ROOT}/M5/single/container.log" \
  --entrypoint python3 --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
  -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" \
  -v "${M5_ROOT}:/work/m5:ro" -v "${SAMPLE}:${SAMPLE_CONTAINER}:ro" \
  -v "${RUN_ROOT}:/work/run:rw" -w /work \
  "${IMAGE}" /work/tools/evaluate_phase11_mixed_precision_single.py \
  --bundle /work/m5 --manifest /work/m5/M5/mixed_precision_manifest.final.json \
  --sample "${SAMPLE_CONTAINER}" --output-dir /work/run/M5/single/output

test -f "${RUN_ROOT}/M5/single/output/result.json"
printf 'fixed90_three_fresh_processes\n' >"${RUN_ROOT}/RUN_STATUS"
for TRIAL in 1 2 3; do
  TRIAL_DIR="${RUN_ROOT}/M5/run_${TRIAL}"
  mkdir -p -- "${TRIAL_DIR}"
  run_container "phase11-m5-test90-r${TRIAL}-node2" "${TRIAL_DIR}/container.log" \
    --entrypoint python3 --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
    -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
    -v "${TOOLS}:/work/tools:ro" \
    -v "${M5_ROOT}:/work/m5:ro" -v "${PROJECT}:/work/project:ro" \
    -v "${RUN_ROOT}:/work/run:rw" -w /work \
    "${IMAGE}" /work/tools/evaluate_phase11_mixed_precision_test90.py \
    --bundle /work/m5 --manifest /work/m5/M5/mixed_precision_manifest.final.json \
    --single-result /work/run/M5/single/output/result.json --project /work/project \
    --output-dir "/work/run/M5/run_${TRIAL}/output"
done

printf 'fixed90_aggregate\n' >"${RUN_ROOT}/RUN_STATUS"
mkdir -- "${RUN_ROOT}/M5/aggregate_container"
run_container phase11-m5-test90-aggregate-node2 "${RUN_ROOT}/M5/aggregate_container/container.log" \
  --entrypoint python3 --memory=8g --pids-limit=256 \
  -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${M5_ROOT}:/work/m5:ro" \
  -v "${RUN_ROOT}:/work/run:rw" -w /work \
  "${IMAGE}" /work/tools/aggregate_phase11_mixed_precision_test90.py \
  --bundle /work/m5 --manifest /work/m5/M5/mixed_precision_manifest.final.json \
  --run-dir /work/run/M5/run_1/output \
  --run-dir /work/run/M5/run_2/output \
  --run-dir /work/run/M5/run_3/output \
  --output-dir /work/run/M5/three_run_summary

test -f "${RUN_ROOT}/M5/three_run_summary/summary.json"
printf 'completed_single_and_three_run_test90\n' >"${RUN_ROOT}/RUN_STATUS"
{
  date -Ins
  hostname
  /usr/local/hyhal/bin/hy-smi || true
  find "${RUN_ROOT}" -type f -print0 | sort -z | xargs -0 sha256sum
} >"${RUN_ROOT}/host_post.txt" 2>&1
cat "${RUN_ROOT}/M5/three_run_summary/summary.json"
