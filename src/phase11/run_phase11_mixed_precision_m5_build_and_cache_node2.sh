#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 BASE_M0_M4_BUILD_ROOT FP16_SOURCE FP16_SOURCE_REPORT INT8_QUANTIZATION_REPORT NEW_M5_ROOT" >&2
  exit 64
fi

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASE_ROOT="$(realpath -e -- "$1")"
FP16_SOURCE="$(realpath -e -- "$2")"
FP16_REPORT="$(realpath -e -- "$3")"
INT8_REPORT="$(realpath -e -- "$4")"
OUTPUT_PARENT="$(realpath -m -- "$(dirname -- "$5")")"
OUTPUT_NAME="$(basename -- "$5")"
OUTPUT_ROOT="${OUTPUT_PARENT}/${OUTPUT_NAME}"
RUNTIME_TOOLS_HOST='/var/tmp/phase11_int8_backbone_segment25_evidence_bundle_20260817/tools'
LSMOD_SHIM="${RUNTIME_TOOLS_HOST}/lsmod"

[[ "${OUTPUT_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test -d /opt/hyhal
test "$(stat -c %s -- "${LSMOD_SHIM}")" = 819664
test "$(sha256sum -- "${LSMOD_SHIM}" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test -f "${BASE_ROOT}/M4/mixed_precision_manifest.final.json"
test ! -e "${OUTPUT_ROOT}"
mkdir -p -- "${OUTPUT_PARENT}"

# Share the exact formal-performance lock.  If the M0--M4 run is active this
# launcher exits 75 before starting even the CPU-only construction container.
exec 9>/var/tmp/phase11-mixed-precision-formal-performance-node2.lock
flock -n 9 || exit 75
test -z "$(docker ps -q)"

BASE_BUILDER="${TOOLS}/build_phase11_sensitivity_mixed_precision_m0_m4.py"
COMMON="${TOOLS}/phase11_mixed_precision_common.py"
BASE_COMPILER="${TOOLS}/compile_phase11_mixed_fp16_segment_caches.py"
M5_BUILDER="${TOOLS}/build_phase11_sensitivity_mixed_precision_m5.py"
M5_COMPILER="${TOOLS}/compile_phase11_mixed_m5_incremental_fp16_caches.py"
M5_FINALIZER="${TOOLS}/finalize_phase11_mixed_precision_m5_cache_manifest.py"

lock_file() {
  local path="$1" expected_size="$2" expected_sha="$3"
  test "$(stat -c %s -- "${path}")" = "${expected_size}"
  test "$(sha256sum -- "${path}" | awk '{print $1}')" = "${expected_sha}"
}

lock_file "${BASE_BUILDER}" 43770 d2555c1bdd5c6c975218e5c371e1ae5de92075fe3e734e21c7ff936ed9ae81b1
lock_file "${COMMON}" 23030 f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8
lock_file "${BASE_COMPILER}" 14539 f58ad733e3821dcee1591cc9c6188b0a02917b0e55ae8889ec8e102c24d14ee9
lock_file "${M5_BUILDER}" 22085 59316ce267454174ac91a290cc99fcbef979a01c53218789d43fbad3bfddfeea
lock_file "${M5_COMPILER}" 8127 637ee955f241615e78ede433e5886f26f6b06392a8fcae1658ebaff8284159b6
lock_file "${M5_FINALIZER}" 14258 2e79a196d03fc1785c21d949486db35b628e6690c1a5a3690b3ea5511cb30b74

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
test "${IMAGE_ID}" = "${IMAGE}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

ACTIVE_CONTAINER=''
BUILD_LOG="${OUTPUT_PARENT}/.${OUTPUT_NAME}.build.$$.log"
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

run_container phase11-m5-static-build-node2 "${BUILD_LOG}" \
  --entrypoint python3 --memory=54g --pids-limit=1024 \
  -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${BASE_ROOT}:/work/base:ro" \
  -v "${FP16_SOURCE}:/work/input/fp16_full.onnx:ro" \
  -v "${FP16_REPORT}:/work/input/fp16_report.json:ro" \
  -v "${INT8_REPORT}:/work/input/int8_report.json:ro" \
  -v "${OUTPUT_PARENT}:/work/output_parent:rw" -w /work \
  "${IMAGE}" /work/tools/build_phase11_sensitivity_mixed_precision_m5.py \
  --base-build-root /work/base \
  --fp16-source /work/input/fp16_full.onnx \
  --fp16-source-report /work/input/fp16_report.json \
  --int8-quantization-report /work/input/int8_report.json \
  --output-root "/work/output_parent/${OUTPUT_NAME}" --payload-mode copy

test -d "${OUTPUT_ROOT}"
# The builder runs as container root and atomically publishes the new tree.
# Return ownership to the invoking experiment user before the host launcher
# creates audit files; later GPU containers can still write through the mount.
run_container phase11-m5-normalize-output-owner-node2 "${OUTPUT_PARENT}/.${OUTPUT_NAME}.chown.$$.log" \
  --entrypoint chown -v "${OUTPUT_ROOT}:/work/m5:rw" \
  "${IMAGE}" -R "${HOST_UID}:${HOST_GID}" /work/m5
test "$(stat -c %u -- "${OUTPUT_ROOT}")" = "${HOST_UID}"
test "$(stat -c %g -- "${OUTPUT_ROOT}")" = "${HOST_GID}"
mkdir -- "${OUTPUT_ROOT}/launcher_evidence"
mv -- "${BUILD_LOG}" "${OUTPUT_ROOT}/launcher_evidence/build.log"
mv -- "${OUTPUT_PARENT}/.${OUTPUT_NAME}.chown.$$.log" \
  "${OUTPUT_ROOT}/launcher_evidence/normalize_owner.log"
docker image inspect "${IMAGE}" >"${OUTPUT_ROOT}/launcher_evidence/image_inspect.json"
{
  date -Ins
  hostname
  printf 'image_id=%s\n' "${IMAGE_ID}"
  printf 'base_root=%s\n' "${BASE_ROOT}"
  printf 'fp16_source=%s\n' "${FP16_SOURCE}"
  printf 'fp16_source_sha256=%s\n' "$(sha256sum -- "${FP16_SOURCE}" | awk '{print $1}')"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'reused_fp16_blocks=0,14,17,18\n'
  printf 'new_fp16_blocks=15,16,19,20,21\n'
  printf 'static_lsmod_shim=%s\n' "${LSMOD_SHIM}"
  printf 'static_lsmod_sha256=9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12\n'
  printf 'm0_m4_formal_performance_lock_shared=true\n'
} >"${OUTPUT_ROOT}/launcher_evidence/host_protocol.txt"
printf 'cache_compile\n' >"${OUTPUT_ROOT}/RUN_STATUS"

run_container phase11-m5-cache-compile-node2 "${OUTPUT_ROOT}/launcher_evidence/cache_compile.log" \
  --entrypoint python3 --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 \
  -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" \
  -v "${OUTPUT_ROOT}:/work/m5:rw" -w /work \
  "${IMAGE}" /work/tools/compile_phase11_mixed_m5_incremental_fp16_caches.py \
  --m5-manifest /work/m5/M5/mixed_precision_manifest.json \
  --cache-root /work/m5/_shared/m5_new_fp16_caches \
  --output-dir /work/m5/_shared/m5_new_fp16_cache_evidence --device-id 0

printf 'manifest_finalize\n' >"${OUTPUT_ROOT}/RUN_STATUS"
run_container phase11-m5-cache-finalize-node2 "${OUTPUT_ROOT}/launcher_evidence/cache_finalize.log" \
  --entrypoint python3 --memory=8g --pids-limit=256 \
  -e PATH=/work/runtime_tools:/work/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v "${RUNTIME_TOOLS_HOST}:/work/runtime_tools:ro" \
  -v "${TOOLS}:/work/tools:ro" -v "${OUTPUT_ROOT}:/work/m5:rw" -w /work \
  "${IMAGE}" /work/tools/finalize_phase11_mixed_precision_m5_cache_manifest.py \
  --build-root /work/m5 \
  --cache-compile-result /work/m5/_shared/m5_new_fp16_cache_evidence/cache_compile_result.json

test -f "${OUTPUT_ROOT}/M5/mixed_precision_manifest.final.json"
printf 'cache_finalized_static_pass\n' >"${OUTPUT_ROOT}/RUN_STATUS"
{
  date -Ins
  hostname
  /usr/local/hyhal/bin/hy-smi || true
  find "${OUTPUT_ROOT}" -type f -print0 | sort -z | xargs -0 sha256sum
} >"${OUTPUT_ROOT}/launcher_evidence/host_post.txt" 2>&1
cat "${OUTPUT_ROOT}/cache_finalize_summary.json"
