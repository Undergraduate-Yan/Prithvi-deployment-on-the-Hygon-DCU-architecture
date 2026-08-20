#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 M5_BUNDLE FP16_FULL_BUNDLE NEW_OUTPUT_DIR" >&2
  exit 64
fi

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
M5_MANIFEST_SHA='f4dc433595c43332d4e29c2351363b3245f214273f058551788d50be29c388da'
FP16_MANIFEST_SHA='1f433a917aed5fead6dd95597a5a03f500a93ce564067e9dad4d25f4682bad25'
INFER_SHA='fc338d90f7592f6a75cfa28191702598777b21914415e7178cef932de60571b7'
LSMOD_SHA='9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12'
TRIAL_SIZE=18146
TRIAL_SHA='da7ae2c77e4b469ba0674751e996a5dfcaf0226a638c90b364e3a3503e821560'
AGGREGATOR_SIZE=26628
AGGREGATOR_SHA='5904f9c7f2f4a93c0613746a5c780021b02a486155f068921bf42e23c8dbaf5c'
DOCKER_PATH='/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin'

TOOLS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TRIAL_TOOL="${TOOLS}/benchmark_bundle_cli_trial.py"
AGGREGATOR="${TOOLS}/aggregate_bundle_cli_pair.py"
M5_BUNDLE="$(realpath -e -- "$1")"
FP16_BUNDLE="$(realpath -e -- "$2")"
mkdir -p -- "$(dirname -- "$3")"
RUN_ROOT="$(realpath -m -- "$3")"

contained_by() {
  local child="$1" parent="$2"
  [[ "${child}/" == "${parent}/"* ]]
}

if [[ "${M5_BUNDLE}" == "${FP16_BUNDLE}" ]] || \
   contained_by "${RUN_ROOT}" "${M5_BUNDLE}" || \
   contained_by "${RUN_ROOT}" "${FP16_BUNDLE}" || \
   contained_by "${M5_BUNDLE}" "${RUN_ROOT}" || \
   contained_by "${FP16_BUNDLE}" "${RUN_ROOT}"; then
  echo 'bundle/output roots must be distinct and non-nested' >&2
  exit 64
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite run root: ${RUN_ROOT}" >&2
  exit 73
fi

lock_file() {
  local path="$1" expected_size="$2" expected_sha="$3"
  [[ -f "${path}" && ! -L "${path}" ]]
  [[ "$(stat -c %s -- "${path}")" == "${expected_size}" ]]
  [[ "$(sha256sum -- "${path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

lock_sha() {
  local path="$1" expected_sha="$2"
  [[ -f "${path}" && ! -L "${path}" ]]
  [[ "$(sha256sum -- "${path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

lock_file "${TRIAL_TOOL}" "${TRIAL_SIZE}" "${TRIAL_SHA}"
lock_file "${AGGREGATOR}" "${AGGREGATOR_SIZE}" "${AGGREGATOR_SHA}"
lock_sha "${M5_BUNDLE}/manifest.json" "${M5_MANIFEST_SHA}"
lock_sha "${FP16_BUNDLE}/manifest.json" "${FP16_MANIFEST_SHA}"
lock_sha "${M5_BUNDLE}/tools/infer_k100.py" "${INFER_SHA}"
lock_sha "${FP16_BUNDLE}/tools/infer_k100.py" "${INFER_SHA}"
lock_file "${M5_BUNDLE}/tools/bin/lsmod" 819664 "${LSMOD_SHA}"
lock_file "${FP16_BUNDLE}/tools/bin/lsmod" 819664 "${LSMOD_SHA}"
[[ -x "${M5_BUNDLE}/tools/bin/lsmod" && -x "${FP16_BUNDLE}/tools/bin/lsmod" ]]

[[ "$(hostname)" == 'machine2' ]]
[[ -c /dev/kfd && -d /dev/dri && -d /opt/hyhal ]]
[[ -r /sys/class/drm/card1/device/gpu_busy_percent ]]
[[ -z "$(docker ps -q)" ]]
IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
[[ "${IMAGE_ID}" == "${IMAGE}" ]]

exec 9>/var/tmp/phase11-k100-2-deployment-cli-pair.lock
flock -n 9 || exit 75

ACTIVE_CONTAINER=''
cleanup() {
  if [[ -n "${ACTIVE_CONTAINER}" ]]; then
    docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_idle_k100() {
  [[ -z "$(docker ps -q)" ]]
  local sample
  for _ in 1 2 3 4 5; do
    sample="$(tr -d '\r\n ' </sys/class/drm/card1/device/gpu_busy_percent)"
    [[ "${sample}" == '0' ]]
    sleep 0.2
  done
}

mkdir -- "${RUN_ROOT}"
docker image inspect "${IMAGE}" >"${RUN_ROOT}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'image_id=%s\n' "${IMAGE_ID}"
  printf 'm5_bundle=%s\n' "${M5_BUNDLE}"
  printf 'fp16_bundle=%s\n' "${FP16_BUNDLE}"
  printf 'm5_manifest_sha256=%s\n' "${M5_MANIFEST_SHA}"
  printf 'fp16_manifest_sha256=%s\n' "${FP16_MANIFEST_SHA}"
  printf 'trial_tool_sha256=%s\n' "${TRIAL_SHA}"
  printf 'aggregator_sha256=%s\n' "${AGGREGATOR_SHA}"
  printf 'docker_level_path=%s\n' "${DOCKER_PATH}"
  printf 'protocol=each candidate 3 fresh containers; 30 warmups + 100 measurements\n'
  printf 'order=trial1:m5,fp16;trial2:fp16,m5;trial3:m5,fp16\n'
} >"${RUN_ROOT}/host_protocol.txt"
printf 'timed_trials\n' >"${RUN_ROOT}/RUN_STATUS"

run_candidate() {
  local trial_index="$1" position="$2" variant="$3" sequence="$4" bundle="$5"
  local evidence="${RUN_ROOT}/trial_$(printf '%02d' "${trial_index}")/position_$(printf '%02d' "${position}")_${variant}"
  mkdir -p -- "${evidence}"
  wait_for_idle_k100

  local name="phase11-cli-pair-t${trial_index}-p${position}-${variant}-node2-${BASHPID}"
  local cid wait_command_rc container_rc
  cid="$(docker create \
    --name "${name}" \
    --entrypoint /usr/bin/python3 \
    --ulimit stack=-1:-1 \
    --memory=54g --pids-limit=2048 \
    --env "PATH=${DOCKER_PATH}" \
    --env "PHASE11_K100_IMAGE_ID=${IMAGE_ID}" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size=16g \
    --volume /opt/hyhal:/opt/hyhal:ro \
    --volume "${bundle}:/bundle:ro" \
    --volume "${TOOLS}:/tools:ro" \
    --volume "${evidence}:/run:rw" \
    "${IMAGE}" \
    /tools/benchmark_bundle_cli_trial.py \
    --bundle /bundle \
    --variant "${variant}" \
    --trial-index "${trial_index}" \
    --candidate-position "${position}" \
    --sequence "${sequence}" \
    --device 0 \
    --output /run/result.json)"
  ACTIVE_CONTAINER="${cid}"
  printf '%s\n' "${cid}" >"${evidence}/container_id.txt"
  docker start "${cid}" >"${evidence}/container_start.stdout" 2>"${evidence}/container_start.stderr"
  set +e
  container_rc="$(docker wait "${cid}")"
  wait_command_rc=$?
  set -e
  docker logs "${cid}" >"${evidence}/container.log" 2>&1 || true
  docker inspect "${cid}" >"${evidence}/container_inspect_exited.json"
  printf '%s\n' "${container_rc}" >"${evidence}/container.exit"
  docker rm "${cid}" >/dev/null
  ACTIVE_CONTAINER=''
  if [[ "${wait_command_rc}" -ne 0 || "${container_rc}" -ne 0 || ! -f "${evidence}/result.json" ]]; then
    printf 'trial_failed trial=%s position=%s variant=%s wait_rc=%s container_rc=%s\n' \
      "${trial_index}" "${position}" "${variant}" "${wait_command_rc}" "${container_rc}" \
      >"${RUN_ROOT}/RUN_STATUS"
    return 1
  fi
}

for TRIAL in 1 2 3; do
  if (( TRIAL % 2 == 1 )); then
    SEQUENCE='m5,fp16'
    VARIANTS=(m5 fp16)
  else
    SEQUENCE='fp16,m5'
    VARIANTS=(fp16 m5)
  fi
  for POSITION_ZERO in 0 1; do
    POSITION=$((POSITION_ZERO + 1))
    VARIANT="${VARIANTS[POSITION_ZERO]}"
    if [[ "${VARIANT}" == 'm5' ]]; then
      BUNDLE="${M5_BUNDLE}"
    else
      BUNDLE="${FP16_BUNDLE}"
    fi
    run_candidate "${TRIAL}" "${POSITION}" "${VARIANT}" "${SEQUENCE}" "${BUNDLE}"
  done
done

printf 'aggregate\n' >"${RUN_ROOT}/RUN_STATUS"
/usr/bin/python3 "${AGGREGATOR}" \
  --run-root "${RUN_ROOT}" \
  --expected-trial-script-sha256 "${TRIAL_SHA}" \
  --output "${RUN_ROOT}/summary.json" \
  >"${RUN_ROOT}/aggregate.stdout" 2>"${RUN_ROOT}/aggregate.stderr"
[[ -s "${RUN_ROOT}/summary.json" ]]
printf 'completed\n' >"${RUN_ROOT}/RUN_STATUS"
{
  date -Ins
  hostname
  printf 'image_id=%s\n' "${IMAGE_ID}"
  find "${RUN_ROOT}" -type f ! -name host_post.txt -print0 | sort -z | xargs -0 sha256sum
} >"${RUN_ROOT}/host_post.txt" 2>&1
cat "${RUN_ROOT}/summary.json"
