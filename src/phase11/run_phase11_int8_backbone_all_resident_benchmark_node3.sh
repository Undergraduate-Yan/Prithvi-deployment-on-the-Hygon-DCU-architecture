#!/usr/bin/env bash
set -euo pipefail

IMAGE='sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291'
ROOT='/var/tmp/20260815-phase11-int8-backbone-node3'
TOOLS="${ROOT}/tools"
TRIAL_SCRIPT="${TOOLS}/benchmark_phase11_int8_backbone_all_resident_trial.py"
AGGREGATOR="${TOOLS}/aggregate_phase11_int8_backbone_all_resident_benchmark.py"
RUN="${ROOT}/segment25_all_resident_absolute_benchmark_v2"

exec 9>/var/tmp/phase11-int8-backbone-all-resident-benchmark-node3.lock
flock -n 9 || exit 75
test "$(hostname)" = machine3
test -z "$(docker ps -q)"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${TOOLS}/lsmod")" = 819664
test "$(sha256sum "${TOOLS}/lsmod" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test "$(stat -c %s "${TRIAL_SCRIPT}")" = 14454
test "$(sha256sum "${TRIAL_SCRIPT}" | awk '{print $1}')" = dd2a617f6cd1122c293bd4dc0f9cb8fe696fd174de44ed6ce37579acd561c10e
test "$(stat -c %s "${AGGREGATOR}")" = 6461
test "$(sha256sum "${AGGREGATOR}" | awk '{print $1}')" = 4a61150e8b2bf044bd4f6f74294c9ac0fa05c2272b8f1223ef518290cac3af23
test ! -e "${RUN}"

mkdir -p "${RUN}"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'batch=1\nwarmup=30\nmeasured=100\nfresh_process_trials=3\n'
  printf 'scope=25 resident sessions, fixed device OrtValues, excludes H2D/D2H and session load\n'
  printf 'same_protocol_fp32_baseline=false\nint8_speedup_claim=false\n'
} >"${RUN}/protocol.txt"

for TRIAL in 1 2 3; do
  TRIAL_ROOT="${RUN}/trial_${TRIAL}"
  CONTAINER="phase11-int8-backbone-absolute-trial-${TRIAL}-node3"
  mkdir -p "${TRIAL_ROOT}"
  {
    date -Ins
    free -h
    /usr/local/hyhal/bin/hy-smi || true
  } >"${TRIAL_ROOT}/host_pre.txt" 2>&1
  (
    while true; do
      date -Ins
      docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' "${CONTAINER}" 2>/dev/null || true
      /usr/local/hyhal/bin/hy-smi --showmeminfo vram 2>/dev/null || true
      sleep 3
    done
  ) >"${TRIAL_ROOT}/telemetry.log" 2>&1 &
  TELEMETRY_PID=$!

  set +e
  docker run --rm --name "${CONTAINER}" --entrypoint python3 \
    --memory=54g --pids-limit=1024 \
    -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${ROOT}:/work/root:ro" -v "${TRIAL_ROOT}:/work/trial:rw" \
    -w /work "${IMAGE}" /work/root/tools/benchmark_phase11_int8_backbone_all_resident_trial.py \
    --root /work/root --trial-index "${TRIAL}" --output-dir /work/trial/output \
    >"${TRIAL_ROOT}/benchmark.log" 2>&1
  RC=$?
  set -e
  kill "${TELEMETRY_PID}" 2>/dev/null || true
  wait "${TELEMETRY_PID}" 2>/dev/null || true
  printf '%s\n' "${RC}" >"${TRIAL_ROOT}/benchmark.exit"
  {
    date -Ins
    free -h
    /usr/local/hyhal/bin/hy-smi || true
  } >"${TRIAL_ROOT}/host_post.txt" 2>&1
  if [[ "${RC}" -ne 0 ]]; then
    test ! -f "${TRIAL_ROOT}/output/result.json" || cat "${TRIAL_ROOT}/output/result.json"
    exit "${RC}"
  fi
done

set +e
docker run --rm --entrypoint python3 \
  -v "${TOOLS}:/work/tools:ro" -v "${RUN}:/work/run:rw" -w /work "${IMAGE}" \
  /work/tools/aggregate_phase11_int8_backbone_all_resident_benchmark.py \
  --run-root /work/run >"${RUN}/aggregate.log" 2>&1
RC=$?
set -e
printf '%s\n' "${RC}" >"${RUN}/aggregate.exit"
test ! -f "${RUN}/summary.json" || cat "${RUN}/summary.json"
exit "${RC}"
