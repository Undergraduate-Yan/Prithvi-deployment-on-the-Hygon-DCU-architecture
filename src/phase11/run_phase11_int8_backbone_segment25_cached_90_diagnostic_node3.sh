#!/usr/bin/env bash
set -euo pipefail

IMAGE='sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291'
PROJECT='/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100'
ROOT='/var/tmp/20260815-phase11-int8-backbone-node3'
TOOLS="${ROOT}/tools"
RUN="${ROOT}/segment25_cached_end_to_end_test90_v2"
CONTAINER='phase11-int8-backbone-segment25-cached-test90-node3'
EVALUATOR="${TOOLS}/evaluate_phase11_int8_backbone_segment25_cached_90_diagnostic.py"
BASE="${PROJECT}/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py"
FROZEN_INPUTS="${PROJECT}/phase11_frozen_test90/frozen_fp32_test90_inputs.npz"
FROZEN_MANIFEST="${PROJECT}/phase11_frozen_test90/manifest.json"
SINGLE="${ROOT}/segment25_cached_end_to_end_single/output/result.json"

exec 9>/var/tmp/phase11-int8-backbone-segment25-cached-test90-node3.lock
flock -n 9 || exit 75
test "$(hostname)" = machine3
test -z "$(docker ps -q)"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${TOOLS}/lsmod")" = 819664
test "$(sha256sum "${TOOLS}/lsmod" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test "$(stat -c %s "${EVALUATOR}")" = 19417
test "$(sha256sum "${EVALUATOR}" | awk '{print $1}')" = 18db1ddbd4d40ff2122f16243bdc8b4b620e5419f759be955f6bd14d3a2a7d01
test "$(stat -c %s "${BASE}")" = 16657
test "$(sha256sum "${BASE}" | awk '{print $1}')" = a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f
test "$(stat -c %s "${FROZEN_INPUTS}")" = 144553732
test "$(sha256sum "${FROZEN_INPUTS}" | awk '{print $1}')" = 6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86
test "$(stat -c %s "${FROZEN_MANIFEST}")" = 28349
test "$(sha256sum "${FROZEN_MANIFEST}" | awk '{print $1}')" = cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5
test "$(stat -c %s "${SINGLE}")" = 42574
test "$(sha256sum "${SINGLE}" | awk '{print $1}')" = 4e010e3e1e2e02b4c845f25b16681455e8f6da4cfca9b65b401f5f6555edaa4f
test "$(stat -c %s "${PROJECT}/evaluate_full_test_k100_onnx_int8.py")" = 59386
test "$(sha256sum "${PROJECT}/evaluate_full_test_k100_onnx_int8.py" | awk '{print $1}')" = cd746e45ac45e8da148e6481e318c3701473088c2bba2397bbf85e6917517435
test ! -e "${RUN}"

mkdir -p "${RUN}"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
{
  date -Ins
  hostname
  printf 'variant=int8_backbone_compat_25_static_batch1_cached_host_staged_test90\n'
  printf 'strict_numeric_admission_already_failed=true\nperformance_claim=false\n'
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${RUN}/host_pre.txt" 2>&1

(
  while true; do
    date -Ins
    docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' "${CONTAINER}" 2>/dev/null || true
    grep '^MemAvailable:' /proc/meminfo || true
    sleep 15
  done
) >"${RUN}/telemetry.log" 2>&1 &
TELEMETRY_PID=$!

set +e
docker run --rm --name "${CONTAINER}" --entrypoint python3 \
  --memory=16g --pids-limit=1024 \
  -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${ROOT}:/work/root:ro" \
  -v "${PROJECT}:/workspace:ro" -v "${RUN}:/work/run:rw" \
  -w /work "${IMAGE}" /work/root/tools/evaluate_phase11_int8_backbone_segment25_cached_90_diagnostic.py \
  --root /work/root \
  --single-result /work/root/segment25_cached_end_to_end_single/output/result.json \
  --base-evaluator /workspace/phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py \
  --helper /workspace/evaluate_full_test_k100_onnx_int8.py \
  --fp32-result /workspace/outputs/k100_full_test_fp32/20260807-171433/result.json \
  --fp32-csv /workspace/outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv \
  --fp32-tensors /workspace/outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt \
  --frozen-inputs /workspace/phase11_frozen_test90/frozen_fp32_test90_inputs.npz \
  --frozen-input-manifest /workspace/phase11_frozen_test90/manifest.json \
  --output-dir /work/run/output >"${RUN}/evaluation.log" 2>&1
RC=$?
set -e

kill "${TELEMETRY_PID}" 2>/dev/null || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
printf '%s\n' "${RC}" >"${RUN}/evaluation.exit"
{
  date -Ins
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${RUN}/host_post.txt" 2>&1
test ! -f "${RUN}/output/result.json" || cat "${RUN}/output/result.json"
exit "${RC}"
