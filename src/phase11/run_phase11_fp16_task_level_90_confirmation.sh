#!/usr/bin/env bash
set -euo pipefail
IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
PROJECT='/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100'
SINGLE_ROOT='/var/tmp/20260814-phase11-fp16-compat-migraphx-single-cache-save'
DIAG_ROOT='/var/tmp/20260814-phase11-fp16-compat-cached-90-diagnostic'
CANDIDATE='/var/tmp/20260814-phase11-fp16-compat-build/output/candidate.onnx'
CACHE="${SINGLE_ROOT}/cache/prithvi_fp16_compat.mxr"
SINGLE="${SINGLE_ROOT}/output/result.json"
FROZEN_RESULT="${DIAG_ROOT}/output/result.json"
FROZEN_CSV="${DIAG_ROOT}/output/per_sample_metrics.csv"
WRAPPER='/var/tmp/confirm_phase11_fp16_task_level_90.py'
EVALUATOR='/var/tmp/evaluate_phase11_fp16_compat_cached_90_diagnostic.py'
BASE='/var/tmp/evaluate_phase11_migraphx_barrier_fp32_90.py'
INPUTS='/var/tmp/20260814-phase11-frozen-input-pack-rerun/frozen_fp32_test90_inputs.npz'
MANIFEST='/var/tmp/20260814-phase11-frozen-input-pack-rerun/manifest.json'
RUN='/var/tmp/20260814-phase11-fp16-task-level-90-confirmation'

exec 9>/var/tmp/phase11-fp16-task-level-confirmation.lock
flock -n 9 || exit 75
test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${WRAPPER}")" = 7558
test "$(sha256sum "${WRAPPER}"|awk '{print $1}')" = c70d5a3c0b9ad9556fd40cefc0397535f4b34ecbea1375fb69b8a2ae7655b5fa
test "$(stat -c %s "${EVALUATOR}")" = 11488
test "$(sha256sum "${EVALUATOR}"|awk '{print $1}')" = 5817db57933914e1d560bfb614d20d047c5a3301ff3482e0c8b00f6f3a6b83ef
test "$(stat -c %s "${FROZEN_RESULT}")" = 5993
test "$(sha256sum "${FROZEN_RESULT}"|awk '{print $1}')" = 48de1f62acf10116254574851d5974528c4ce3df7363504b2fe59424de3a72dc
test "$(stat -c %s "${FROZEN_CSV}")" = 25666
test "$(sha256sum "${FROZEN_CSV}"|awk '{print $1}')" = 69ffbf22eb2b7380a3675ebe5650fbde12d1af951ecf3e168fe22126666cd530
test ! -e "${RUN}"
mkdir -p "${RUN}/output"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
{ date -Ins; hostname; hy-smi --showmeminfo vram || true; } >"${RUN}/host_pre.txt" 2>&1
set +e
docker run --rm --name phase11-fp16-task-level-confirmation \
 --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=16g \
 -v /opt/hyhal:/opt/hyhal:ro -v "${PROJECT}:/workspace:ro" \
 -v "${CANDIDATE}:/work/candidate.onnx:ro" -v "${CACHE}:/work/cache.mxr:ro" \
 -v "${SINGLE}:/work/single.json:ro" -v "${FROZEN_RESULT}:/work/frozen_result.json:ro" \
 -v "${FROZEN_CSV}:/work/frozen.csv:ro" -v "${WRAPPER}:/work/confirm.py:ro" \
 -v "${EVALUATOR}:/work/evaluator.py:ro" -v "${BASE}:/work/base.py:ro" \
 -v "${INPUTS}:/work/inputs.npz:ro" -v "${MANIFEST}:/work/manifest.json:ro" \
 -v "${RUN}/output:/work/output:rw" -w /work "${IMAGE}" python3 /work/confirm.py \
 --evaluator /work/evaluator.py --frozen-result /work/frozen_result.json --frozen-csv /work/frozen.csv \
 --candidate /work/candidate.onnx --cache /work/cache.mxr --single-result /work/single.json \
 --base-evaluator /work/base.py --helper /workspace/evaluate_full_test_k100_onnx_int8.py \
 --fp32-result /workspace/outputs/k100_full_test_fp32/20260807-171433/result.json \
 --fp32-csv /workspace/outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv \
 --fp32-tensors /workspace/outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt \
 --frozen-inputs /work/inputs.npz --frozen-input-manifest /work/manifest.json --output-dir /work/output \
 2>&1 | tee "${RUN}/confirmation.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "${RC}" >"${RUN}/confirmation.exit"
{ date -Ins; hy-smi --showmeminfo vram || true; } >"${RUN}/host_post.txt" 2>&1
exit "${RC}"
