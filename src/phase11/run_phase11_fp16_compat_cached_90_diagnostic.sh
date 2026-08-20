#!/usr/bin/env bash
set -euo pipefail
IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
PROJECT='/mnt/jfs/prithvi_k100_ycg/work/prithvi_k100'
ROOT='/var/tmp/20260814-phase11-fp16-compat-migraphx-single-cache-save'
CANDIDATE='/var/tmp/20260814-phase11-fp16-compat-build/output/candidate.onnx'
CACHE="${ROOT}/cache/prithvi_fp16_compat.mxr"
SINGLE="${ROOT}/output/result.json"
EVALUATOR='/var/tmp/evaluate_phase11_fp16_compat_cached_90_diagnostic.py'
BASE='/var/tmp/evaluate_phase11_migraphx_barrier_fp32_90.py'
FROZEN_INPUTS='/var/tmp/20260814-phase11-frozen-input-pack-rerun/frozen_fp32_test90_inputs.npz'
FROZEN_MANIFEST='/var/tmp/20260814-phase11-frozen-input-pack-rerun/manifest.json'
RUN='/var/tmp/20260814-phase11-fp16-compat-cached-90-diagnostic'
OUT="${RUN}/output"

exec 9>/var/tmp/phase11-fp16-compat-cached-90-diagnostic.lock
flock -n 9 || exit 75
test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${CANDIDATE}")" = 638970735
test "$(sha256sum "${CANDIDATE}"|awk '{print $1}')" = 8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7
test "$(stat -c %s "${CACHE}")" = 708193607
test "$(sha256sum "${CACHE}"|awk '{print $1}')" = 9fe8985dc8ce4a3cf9d829ad66a91de41bb67b22e01b8181c43bf15c74f7dd0a
test "$(stat -c %s "${SINGLE}")" = 3316
test "$(sha256sum "${SINGLE}"|awk '{print $1}')" = 5e93e828ad44ca703c1eac5b08fd2c332d70002cec13c6d3c3c1b688f5a35a89
test "$(stat -c %s "${EVALUATOR}")" = 11488
test "$(sha256sum "${EVALUATOR}"|awk '{print $1}')" = 5817db57933914e1d560bfb614d20d047c5a3301ff3482e0c8b00f6f3a6b83ef
test "$(stat -c %s "${BASE}")" = 16657
test "$(sha256sum "${BASE}"|awk '{print $1}')" = a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f
test "$(stat -c %s "${FROZEN_INPUTS}")" = 144553732
test "$(sha256sum "${FROZEN_INPUTS}"|awk '{print $1}')" = 6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86
test "$(stat -c %s "${FROZEN_MANIFEST}")" = 28349
test "$(sha256sum "${FROZEN_MANIFEST}"|awk '{print $1}')" = cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5
test ! -e "${RUN}"
mkdir -p "${OUT}"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
{ date -Ins; hostname; hy-smi --showmeminfo vram || true; } >"${RUN}/host_pre.txt" 2>&1

set +e
docker run --rm --name phase11-fp16-compat-cached-90 \
 --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=16g \
 -v /opt/hyhal:/opt/hyhal:ro -v "${PROJECT}:/workspace:ro" \
 -v "${CANDIDATE}:/work/candidate.onnx:ro" -v "${CACHE}:/work/cache.mxr:ro" \
 -v "${SINGLE}:/work/single.json:ro" -v "${EVALUATOR}:/work/evaluate.py:ro" \
 -v "${BASE}:/work/base.py:ro" -v "${FROZEN_INPUTS}:/work/frozen.npz:ro" \
 -v "${FROZEN_MANIFEST}:/work/manifest.json:ro" -v "${OUT}:/work/output:rw" \
 -e PRITHVI_IMAGE_ID="${IMAGE}" -w /work "${IMAGE}" python3 /work/evaluate.py \
 --candidate /work/candidate.onnx --cache /work/cache.mxr --single-result /work/single.json \
 --base-evaluator /work/base.py --helper /workspace/evaluate_full_test_k100_onnx_int8.py \
 --fp32-result /workspace/outputs/k100_full_test_fp32/20260807-171433/result.json \
 --fp32-csv /workspace/outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv \
 --fp32-tensors /workspace/outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt \
 --frozen-inputs /work/frozen.npz --frozen-input-manifest /work/manifest.json --output-dir /work/output \
 2>&1 | tee "${RUN}/evaluation.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "${RC}" >"${RUN}/evaluation.exit"
{ date -Ins; hy-smi --showmeminfo vram || true; } >"${RUN}/host_post.txt" 2>&1
exit "${RC}"
