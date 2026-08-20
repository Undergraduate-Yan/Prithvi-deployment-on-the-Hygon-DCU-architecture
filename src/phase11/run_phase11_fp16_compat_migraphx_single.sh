#!/usr/bin/env bash
set -euo pipefail

IMAGE_ID='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
SOURCE='/var/tmp/20260814-phase11-fp16-source-reassembled/prithvi300_upernet_fp16.onnx'
CANDIDATE='/var/tmp/20260814-phase11-fp16-compat-build/output/candidate.onnx'
BUILD_REPORT='/var/tmp/20260814-phase11-fp16-compat-build/output/build_report.json'
SAMPLE='/var/tmp/20260813-fp32-ln-numeric-parity-localdisk-rerun/input/sample_and_logits.pt'
SCRIPT='/var/tmp/evaluate_phase11_fp16_compat_migraphx_single.py'
RUN_ROOT='/var/tmp/20260814-phase11-fp16-compat-migraphx-single-cache-save'
OUTPUT="${RUN_ROOT}/output"
CACHE="${RUN_ROOT}/cache/prithvi_fp16_compat.mxr"

exec 9>/var/tmp/phase11-fp16-compat-migraphx-single.lock
flock -n 9 || exit 75
test "$(hostname)" = machine2
test "$(docker image inspect "${IMAGE_ID}" --format '{{.Id}}')" = "${IMAGE_ID}"
test -z "$(docker ps -q)"
test -d /opt/hyhal
test "$(stat -c %s "${SOURCE}")" = 638894819
test "$(sha256sum "${SOURCE}" | awk '{print $1}')" = 10df534d4dbaabf8336e4195a60d2ebd719b14c7a3e34362d48a0609425e6dbd
test "$(stat -c %s "${CANDIDATE}")" = 638970735
test "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" = 8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7
test "$(stat -c %s "${BUILD_REPORT}")" = 15591
test "$(sha256sum "${BUILD_REPORT}" | awk '{print $1}')" = 93293dc0909f5158f983ce9dc5b7953cf4956a116f73e54aab7e86e2a7cd3fba
test "$(stat -c %s "${SAMPLE}")" = 4820344
test "$(sha256sum "${SAMPLE}" | awk '{print $1}')" = 8ba78a29f324ab0bf3c188c8cb105e76b06925f22abb8c2da104a2fd87376e4b
test "$(stat -c %s "${SCRIPT}")" = 9824
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = ea20d8287dc6ea5f7d24dd7979ecad0a7cf2426be85c5efa91a1778ef94eb284
test ! -e "${RUN_ROOT}"
mkdir -p "${OUTPUT}" "$(dirname "${CACHE}")"
docker image inspect "${IMAGE_ID}" >"${RUN_ROOT}/image_inspect.json"
{
  date -Ins
  hostname
  docker ps -a
  hy-smi || true
  hy-smi --showmeminfo vram || true
} >"${RUN_ROOT}/host_pre.txt" 2>&1

set +e
docker run --rm \
  --name phase11-fp16-compat-migraphx-single \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=16g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "${SOURCE}:/work/source.onnx:ro" \
  -v "${CANDIDATE}:/work/candidate.onnx:ro" \
  -v "${BUILD_REPORT}:/work/build_report.json:ro" \
  -v "${SAMPLE}:/work/sample.pt:ro" \
  -v "${SCRIPT}:/work/evaluate.py:ro" \
  -v "${RUN_ROOT}:/work/run:rw" \
  -w /work "${IMAGE_ID}" python3 /work/evaluate.py \
    --source /work/source.onnx \
    --candidate /work/candidate.onnx \
    --build-report /work/build_report.json \
    --sample /work/sample.pt \
    --cache /work/run/cache/prithvi_fp16_compat.mxr \
    --output-dir /work/run/output \
  2>&1 | tee "${RUN_ROOT}/run.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
printf '%s\n' "${RUN_EXIT}" >"${RUN_ROOT}/run.exit"
{
  date -Ins
  docker ps -a
  hy-smi || true
  hy-smi --showmeminfo vram || true
} >"${RUN_ROOT}/host_post.txt" 2>&1
exit "${RUN_EXIT}"
