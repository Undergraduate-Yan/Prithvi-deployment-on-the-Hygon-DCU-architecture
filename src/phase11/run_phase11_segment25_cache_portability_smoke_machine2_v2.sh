#!/usr/bin/env bash
set -euo pipefail

IMAGE='sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01'
BUNDLE='/var/tmp/phase11_int8_backbone_segment25_evidence_bundle_20260817'
RUN_ROOT='/var/tmp/20260817-phase11-segment25-crossnode-machine2'
RUN="${RUN_ROOT}/cache_portability_smoke_v2"
FAILED_ATTEMPT="${RUN_ROOT}/cache_portability_smoke"
PROBE="${BUNDLE}/tools/probe_phase11_int8_backbone_segment25_all_sessions_resident.py"
STRACE='/usr/bin/strace'
CONTAINER='phase11-segment25-cache-portability-smoke-machine2-v2'

exec 9>/var/tmp/phase11-segment25-cache-portability-smoke-machine2-v2.lock
flock -n 9 || exit 75
test "$(hostname)" = machine2
test -z "$(docker ps -q)"
test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${BUNDLE}/BUNDLE_MANIFEST.json")" = 126255
test "$(sha256sum "${BUNDLE}/BUNDLE_MANIFEST.json" | awk '{print $1}')" = 0abd5a99c3fa9ebcd4236bf6f57090f9fce102777535402313e081777eb9c6cc
test "$(stat -c %s /var/tmp/phase11_segment25_bundle_transfer_verification_machine2.json)" = 523
test "$(sha256sum /var/tmp/phase11_segment25_bundle_transfer_verification_machine2.json | awk '{print $1}')" = f27361082e0d871312eda7478f45c3f3d94a9defc7e0319102c6ffb72412cc09
test "$(stat -c %s /var/tmp/phase11_runtime_fingerprint_comparison_machine2.json)" = 4463
test "$(sha256sum /var/tmp/phase11_runtime_fingerprint_comparison_machine2.json | awk '{print $1}')" = 0feb3e326db7b004667d0eb7321bd34e3d0b04782c8c25d8f0dda9b3c937a8a6
test "$(stat -c %s "${PROBE}")" = 13356
test "$(sha256sum "${PROBE}" | awk '{print $1}')" = 62c93b1af40f0ed4baa7c2c15403bcdc227926e09c9944f62584696c283b7c98
test "$(stat -c %s "${BUNDLE}/tools/lsmod")" = 819664
test "$(sha256sum "${BUNDLE}/tools/lsmod" | awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test "$(stat -c %s "${STRACE}")" = 1489088
test "$(sha256sum "${STRACE}" | awk '{print $1}')" = e757b2a5fb6d2080521b8f7508981269654da1bb7bea9fc9910759187e0a97f4
test -f "${FAILED_ATTEMPT}/probe.exit"
test "$(tr -d '[:space:]' <"${FAILED_ATTEMPT}/probe.exit")" = 1
grep -Fq "Can't stat '/usr/local/bin/python3'" "${FAILED_ATTEMPT}/probe.log"
test ! -e "${RUN}"

mkdir -p "${RUN}"
docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"
cp /var/tmp/phase11_segment25_bundle_transfer_verification_machine2.json "${RUN}/"
cp /var/tmp/phase11_runtime_fingerprint_comparison_machine2.json "${RUN}/"
sha256sum "${FAILED_ATTEMPT}/probe.exit" "${FAILED_ATTEMPT}/probe.log" >"${RUN}/failed_attempt_identity.sha256"
{
  date -Ins
  hostname
  printf 'scope=cross_node_origin_cache_load_and_all_resident_single_smoke\n'
  printf 'retry_of=%s; retry_reason=incorrect_python_executable_path_before_probe_start\n' "${FAILED_ATTEMPT}"
  printf 'exact_image_identity=false; critical_runtime_content_match=true\n'
  printf 'strace=openat,read,mmap,munmap; bundle_mount=read_only\n'
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${RUN}/host_pre.txt" 2>&1

(
  while true; do
    date -Ins
    docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' "${CONTAINER}" 2>/dev/null || true
    grep '^MemAvailable:' /proc/meminfo || true
    /usr/local/hyhal/bin/hy-smi --showmeminfo vram 2>/dev/null || true
    sleep 3
  done
) >"${RUN}/telemetry.log" 2>&1 &
TELEMETRY_PID=$!

set +e
docker run --rm --name "${CONTAINER}" --entrypoint /usr/local/bin/strace \
  --ulimit stack=-1:-1 --memory=54g --pids-limit=1024 --network=none \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v "${STRACE}:/usr/local/bin/strace:ro" \
  -v "${BUNDLE}:/work/root:ro" -v "${RUN}:/work/run:rw" -w /work "${IMAGE}" \
  -ff -yy -qq -s 512 -e trace=openat,read,mmap,munmap -o /work/run/cache_trace \
  /usr/bin/python3 /work/root/tools/probe_phase11_int8_backbone_segment25_all_sessions_resident.py \
  --root /work/root --output-dir /work/run/output >"${RUN}/probe.log" 2>&1
RC=$?
set -e

kill "${TELEMETRY_PID}" 2>/dev/null || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
printf '%s\n' "${RC}" >"${RUN}/probe.exit"
{
  date -Ins
  free -h
  /usr/local/hyhal/bin/hy-smi || true
} >"${RUN}/host_post.txt" 2>&1
test ! -f "${RUN}/output/result.json" || cat "${RUN}/output/result.json"
exit "${RC}"
