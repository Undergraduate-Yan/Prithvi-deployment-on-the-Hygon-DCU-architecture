#!/usr/bin/env bash
set -euo pipefail
IMAGE='sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291'; ROOT='/var/tmp/20260815-phase11-int8-backbone-node3'; TOOLS="${ROOT}/tools"; RUN="${ROOT}/segment25_cached_end_to_end_single"; CONTAINER='phase11-int8-backbone-segment25-cached-single-node3'
exec 9>/var/tmp/phase11-int8-backbone-segment25-cached-single-node3.lock; flock -n 9 || exit 75
test "$(hostname)" = machine3; test -z "$(docker ps -q)"; test "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" = "${IMAGE}"
test "$(stat -c %s "${TOOLS}/lsmod")" = 819664; test "$(sha256sum "${TOOLS}/lsmod"|awk '{print $1}')" = 9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12
test "$(stat -c %s "${TOOLS}/evaluate_phase11_int8_backbone_segment25_cached_single.py")" = 8234; test "$(sha256sum "${TOOLS}/evaluate_phase11_int8_backbone_segment25_cached_single.py"|awk '{print $1}')" = 3c46bec2d2acf2484c355cfaeffdce03b678c00203ce68858079f421e22a37b1
test "$(cat "${ROOT}/segment25_static_remaining_caches/launcher.exit")" = 0; test "$(cat "${ROOT}/segment25_static_remaining_caches/LATEST_COMPLETED_SEGMENT")" = 24
test ! -e "${RUN}"; mkdir -p "${RUN}"; docker image inspect "${IMAGE}" >"${RUN}/image_inspect.json"; { date -Ins; hostname; printf 'variant=int8_backbone_compat_25_static_batch1_cached_host_staged_single\nperformance_claim=false\n'; free -h; /usr/local/hyhal/bin/hy-smi || true; } >"${RUN}/host_pre.txt" 2>&1
(
 while true; do date -Ins; docker stats --no-stream --format 'name={{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}' "${CONTAINER}" 2>/dev/null || true; grep '^MemAvailable:' /proc/meminfo || true; sleep 15; done
) >"${RUN}/telemetry.log" 2>&1 & TELEMETRY_PID=$!
set +e
docker run --rm --name "${CONTAINER}" --entrypoint python3 --memory=54g --pids-limit=1024 -e PATH=/work/root/tools:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g -v /opt/hyhal:/opt/hyhal:ro -v "${ROOT}:/work/root:ro" -v "${RUN}:/work/run:rw" -w /work "${IMAGE}" /work/root/tools/evaluate_phase11_int8_backbone_segment25_cached_single.py --root /work/root --output-dir /work/run/output >"${RUN}/evaluation.log" 2>&1
RC=$?; set -e; kill "${TELEMETRY_PID}" 2>/dev/null || true; wait "${TELEMETRY_PID}" 2>/dev/null || true; printf '%s\n' "${RC}" >"${RUN}/evaluation.exit"; { date -Ins; free -h; /usr/local/hyhal/bin/hy-smi || true; } >"${RUN}/host_post.txt" 2>&1; test ! -f "${RUN}/output/result.json" || cat "${RUN}/output/result.json"; exit "${RC}"
