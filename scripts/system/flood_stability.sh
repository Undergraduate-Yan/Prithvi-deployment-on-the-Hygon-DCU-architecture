#!/usr/bin/env bash
# Measure one flood variant in an isolated container for the registered duration.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 LABEL RESOLVED_CONFIG INPUT_NPY OUTPUT_DIRECTORY" >&2
  exit 2
fi
LABEL=$1
CONFIG=$(realpath "$2")
INPUT=$(realpath "$3")
OUT=$(realpath -m "$4")
: "${PRITHVI_REPO:?}" "${PRITHVI_WORKSPACE:?}" "${PRITHVI_CONTAINER_IMAGE:?}"
: "${PRITHVI_SHIM:?}" "${PRITHVI_HWMON:?}" "${PRITHVI_DEVICE_SYSFS:?}"
test ! -e "$OUT"
test -d /sys/class/kfd/kfd/proc
test -z "$(ls -A /sys/class/kfd/kfd/proc)"
mkdir -p "$OUT"
TOOLS="$PRITHVI_REPO/src/system"
cid=$(docker run -d --network none --pid=host --hostname "$(hostname)" \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
  -v "$PRITHVI_WORKSPACE:$PRITHVI_WORKSPACE:rw" -v /opt/hyhal:/opt/hyhal:ro \
  -v "$PRITHVI_SHIM:/bin/kmod:ro" -v "$PRITHVI_SHIM:/usr/sbin/lsmod:ro" \
  -e PRITHVI_HWMON -e PRITHVI_DEVICE_SYSFS \
  -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 -e ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2 --entrypoint python3 \
  "$PRITHVI_CONTAINER_IMAGE" "$TOOLS/flood_stability.py" --label "$LABEL" \
  --config "$CONFIG" --input "$INPUT" --output-dir "$OUT/run_wrapper" \
  --duration-seconds 3600 --runner "$TOOLS/flood_system.py" --sampler "$TOOLS/telemetry.py")
printf '%s\n' "$cid" > "$OUT/container_id.txt"
rc=$(docker wait "$cid")
docker logs "$cid" > "$OUT/container.log" 2>&1
docker inspect "$cid" > "$OUT/container_inspect.json"
docker rm "$cid" > /dev/null
printf '%s\n' "$rc" > "$OUT/container_exit_code.txt"
test "$rc" = 0
printf 'MEASUREMENT_COMPLETE_REQUIRES_SUMMARY\n' > "$OUT/RUN_STATUS"
