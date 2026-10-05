#!/usr/bin/env bash
set -euo pipefail
IMAGE="${PRITHVI_CONTAINER_IMAGE:?Set an installed vendor runtime image identity}"
ROOT="${PRITHVI_SYSTEM_ROOT:?Set the prepared system workspace}"
OUT="${PHASE6_RECOVERY_OUT:-${ROOT}/10_recovery}"
TOOLS="${PRITHVI_REPO:?Set the repository root}/src/system"
INPUT="${ROOT}/02_input/configuration_validation_input_00.npy"
KMOD_NOOP="${PRITHVI_SHIM:?Set the compiled compatibility shim}"
test ! -e "$OUT"
mkdir -p "$OUT"
printf 'measuring\n' >"$OUT/RUN_STATUS"

stable_idle() {
  local n=0
  while ((n < 10)); do
    if test -z "$(docker ps -q)" && test -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null || true)"; then
      n=$((n + 1))
    else
      n=0
    fi
    sleep 1
  done
}

docker_common=(--network none --hostname "$(hostname)" --entrypoint python3 --pids-limit 1024 --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g -v /opt/hyhal:/opt/hyhal:ro -v "${PRITHVI_WORKSPACE:?Set a common ancestor of code, assets and outputs}:${PRITHVI_WORKSPACE}:rw" -e PRITHVI_HWMON -e PRITHVI_DEVICE_SYSFS -v "${KMOD_NOOP}:/bin/kmod:ro" -v "${KMOD_NOOP}:/usr/sbin/lsmod:ro" -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 -e ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2)

run_cycle() {
  local trial="$1" label="$2" token="$3" config="$4"
  local dir="${OUT}/${token}/trial_$(printf '%02d' "$trial")"
  local fault_name="flood-fault-${token}-${trial}-$$"
  local reload_name="flood-reload-${token}-${trial}-$$"
  local stop sampler fault_cid reload_cid rc i
  stable_idle
  mkdir -p "$dir"
  stop="$dir/sampler.stop"
  python3 "$TOOLS/telemetry.py" --stop "$stop" --output "$dir/telemetry.csv" --interval .2 &
  sampler=$!
  sleep 1

  printf '%s\n' "$(date +%s%N)" >"$dir/fault_container_launch_unix_ns.txt"
  fault_cid=$(docker run "${docker_common[@]}" --name "$fault_name" -d "$IMAGE" "$TOOLS/flood_system.py" --mode stability --label "$label" --config "$config" --input "$INPUT" --output-dir "$dir/faulted_run" --duration-seconds 600)
  printf '%s\n' "$fault_cid" >"$dir/fault_container_id.txt"
  for i in $(seq 1 900); do
    if test -f "$dir/faulted_run/stage.txt" && test "$(cat "$dir/faulted_run/stage.txt")" = steady_state; then break; fi
    if test "$(docker inspect -f '{{.State.Running}}' "$fault_cid" 2>/dev/null || true)" != true; then
      echo "fault target exited before steady state" >&2; exit 1
    fi
    sleep .2
  done
  test -f "$dir/faulted_run/stage.txt"
  test "$(cat "$dir/faulted_run/stage.txt")" = steady_state
  printf '%s\n' "$(date +%s%N)" >"$dir/fault_injection_unix_ns.txt"
  docker kill "$fault_cid" >"$dir/docker_kill_stdout.txt"
  rc=$(docker wait "$fault_cid")
  printf '%s\n' "$rc" >"$dir/fault_exit_code.txt"
  printf '%s\n' "$(date +%s%N)" >"$dir/fault_exit_observed_unix_ns.txt"
  docker logs "$fault_cid" >"$dir/fault_container.log" 2>&1 || true
  docker rm "$fault_cid" >/dev/null
  test "$rc" = 137

  stable_idle
  printf '%s\n' "$(date +%s%N)" >"$dir/post_kill_device_idle_unix_ns.txt"
  printf '%s\n' "$(date +%s%N)" >"$dir/reload_container_launch_unix_ns.txt"
  reload_cid=$(docker run "${docker_common[@]}" --name "$reload_name" -d "$IMAGE" "$TOOLS/flood_cold.py" --label "$label" --config "$config" --input "$INPUT" --output-dir "$dir/recovered_run")
  printf '%s\n' "$reload_cid" >"$dir/reload_container_id.txt"
  set +e
  rc=$(docker wait "$reload_cid")
  set -e
  printf '%s\n' "$rc" >"$dir/reload_exit_code.txt"
  printf '%s\n' "$(date +%s%N)" >"$dir/reload_exit_unix_ns.txt"
  docker logs "$reload_cid" >"$dir/reload_container.log" 2>&1 || true
  docker rm "$reload_cid" >/dev/null
  sleep 1
  touch "$stop"
  wait "$sampler"
  test "$rc" = 0
  test -s "$dir/recovered_run/result.json"
  echo "$(date --iso-8601=seconds) trial=$trial variant=$label recovery_passed" | tee -a "$OUT/progress.log"
}

FP32="$ROOT/00_configs/Mono_FP32.json"
FP16="$ROOT/00_configs/Mono_FP16_Opt.json"
MP="$ROOT/00_configs/MP_RCS_Opt.json"
if test "${PHASE6_RECOVERY_PLAN:-formal}" = replacement_v2; then
  run_cycle 6 Mono-FP32 Mono_FP32 "$FP32"
  run_cycle 6 Mono-FP16-Opt Mono_FP16_Opt "$FP16"
  run_cycle 6 MP-RCS-Opt MP_RCS_Opt "$MP"
  run_cycle 7 MP-RCS-Opt MP_RCS_Opt "$MP"
else
  for trial in $(seq 1 5); do
    case $((trial % 3)) in
      1) run_cycle "$trial" Mono-FP32 Mono_FP32 "$FP32"; run_cycle "$trial" Mono-FP16-Opt Mono_FP16_Opt "$FP16"; run_cycle "$trial" MP-RCS-Opt MP_RCS_Opt "$MP" ;;
      2) run_cycle "$trial" Mono-FP16-Opt Mono_FP16_Opt "$FP16"; run_cycle "$trial" MP-RCS-Opt MP_RCS_Opt "$MP"; run_cycle "$trial" Mono-FP32 Mono_FP32 "$FP32" ;;
      0) run_cycle "$trial" MP-RCS-Opt MP_RCS_Opt "$MP"; run_cycle "$trial" Mono-FP32 Mono_FP32 "$FP32"; run_cycle "$trial" Mono-FP16-Opt Mono_FP16_Opt "$FP16" ;;
    esac
  done
fi
stable_idle
find "$OUT" -type f ! -name SHA256SUMS.txt ! -name RUN_STATUS -print0 | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS.txt"
printf 'PASSED\n' >"$OUT/RUN_STATUS"
