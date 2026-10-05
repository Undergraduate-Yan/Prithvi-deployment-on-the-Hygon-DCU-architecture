#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PRITHVI_CONTAINER_IMAGE:?Set an installed vendor runtime image identity}"
ROOT="${PRITHVI_SYSTEM_ROOT:?Set the prepared system workspace}"; OUT="${ROOT}/07_vram_lineage_v2"; TOOLS="${PRITHVI_REPO:?Set the repository root}/src/system"
INPUT="${ROOT}/02_input/configuration_validation_input_00.npy"; KMOD_NOOP="${PRITHVI_SHIM:?Set the compiled compatibility shim}"
test ! -e "${OUT}"; mkdir -p "${OUT}/environment"; printf 'measuring\n' > "${OUT}/RUN_STATUS"
hostname > "${OUT}/environment/hostname.txt"; cat /proc/meminfo > "${OUT}/environment/meminfo.txt"
test "$(cat ${PRITHVI_DEVICE_SYSFS:?}/mem_info_vram_total)" = 68702699520

wait_idle() {
  local waited=0
  while test -n "$(docker ps -q)" || test -n "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null || true)"; do
    if (( waited % 30 == 0 )); then printf '%s waiting_for_idle seconds=%s\n' "$(date --iso-8601=seconds)" "${waited}"; fi
    test "${waited}" -lt 3600; sleep 5; waited=$((waited + 5))
  done
}

run_one() {
  local label="$1" token="$2" config="$3"
  local dir="${OUT}/${token}"
  local stop="${dir}/sampler.stop" cid rc cname="flood-vram-${token}-$$"
  wait_idle; mkdir -p "${dir}"
  /usr/local/hyhal/bin/hy-smi --showuse --showmemuse --showtemp --showpower --showpids > "${dir}/hysmi_before.txt" 2>&1
  python3 "${TOOLS}/vram_sampler.py" --stop "${stop}" --output "${dir}/vram_samples.csv" --interval 0.05 --vram-used "${PRITHVI_DEVICE_SYSFS}/mem_info_vram_used" --vram-total "${PRITHVI_DEVICE_SYSFS}/mem_info_vram_total" & sampler=$!
  sleep 3
  printf '%s\n' "$(date +%s%N)" > "${dir}/container_launch_unix_ns.txt"
  cid=$(docker run --network none --hostname "$(hostname)" --entrypoint python3 --pids-limit 1024 \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=8g \
    -v /opt/hyhal:/opt/hyhal:ro -v "${PRITHVI_WORKSPACE:?Set a common ancestor of code, assets and outputs}:${PRITHVI_WORKSPACE}:rw" -e PRITHVI_HWMON -e PRITHVI_DEVICE_SYSFS -v "${KMOD_NOOP}:/bin/kmod:ro" -v "${KMOD_NOOP}:/usr/sbin/lsmod:ro" \
    -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 -e ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0 -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2 \
    --name "${cname}" -d "${IMAGE}" "${TOOLS}/flood_system.py" --mode vram --label "${label}" \
    --config "${config}" --input "${INPUT}" --output-dir "${dir}/run" --warmup 0 --measured 500)
  printf '%s\n' "${cid}" > "${dir}/container_id.txt"
  set +e; rc=$(docker wait "${cid}"); set -e
  printf '%s\n' "$(date +%s%N)" > "${dir}/container_exit_unix_ns.txt"
  docker logs "${cid}" > "${dir}/container.log" 2>&1 || true
  docker inspect "${cid}" > "${dir}/container.inspect.json" || true
  docker rm "${cid}" >/dev/null
  sleep 3; touch "${stop}"; wait "${sampler}"
  /usr/local/hyhal/bin/hy-smi --showuse --showmemuse --showtemp --showpower --showpids > "${dir}/hysmi_after.txt" 2>&1
  test "${rc}" = 0; test -s "${dir}/run/result.json"; test -s "${dir}/vram_samples.csv"
  printf '%s variant=%s vram_passed\n' "$(date --iso-8601=seconds)" "${label}" | tee -a "${OUT}/progress.log"
}

run_one 'Mono-FP32' 'Mono_FP32' "${ROOT}/00_configs/Mono_FP32.json"
run_one 'Mono-FP16-Opt' 'Mono_FP16_Opt' "${ROOT}/00_configs/Mono_FP16_Opt.json"
run_one 'MP-RCS-Opt' 'MP_RCS_Opt' "${ROOT}/00_configs/MP_RCS_Opt.json"
wait_idle
find "${OUT}" -type f ! -name SHA256SUMS.txt ! -name RUN_STATUS -print0 | sort -z | xargs -0 sha256sum > "${OUT}/SHA256SUMS.txt"
printf 'PASSED\n' > "${OUT}/RUN_STATUS"
