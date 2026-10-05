#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 ENTRY OUTPUT SHIM CPU TELEMETRY LATENCY_ROOT ROLE [ROLE...]" >&2
  exit 2
fi

ENTRY=$(realpath "$1"); shift
OUT=$(realpath -m "$1"); shift
SHIM=$(realpath "$1"); shift
CPU=$1; shift
TELEMETRY=$(realpath "$1"); shift
LATENCY_ROOT=$(realpath "$1"); shift
ROLES=("$@")
ASSETS="${PRITHVI_CLOUD_ASSETS:?Set the cloud runner asset directory}"
RUNNER="$ASSETS/runner/k100_pipeline_benchmark"
IMAGE="${PRITHVI_CONTAINER_IMAGE:?Set an installed vendor runtime image identity}"
INPUT="$ENTRY/deployment_validation_input_000.float32.raw"

if test -f "${OUT}.skip"; then
  echo "AUXILIARY_RUN_SKIPPED_BY_FROZEN_GATE output=$OUT marker=${OUT}.skip" >&2
  exit 20
fi

test ! -e "$OUT"
test -s "$LATENCY_ROOT/RUN_STATUS"
mkdir -p "$OUT"/{vram,energy,cold_start,recovery,stability,environment}

stable_idle() {
  local record=$1 consecutive=0 attempt kfd
  : >"$record"
  for attempt in $(seq 1 900); do
    kfd=''
    if test -d /sys/class/kfd/kfd/proc; then
      kfd=$(find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f;' 2>/dev/null || true)
    fi
    printf '%s attempt=%s kfd=%s\n' "$(date --utc --iso-8601=ns)" "$attempt" "$kfd" >>"$record"
    if test -z "$kfd"; then consecutive=$((consecutive + 1)); else consecutive=0; fi
    if test "$consecutive" -ge 10; then return 0; fi
    sleep .2
  done
  return 1
}

docker_common=(
  --network none --hostname "$(hostname)" --pid=host --cpuset-cpus="$CPU"
  --pids-limit 1024 --memory 20g --device=/dev/kfd --device=/dev/dri
  --group-add video --ipc=host --shm-size=8g --entrypoint "$RUNNER"
  -v /opt/hyhal:/opt/hyhal:ro -v "${PRITHVI_WORKSPACE:?Set a common ancestor of code, assets and outputs}:${PRITHVI_WORKSPACE}:rw" -e PRITHVI_HWMON -e PRITHVI_DEVICE_SYSFS -v "$SHIM:/usr/bin/kmod:ro"
  -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 -e ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1
  -e NUMEXPR_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2
)

run_sampler() {
  local stop=$1 output=$2 interval=$3
  python3 "$TELEMETRY" --stop "$stop" --output "$output" --interval "$interval" &
  SAMPLER_PID=$!
}

stop_sampler() {
  local stop=$1
  touch "$stop"
  wait "$SAMPLER_PID"
}

run_kfd_watchdog() {
  local stop=$1 output=$2 kfd count
  : >"$output"
  while test ! -e "$stop"; do
    kfd=''
    if test -d /sys/class/kfd/kfd/proc; then
      kfd=$(find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f;' 2>/dev/null || true)
    fi
    count=0
    if test -n "$kfd"; then count=$(awk -F';' '{print NF-1}' <<<"$kfd"); fi
    printf '%s count=%s kfd=%s\n' "$(date --utc --iso-8601=ns)" "$count" "$kfd" >>"$output"
    sleep .5
  done
}

{
  date --utc --iso-8601=ns
  hostname
  sha256sum "$RUNNER" "$ENTRY"/plans/*.tsv "$INPUT" "$TELEMETRY"
  printf 'hardware=海光 K100 AI 加速卡\nrunner=Runner-A-Cloud-Contract-v2-final\n'
  printf 'energy_runs=5\nenergy_warmups=50\nenergy_iterations=1000\nenergy_telemetry_hz=10\n'
  printf 'cold_starts=10\nrecoveries=5\nstability_target_seconds=3600\n'
} >"$OUT/protocol.txt"

for role in "${ROLES[@]}"; do
  plan="$ENTRY/plans/$role.tsv"

  vram="$OUT/vram/$role"; mkdir -p "$vram"
  stable_idle "$vram/idle_gate.txt"
  timeout 10s /usr/local/hyhal/bin/hy-smi --showmemuse --showuse --showpids >"$vram/idle_hysmi.txt" 2>&1 || true
  run_sampler "$vram/sampler.stop" "$vram/telemetry.csv" .1
  sleep 1
  docker run --rm "${docker_common[@]}" "$IMAGE" --plan "$plan" --input-raw "$INPUT" \
    --output-raw "$vram/logits.raw" --latency-csv "$vram/latency.csv" --scope logits \
    --device 0 --warmups 50 --iterations 1000 >"$vram/runner.log" 2>&1
  stop_sampler "$vram/sampler.stop"
  timeout 10s /usr/local/hyhal/bin/hy-smi --showmemuse --showuse --showpids >"$vram/after_hysmi.txt" 2>&1 || true
  echo "VRAM_PASS role=$role" | tee -a "$OUT/progress.log"

  for run in 1 2 3 4 5; do
    energy="$OUT/energy/$role/run_$(printf '%02d' "$run")"; mkdir -p "$energy"
    stable_idle "$energy/idle_gate.txt"
    run_sampler "$energy/sampler.stop" "$energy/idle_telemetry.csv" .1
    sleep 5
    stop_sampler "$energy/sampler.stop"
    run_sampler "$energy/sampler.stop.run" "$energy/run_telemetry.csv" .1
    docker run --rm "${docker_common[@]}" "$IMAGE" --plan "$plan" --input-raw "$INPUT" \
      --output-raw "$energy/logits.raw" --latency-csv "$energy/latency.csv" --scope logits \
      --device 0 --warmups 50 --iterations 1000 >"$energy/runner.log" 2>&1
    stop_sampler "$energy/sampler.stop.run"
    echo "ENERGY_PASS role=$role run=$run" | tee -a "$OUT/progress.log"
  done

  for run in $(seq 1 10); do
    cold="$OUT/cold_start/$role/run_$(printf '%02d' "$run")"; mkdir -p "$cold"
    stable_idle "$cold/idle_gate.txt"
    date --utc --iso-8601=ns >"$cold/outer_started.txt"
    docker run --rm "${docker_common[@]}" "$IMAGE" --plan "$plan" --input-raw "$INPUT" \
      --output-raw "$cold/logits.raw" --latency-csv "$cold/latency.csv" --scope logits \
      --device 0 --warmups 0 --iterations 1 >"$cold/runner.log" 2>&1
    date --utc --iso-8601=ns >"$cold/outer_ended.txt"
  done
  echo "COLD_START_PASS role=$role" | tee -a "$OUT/progress.log"

  for run in $(seq 1 5); do
    recovery="$OUT/recovery/$role/run_$(printf '%02d' "$run")"; mkdir -p "$recovery"
    stable_idle "$recovery/idle_gate.txt"
    cname="cloud-recovery-${role,,}-${run}-$$"
    cid=$(docker run -d --name "$cname" "${docker_common[@]}" "$IMAGE" --plan "$plan" \
      --input-raw "$INPUT" --output-raw "$recovery/killed_logits.raw" \
      --latency-csv "$recovery/killed_latency.csv" --scope logits --device 0 \
      --warmups 50 --iterations 100000)
    printf '%s\n' "$cid" >"$recovery/killed_container_id.txt"
    sleep 3
    docker kill "$cid" >"$recovery/docker_kill.txt"
    docker inspect "$cid" >"$recovery/killed_container.inspect.json"
    docker logs "$cid" >"$recovery/killed_container.log" 2>&1 || true
    docker rm "$cid" >/dev/null
    stable_idle "$recovery/post_kill_idle_gate.txt"
    date --utc --iso-8601=ns >"$recovery/reload_started.txt"
    docker run --rm "${docker_common[@]}" "$IMAGE" --plan "$plan" --input-raw "$INPUT" \
      --output-raw "$recovery/reload_logits.raw" --latency-csv "$recovery/reload_latency.csv" \
      --scope logits --device 0 --warmups 0 --iterations 1 >"$recovery/reload_runner.log" 2>&1
    date --utc --iso-8601=ns >"$recovery/reload_ended.txt"
  done
  echo "RECOVERY_PASS role=$role" | tee -a "$OUT/progress.log"

  stability="$OUT/stability/$role"; mkdir -p "$stability"
  iterations=$(python3 - "$LATENCY_ROOT" "$role" <<'PY'
import csv,math,statistics,sys
from pathlib import Path
root,role=Path(sys.argv[1]),sys.argv[2]
values=[]
for path in root.glob(f'trials/*_{role}/latency.csv'):
    values.extend(float(r['latency_ms']) for r in csv.DictReader(open(path,newline='',encoding='utf8')))
if len(values)!=1000: raise SystemExit(f'expected 1000 pilot calls, got {len(values)}')
median=statistics.median(values)
print(min(250000,max(1,math.ceil(3610.0*1000.0/median))))
PY
)
  printf '%s\n' "$iterations" >"$stability/iterations.txt"
  stable_idle "$stability/idle_gate.txt"
  rm -f "$stability/kfd_watchdog.stop"
  run_kfd_watchdog "$stability/kfd_watchdog.stop" "$stability/kfd_watchdog.log" &
  KFD_WATCHDOG_PID=$!
  run_sampler "$stability/sampler.stop" "$stability/telemetry.csv" 1
  set +e
  docker run --rm "${docker_common[@]}" "$IMAGE" --plan "$plan" --input-raw "$INPUT" \
    --output-raw "$stability/logits.raw" --latency-csv "$stability/latency.csv" \
    --stability-signature-csv "$stability/signatures.csv" --scope logits --device 0 \
    --warmups 5000 --iterations "$iterations" >"$stability/runner.log" 2>&1
  runner_code=$?
  set -e
  stop_sampler "$stability/sampler.stop"
  touch "$stability/kfd_watchdog.stop"
  wait "$KFD_WATCHDOG_PID"
  test "$runner_code" -eq 0
  python3 - "$stability" <<'PY'
import csv,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=list(csv.DictReader(open(root/'latency.csv',newline='',encoding='utf8')))
sig=list(csv.DictReader(open(root/'signatures.csv',newline='',encoding='utf8')))
duration=(int(rows[-1]['ended_unix_ns'])-int(rows[0]['started_unix_ns']))/1e9
telemetry=list(csv.DictReader(open(root/'telemetry.csv',newline='',encoding='utf8')))
groups=[];unresolved=[]
for row in telemetry:
    for pid,value in json.loads(row['kfd_cgroups_json']).items():
        matches=re.findall(r'/docker/([0-9a-f]{64})',value)
        if not matches: unresolved.append({'pid':pid,'value':value})
        else: groups.extend(matches)
checks={'duration_ge_3600':duration>=3600,'errors_zero':all(not r['error'] for r in rows),
        'non_finite_zero':all(int(r['non_finite_count'])==0 for r in sig),
        'prediction_drift_zero':len({r['signature'] for r in sig})==1,
        'row_counts_match':len(rows)==len(sig),
        'kfd_exclusive_entire_run':bool(groups) and len(set(groups))==1 and not unresolved}
(root/'stability_gate.json').write_text(json.dumps({'status':'PASS' if all(checks.values()) else 'FAIL','duration_seconds':duration,'iterations':len(rows),'checks':checks,'kfd_container_cgroup_ids':sorted(set(groups)),'unresolved_kfd_cgroups':unresolved},indent=2,sort_keys=True)+'\n')
assert all(checks.values()),checks
PY
  echo "STABILITY_PASS role=$role iterations=$iterations" | tee -a "$OUT/progress.log"
done

printf 'PASSED\n' >"$OUT/RUN_STATUS"
find "$OUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS.txt"
