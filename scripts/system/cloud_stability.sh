#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ENTRY OUTPUT SHIM CPU TELEMETRY LATENCY_ROOT ROLE" >&2
  exit 2
fi

ENTRY=$(realpath "$1")
OUT=$(realpath -m "$2")
SHIM=$(realpath "$3")
CPU=$4
TELEMETRY=$(realpath "$5")
LATENCY_ROOT=$(realpath "$6")
ROLE=$7
ASSETS="${PRITHVI_CLOUD_ASSETS:?Set the cloud runner asset directory}"
RUNNER="$ASSETS/runner/k100_pipeline_benchmark"
IMAGE="${PRITHVI_CONTAINER_IMAGE:?Set an installed vendor runtime image identity}"
INPUT="$ENTRY/deployment_validation_input_000.float32.raw"
PLAN="$ENTRY/plans/$ROLE.tsv"

test ! -e "$OUT"
test -s "$LATENCY_ROOT/RUN_STATUS"
mkdir -p "$OUT/stability/$ROLE"
STABILITY="$OUT/stability/$ROLE"

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

run_sampler() {
  python3 "$TELEMETRY" --stop "$1" --output "$2" --interval "$3" &
  SAMPLER_PID=$!
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

iterations=$(python3 - "$LATENCY_ROOT" "$ROLE" <<'PY'
import csv,math,statistics,sys
from pathlib import Path
root,role=Path(sys.argv[1]),sys.argv[2]
values=[]
for path in root.glob(f'trials/*_{role}/latency.csv'):
    values.extend(float(row['latency_ms']) for row in csv.DictReader(open(path,newline='',encoding='utf8')))
if len(values)!=1000: raise SystemExit(f'expected 1000 pilot calls, got {len(values)}')
# The pilot may come from another K100 node.  Use a conservative 4800-second
# iteration budget so a faster clean node still clears the frozen 3600-second
# measured-call gate without changing the gate itself.
print(min(250000,max(1,math.ceil(4800.0*1000.0/statistics.median(values)))))
PY
)
printf '%s\n' "$iterations" >"$STABILITY/iterations.txt"
{
  date --utc --iso-8601=ns
  hostname
  sha256sum "$RUNNER" "$PLAN" "$INPUT" "$TELEMETRY"
  printf 'hardware=海光 K100 AI 加速卡\nrole=%s\nresume_scope=full_stability_repeat_after_invalid_attempts\niteration_budget_seconds=4800\nfrozen_acceptance_duration_seconds=3600\n' "$ROLE"
} >"$OUT/protocol.txt"

docker_common=(
  --network none --hostname "$(hostname)" --pid=host --cpuset-cpus="$CPU"
  --pids-limit 1024 --memory 20g --device=/dev/kfd --device=/dev/dri
  --group-add video --ipc=host --shm-size=8g --entrypoint "$RUNNER"
  -v /opt/hyhal:/opt/hyhal:ro -v "${PRITHVI_WORKSPACE:?Set a common ancestor of code, assets and outputs}:${PRITHVI_WORKSPACE}:rw" -e PRITHVI_HWMON -e PRITHVI_DEVICE_SYSFS -v "$SHIM:/usr/bin/kmod:ro"
  -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 -e ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1
  -e NUMEXPR_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2
)

stable_idle "$STABILITY/idle_gate.txt"
run_kfd_watchdog "$STABILITY/kfd_watchdog.stop" "$STABILITY/kfd_watchdog.log" &
KFD_WATCHDOG_PID=$!
run_sampler "$STABILITY/sampler.stop" "$STABILITY/telemetry.csv" 1
set +e
docker run --rm "${docker_common[@]}" "$IMAGE" --plan "$PLAN" --input-raw "$INPUT" \
  --output-raw "$STABILITY/logits.raw" --latency-csv "$STABILITY/latency.csv" \
  --stability-signature-csv "$STABILITY/signatures.csv" --scope logits --device 0 \
  --warmups 5000 --iterations "$iterations" >"$STABILITY/runner.log" 2>&1
runner_code=$?
set -e
touch "$STABILITY/sampler.stop" "$STABILITY/kfd_watchdog.stop"
wait "$SAMPLER_PID"
wait "$KFD_WATCHDOG_PID"
test "$runner_code" -eq 0

python3 - "$STABILITY" <<'PY'
import csv,json,re,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=list(csv.DictReader(open(root/'latency.csv',newline='',encoding='utf8')))
sig=list(csv.DictReader(open(root/'signatures.csv',newline='',encoding='utf8')))
duration=(int(rows[-1]['ended_unix_ns'])-int(rows[0]['started_unix_ns']))/1e9
telemetry=list(csv.DictReader(open(root/'telemetry.csv',newline='',encoding='utf8')))
groups=[];unresolved=[]
for row in telemetry:
    for pid,value in json.loads(row['kfd_cgroups_json']).items():
        matches=re.findall(r'/docker/([0-9a-f]{64})',value)
        if not matches: unresolved.append({'pid':pid,'value':value})
        else: groups.extend(matches)
checks={'duration_ge_3600':duration>=3600,'errors_zero':all(not row['error'] for row in rows),
        'non_finite_zero':all(int(row['non_finite_count'])==0 for row in sig),
        'prediction_drift_zero':len({row['signature'] for row in sig})==1,
        'row_counts_match':len(rows)==len(sig),
        'kfd_exclusive_entire_run':bool(groups) and len(set(groups))==1 and not unresolved}
value={'schema':'phase7r_stability_gate_v2','status':'PASS' if all(checks.values()) else 'FAIL',
       'duration_seconds':duration,'iterations':len(rows),'checks':checks,
       'kfd_container_cgroup_ids':sorted(set(groups)),'unresolved_kfd_cgroups':unresolved}
(root/'stability_gate.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
assert all(checks.values()),checks
PY

printf 'PASSED\n' >"$OUT/RUN_STATUS"
find "$OUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS.txt"
echo "STABILITY_ONLY_PASS role=$ROLE iterations=$iterations"
