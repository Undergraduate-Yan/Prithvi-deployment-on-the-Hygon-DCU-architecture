#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 ENTRY_ROOT OUTPUT_ROOT SHIM_PATH CPU" >&2
  exit 2
fi

ENTRY=$(realpath "$1")
OUT=$(realpath -m "$2")
SHIM=$(realpath "$3")
CPU=$4
ASSETS="${PRITHVI_CLOUD_ASSETS:?Set the cloud runner asset directory}"
RUNNER="$ASSETS/runner/k100_pipeline_benchmark"
IMAGE="${PRITHVI_CONTAINER_IMAGE:?Set an installed vendor runtime image identity}"
INPUT="$ENTRY/deployment_validation_input_000.float32.raw"
ROLES=(Cloud-RCS-FP32-Compat Cloud-RCS-FP16-Opt-v2 Cloud-RCS-MP-Task-v2)

test ! -e "$OUT"
test -x "$RUNNER"
mkdir -p "$OUT/precheck" "$OUT/trials" "$OUT/environment"

stable_idle() {
  local record=$1 consecutive=0 attempt kfd
  : >"$record"
  for attempt in $(seq 1 600); do
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
  -v /opt/hyhal:/opt/hyhal:ro -v "${PRITHVI_WORKSPACE:?}:${PRITHVI_WORKSPACE}:rw" -v "$SHIM:/usr/bin/kmod:ro"
  -e MIGRAPHX_GPU_COMPILE_PARALLEL=1 -e ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1
  -e NUMEXPR_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2
)

{
  date --utc --iso-8601=ns
  hostname
  sha256sum "$RUNNER" "$ENTRY"/plans/*.tsv "$INPUT"
  printf 'hardware=海光 K100 AI 加速卡\nsessions=14\nrunner=Runner-A-Cloud-Contract-v2\n'
  printf 'warmups=50\nmeasured_calls=200\nfresh_processes=5\nscope=full pipeline plus FP32 logits D2H\n'
} >"$OUT/protocol.txt"

for role in "${ROLES[@]}"; do
  dir="$OUT/precheck/$role"
  mkdir -p "$dir/profiles"
  stable_idle "$dir/idle_gate.txt"
  docker run --rm "${docker_common[@]}" "$IMAGE" \
    --plan "$ENTRY/plans/$role.tsv" --input-raw "$INPUT" \
    --output-raw "$dir/logits.float32.raw" --latency-csv "$dir/latency.csv" \
    --scope logits --device 0 --warmups 1 --iterations 1 --profile-dir "$dir/profiles" \
    >"$dir/runner.log" 2>&1
  python3 - "$dir" <<'PY'
import csv,json,sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]); profiles=sorted((root/'profiles').glob('session_*.json'))
assert len(profiles)==14,len(profiles)
for path in profiles:
    counts=Counter(e.get('args',{}).get('provider') for e in json.load(open(path,encoding='utf8')))
    assert counts['MIGraphXExecutionProvider']>0 and counts['CPUExecutionProvider']==0,(path,counts)
rows=list(csv.DictReader(open(root/'latency.csv',newline='',encoding='utf8')))
assert len(rows)==1 and not rows[0]['error']
PY
done

orders=(
  'Cloud-RCS-FP32-Compat Cloud-RCS-FP16-Opt-v2 Cloud-RCS-MP-Task-v2'
  'Cloud-RCS-MP-Task-v2 Cloud-RCS-FP32-Compat Cloud-RCS-FP16-Opt-v2'
  'Cloud-RCS-FP16-Opt-v2 Cloud-RCS-MP-Task-v2 Cloud-RCS-FP32-Compat'
  'Cloud-RCS-FP32-Compat Cloud-RCS-MP-Task-v2 Cloud-RCS-FP16-Opt-v2'
  'Cloud-RCS-FP16-Opt-v2 Cloud-RCS-FP32-Compat Cloud-RCS-MP-Task-v2'
)

for round in 1 2 3 4 5; do
  position=0
  for role in ${orders[$((round - 1))]}; do
    position=$((position + 1))
    token="round_$(printf '%02d' "$round")_position_${position}_${role}"
    dir="$OUT/trials/$token"
    mkdir -p "$dir"
    stable_idle "$dir/idle_gate.txt"
    started=$(date --utc --iso-8601=ns)
    docker run --rm "${docker_common[@]}" "$IMAGE" \
      --plan "$ENTRY/plans/$role.tsv" --input-raw "$INPUT" \
      --output-raw "$dir/logits.float32.raw" --latency-csv "$dir/latency.csv" \
      --scope logits --device 0 --warmups 50 --iterations 200 >"$dir/runner.log" 2>&1
    ended=$(date --utc --iso-8601=ns)
    printf 'started=%s\nended=%s\nround=%s\nposition=%s\nrole=%s\n' \
      "$started" "$ended" "$round" "$position" "$role" >"$dir/trial_manifest.txt"
    python3 - "$dir/latency.csv" <<'PY'
import csv,math,sys
rows=list(csv.DictReader(open(sys.argv[1],newline='',encoding='utf8')))
assert len(rows)==200
assert all(not r['error'] and math.isfinite(float(r['latency_ms'])) and float(r['latency_ms'])>0 for r in rows)
PY
    echo "LATENCY_PASS round=$round position=$position role=$role" | tee -a "$OUT/progress.log"
  done
done

printf 'PASSED\n' >"$OUT/RUN_STATUS"
find "$OUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS.txt"
