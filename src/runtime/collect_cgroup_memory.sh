#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER=''
OUTPUT=''
BUDGET_BYTES=''
INTERVAL='1'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --budget-bytes) BUDGET_BYTES="$2"; shift 2 ;;
    --interval-seconds) INTERVAL="$2"; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "${CONTAINER}" && -n "${OUTPUT}" && -n "${BUDGET_BYTES}" ]]

CID=$(docker inspect --format '{{.Id}}' "${CONTAINER}")
CGROUP=''
for candidate in \
  "/sys/fs/cgroup/memory/docker/${CID}" \
  "/sys/fs/cgroup/memory/system.slice/docker-${CID}.scope"; do
  if [[ -r "${candidate}/memory.usage_in_bytes" ]]; then
    CGROUP="${candidate}"
    break
  fi
done
[[ -n "${CGROUP}" ]] || { printf 'cgroup v1 memory path not found for %s\n' "${CID}" >&2; exit 3; }

mkdir -p "$(dirname "${OUTPUT}")"
printf 'timestamp_utc,cgroup_version,memory_usage_bytes,memory_max_usage_bytes,memory_failcnt,oom_kill_disable,under_oom,mem_available_bytes,budget_exceeded\n' >"${OUTPUT}"
while true; do
  [[ -r "${CGROUP}/memory.usage_in_bytes" ]] || break
  USAGE=$(cat "${CGROUP}/memory.usage_in_bytes" 2>/dev/null) || break
  PEAK=$(cat "${CGROUP}/memory.max_usage_in_bytes" 2>/dev/null) || break
  FAILCNT=$(cat "${CGROUP}/memory.failcnt" 2>/dev/null) || break
  read -r OOM_DISABLE UNDER_OOM < <(awk '/oom_kill_disable/{a=$2}/under_oom/{b=$2}END{print a,b}' "${CGROUP}/memory.oom_control" 2>/dev/null) || break
  MEM_AVAILABLE=$(awk '/^MemAvailable:/{print $2*1024}' /proc/meminfo)
  EXCEEDED=false
  if (( USAGE >= BUDGET_BYTES )); then EXCEEDED=true; fi
  printf '%s,1,%s,%s,%s,%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "${USAGE}" "${PEAK}" "${FAILCNT}" "${OOM_DISABLE}" "${UNDER_OOM}" "${MEM_AVAILABLE}" "${EXCEEDED}" >>"${OUTPUT}"
  if [[ "${EXCEEDED}" = true ]]; then
    printf '%s\n' "memory_budget_exceeded" >"${OUTPUT}.stop_reason"
    docker stop --time 5 "${CONTAINER}" >/dev/null 2>&1 || true
    break
  fi
  RUNNING=$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || printf false)
  [[ "${RUNNING}" = true ]] || break
  sleep "${INTERVAL}"
done
