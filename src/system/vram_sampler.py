#!/usr/bin/env python3
'High-frequency K100 VRAM sampling.'
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))


import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--vram-used", type=Path, default=Path("/sys/class/drm/card1/device/mem_info_vram_used"))
    parser.add_argument("--vram-total", type=Path, default=Path("/sys/class/drm/card1/device/mem_info_vram_total"))
    parser.add_argument("--kfd-proc", type=Path, default=Path("/sys/class/kfd/kfd/proc"))
    args = parser.parse_args()
    if args.stop.exists() or args.output.exists() or not 0.01 <= args.interval <= 1.0:
        raise RuntimeError("invalid sampler precondition")
    total = int(args.vram_total.read_text(encoding="utf-8").strip())
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("timestamp_utc", "unix_ns", "monotonic_ns", "vram_used_bytes", "vram_total_bytes", "kfd_pids", "kfd_cgroups_json"))
        writer.writeheader()
        while not args.stop.exists():
            pids = sorted(entry.name for entry in args.kfd_proc.iterdir()) if args.kfd_proc.exists() else []
            cgroups = {}
            for pid in pids:
                try:
                    cgroups[pid] = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").strip().replace("\n", "|")
                except (FileNotFoundError, PermissionError):
                    cgroups[pid] = "process_disappeared_during_sample"
            writer.writerow({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "unix_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(), "vram_used_bytes": int(args.vram_used.read_text(encoding="utf-8").strip()),
                "vram_total_bytes": total, "kfd_pids": ";".join(pids),
                "kfd_cgroups_json": json.dumps(cgroups, sort_keys=True, separators=(",", ":")),
            })
            stream.flush(); time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
