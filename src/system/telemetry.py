#!/usr/bin/env python3
'K100 power, temperature, clocks, VRAM and process telemetry.'
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import argparse, csv, json, time, os
from datetime import datetime, timezone
from pathlib import Path

HWMON = None
DEVICE = None
KFD = Path("/sys/class/kfd/kfd/proc")

def read(path): return int(path.read_text(encoding="utf8").strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    global HWMON, DEVICE
    HWMON = Path(os.environ["PRITHVI_HWMON"]).resolve(strict=True)
    DEVICE = Path(os.environ["PRITHVI_DEVICE_SYSFS"]).resolve(strict=True)
    if args.stop.exists() or args.output.exists() or not 0.05 <= args.interval <= 5.0:
        raise RuntimeError("sampler precondition failed")
    fields = ("timestamp_utc", "unix_ns", "monotonic_ns", "power_microwatts", "temp_edge_millic", "temp_junction_millic", "temp_mem_millic", "vram_used_bytes", "vram_total_bytes", "sclk_hz", "mclk_hz", "power_cap_microwatts", "kfd_pids", "kfd_cgroups_json")
    with args.output.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        while not args.stop.exists():
            pids = sorted(item.name for item in KFD.iterdir()) if KFD.exists() else []
            cgroups = {}
            for pid in pids:
                try: cgroups[pid] = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf8").strip().replace("\n", "|")
                except (FileNotFoundError, PermissionError): cgroups[pid] = "process_disappeared_during_sample"
            writer.writerow({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                "power_microwatts": read(HWMON / "power1_average"), "temp_edge_millic": read(HWMON / "temp1_input"),
                "temp_junction_millic": read(HWMON / "temp2_input"), "temp_mem_millic": read(HWMON / "temp3_input"),
                "vram_used_bytes": read(DEVICE / "mem_info_vram_used"), "vram_total_bytes": read(DEVICE / "mem_info_vram_total"),
                "sclk_hz": read(HWMON / "freq1_input"), "mclk_hz": read(HWMON / "freq2_input"),
                "power_cap_microwatts": read(HWMON / "power1_cap"), "kfd_pids": ";".join(pids),
                "kfd_cgroups_json": json.dumps(cgroups, sort_keys=True, separators=(",", ":")),
            })
            stream.flush(); time.sleep(args.interval)
    return 0

if __name__ == "__main__": raise SystemExit(main())
