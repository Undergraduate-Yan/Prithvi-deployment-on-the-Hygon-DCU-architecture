#!/usr/bin/env python3
"""Summarize an isolated flood sustained-operation measurement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

EXPECTED_INPUT = "7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    raw = args.raw_dir.resolve(strict=True)
    wrapper_root = raw / "run_wrapper"
    result_path = wrapper_root / "run/result.json"
    telemetry_path = wrapper_root / "telemetry.csv"
    wrapper = json.loads((wrapper_root / "wrapper_result.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    inspect = json.loads((raw / "container_inspect.json").read_text(encoding="utf-8"))[0]
    cid = (raw / "container_id.txt").read_text().strip()

    rows = list(csv.DictReader(telemetry_path.open(newline="", encoding="utf-8")))
    start, end = result["measurement_window"]["started_unix_ns"], result["measurement_window"]["ended_unix_ns"]
    measured = [row for row in rows if start <= int(row["unix_ns"]) <= end]
    foreign_pids: set[str] = set()
    foreign_records = 0
    active_rows = exclusive_rows = 0
    for row in measured:
        cgroups = json.loads(row["kfd_cgroups_json"])
        if cgroups:
            active_rows += 1
        foreign = False
        own = False
        for pid, cgroup in cgroups.items():
            if cid in cgroup or cid[:12] in cgroup:
                own = True
            else:
                foreign = True
                foreign_pids.add(pid)
                foreign_records += 1
        if own and not foreign:
            exclusive_rows += 1
    intervals = np.diff(np.asarray([int(row["unix_ns"]) for row in measured], dtype=np.float64)) / 1e9
    gates = {
        "container_exit_zero": inspect["State"]["ExitCode"] == 0,
        "wrapper_passed": wrapper["status"] == "PASSED" and wrapper["runner_exit_code"] == 0 and wrapper["sampler_exit_code"] == 0,
        "runner_passed": result["status"] == "PASSED" and all(result["gates"].values()),
        "reported_variant": result["variant"] in {"Mono-FP32", "Mono-FP16-Opt", "MP-RCS-Opt"},
        "configuration_validation_input_fixed": result["input"]["sha256"] == EXPECTED_INPUT,
        "measurement_at_least_3600_seconds": result["measurement_window"]["wall_seconds"] >= 3600,
        "zero_errors_nonfinite_prediction_drift": result["measurements"]["errors"] == 0 and result["measurements"]["nonfinite"] == 0 and result["measurements"]["prediction_drift"] == 0,
        "minimum_700_telemetry_samples": len(measured) >= 700,
        "telemetry_interval_near_5_seconds": len(intervals) >= 699 and 4.0 <= float(np.median(intervals)) <= 6.0,
        "zero_foreign_kfd_records": foreign_records == 0,
        "all_active_kfd_rows_owned_by_test": active_rows > 0 and exclusive_rows == active_rows,
    }
    passed = all(gates.values())
    summary = {
        "schema": "phase6_fp32_isolated_stability_summary_v1", "status": "PASSED" if passed else "FAILED",
        "hardware": "海光 K100 AI 加速卡", "node": args.node, "variant": result["variant"],
        "requested_seconds": 3600, "actual_seconds": result["measurement_window"]["wall_seconds"],
        "calls": result["measurements"]["count"], "errors": result["measurements"]["errors"],
        "nonfinite": result["measurements"]["nonfinite"], "prediction_drift": result["measurements"]["prediction_drift"],
        "application_throughput_per_second": result["measurements"]["throughput_per_second"],
        "median_latency_ms": result["measurements"]["median_ms"], "p95_latency_ms": result["measurements"]["p95_ms"],
        "prediction_sha256": result["baseline"]["prediction_sha256"], "telemetry_samples": len(measured),
        "median_telemetry_interval_seconds": float(np.median(intervals)),
        "foreign_kfd_records": foreign_records, "foreign_kfd_pids": sorted(foreign_pids),
        "active_kfd_rows": active_rows, "exclusive_kfd_rows": exclusive_rows,
        "mean_power_w": statistics.fmean(int(row["power_microwatts"]) for row in measured) / 1e6,
        "max_edge_temp_c": max(int(row["temp_edge_millic"]) for row in measured) / 1000,
        "max_junction_temp_c": max(int(row["temp_junction_millic"]) for row in measured) / 1000,
        "max_memory_temp_c": max(int(row["temp_mem_millic"]) for row in measured) / 1000,
        "max_vram_bytes": max(int(row["vram_used_bytes"]) for row in measured),
        "gates": gates, "configuration_validation_input_sha256": EXPECTED_INPUT,
        "claim_boundary": "Application throughput includes Python scheduling, hashing, fixed-sample checks and telemetry; it does not replace the controlled short benchmark. Power, temperature and VRAM are attributable to the measured variant only because every measured KFD record is cgroup-exclusive.",
        "source_files": {"result": {"sha256": sha256(result_path)}, "telemetry": {"sha256": sha256(telemetry_path)}, "container_inspect": {"sha256": sha256(raw / "container_inspect.json")}},
    }
    summary_path = root / "flood_stability_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# Isolated flood 60-minute stability

Status: **{summary['status']}**

- Node: {summary['node']} / 海光 K100 AI 加速卡
- Duration: {summary['actual_seconds']:.3f} s
- Calls: {summary['calls']}
- Errors / non-finite / prediction drift: {summary['errors']} / {summary['nonfinite']} / {summary['prediction_drift']}
- Foreign KFD records: {summary['foreign_kfd_records']}
- Application throughput: {summary['application_throughput_per_second']:.3f} calls/s
- Mean power: {summary['mean_power_w']:.3f} W
- Max edge/junction/memory temperature: {summary['max_edge_temp_c']:.1f} / {summary['max_junction_temp_c']:.1f} / {summary['max_memory_temp_c']:.1f} C
- Max observed VRAM: {summary['max_vram_bytes']} bytes

Isolated telemetry supports attribution to the measured variant only within the recorded node and measurement scope.
"""
    (root / "flood_stability_report.md").write_text(report, encoding="utf-8")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    (root / "RUN_STATUS").write_text("PASSED\n" if passed else "FAILED\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "calls": summary["calls"], "foreign_kfd_records": foreign_records}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
