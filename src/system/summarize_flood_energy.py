#!/usr/bin/env python3
'Summarize flood energy measurements from raw records.'
from __future__ import annotations
import argparse
import csv, hashlib, json, statistics
from pathlib import Path
import numpy as np

VARIANTS = ("Mono_FP32", "Mono_FP16_Opt", "MP_RCS_Opt")
EXPECTED = "7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80"

def ident(path):
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def integrate(rows, start, end):
    t = np.asarray([r["unix_ns"] for r in rows], dtype=np.float64) / 1e9
    p = np.asarray([r["power_microwatts"] for r in rows], dtype=np.float64) / 1e6
    if start / 1e9 < t.min() or end / 1e9 > t.max():
        raise RuntimeError("telemetry does not bracket measurement")
    mask = (t > start / 1e9) & (t < end / 1e9)
    x = np.concatenate(([start / 1e9], t[mask], [end / 1e9])); y = np.interp(x, t, p)
    energy = float(np.trapezoid(y, x))
    return energy, energy / ((end - start) / 1e9), int(mask.sum() + 2)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path, action='append')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    roots = [p.resolve(strict=True) for p in args.input_root]
    if len(roots) != 1: raise ValueError("one input root required")
    root = roots[0]
    phase_root = args.output_dir
    if (root / "RUN_STATUS").read_text(encoding="utf8").strip() != "PASSED":
        raise RuntimeError("energy RUN_STATUS failed")
    trials, audit, excluded = [], [], []
    for variant in VARIANTS:
        dirs = sorted((root / variant).glob("trial_*"))
        if len(dirs) < 5:
            raise RuntimeError(f"{variant} trial count {len(dirs)}")
        for directory in dirs:
            result_path = directory / "run/result.json"; telemetry_path = directory / "telemetry.csv"
            result = json.loads(result_path.read_text(encoding="utf8"))
            if result["status"] != "PASSED" or not all(result["gates"].values()) or result["input"]["sha256"] != EXPECTED or result["protocol"]["warmup"] != 50 or result["protocol"]["measured"] != 1000:
                raise RuntimeError(f"result gate {directory}")
            rows = []
            with telemetry_path.open(newline="", encoding="utf8") as stream:
                for row in csv.DictReader(stream):
                    for key in ("unix_ns", "power_microwatts", "temp_edge_millic", "temp_junction_millic", "temp_mem_millic", "vram_used_bytes", "vram_total_bytes", "sclk_hz", "mclk_hz", "power_cap_microwatts"):
                        row[key] = int(row[key])
                    rows.append(row)
            audit.extend(ident(path) for path in (result_path, telemetry_path, directory / "container_id.txt"))
            launch = int((directory / "container_launch_unix_ns.txt").read_text()); exit_ns = int((directory / "container_exit_unix_ns.txt").read_text())
            container_id = (directory / "container_id.txt").read_text().strip()
            start = result["measurement_window"]["started_unix_ns"]; end = result["measurement_window"]["ended_unix_ns"]
            idle = [row for row in rows if row["unix_ns"] < launch]; post = [row for row in rows if row["unix_ns"] >= exit_ns]
            measured = [row for row in rows if start <= row["unix_ns"] <= end]
            trial_index = int(directory.name.split("_")[-1])
            if not idle or not post or {row["kfd_pids"] for row in idle} != {""}:
                excluded.append({"variant": variant, "trial": trial_index, "reason": "nonempty or missing paired idle window"}); continue
            foreign = []
            for row in measured:
                for pid, cgroup in json.loads(row["kfd_cgroups_json"]).items():
                    if container_id not in cgroup:
                        foreign.append((pid, cgroup))
            if foreign:
                excluded.append({"variant": variant, "trial": trial_index, "reason": "foreign KFD cgroup in exact measurement window", "foreign_pids": sorted({pid for pid, _ in foreign})}); continue
            if {row["power_cap_microwatts"] for row in rows} != {400000000}:
                raise RuntimeError(f"power cap drift {directory}")
            if len(measured) < 50:
                raise RuntimeError(f"too few power samples {directory}: {len(measured)}")
            energy, run_mean, integrated_samples = integrate(rows, start, end)
            idle_mean = statistics.fmean(row["power_microwatts"] for row in idle) / 1e6; dynamic = run_mean - idle_mean
            if dynamic <= 0:
                raise RuntimeError(f"nonpositive dynamic power {directory}")
            duration = (end - start) / 1e9; throughput = result["measurements"]["throughput_per_second"]
            trials.append({
                "node": args.node, "hardware": "海光 K100 AI 加速卡", "variant": variant, "trial": trial_index,
                "independent_process": True, "warmup": 50, "measured_inferences": 1000, "sampling_hz": 10,
                "idle_power_w": idle_mean, "run_power_w": run_mean, "dynamic_power_w": dynamic,
                "total_energy_j": energy, "total_energy_per_inference_j": energy / 1000,
                "dynamic_energy_per_inference_j": dynamic * duration / 1000,
                "throughput_per_second": throughput, "throughput_per_total_watt": throughput / run_mean,
                "throughput_per_dynamic_watt": throughput / dynamic, "measurement_seconds": duration,
                "integrated_samples": integrated_samples,
                "max_temp_edge_c": max(row["temp_edge_millic"] for row in measured) / 1000,
                "max_temp_junction_c": max(row["temp_junction_millic"] for row in measured) / 1000,
                "max_temp_mem_c": max(row["temp_mem_millic"] for row in measured) / 1000,
                "max_vram_used_bytes": max(row["vram_used_bytes"] for row in measured), "power_cap_w": 400,
                "exclusive_kfd_cgroup_gate": True, "configuration_validation_input_sha256": EXPECTED,
                
            })
        if len([row for row in trials if row["variant"] == variant]) != 5:
            raise RuntimeError(f"{variant} does not have exactly five clean trials")
    result_csv = args.output_dir / "energy_results.csv"
    with result_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0])); writer.writeheader(); writer.writerows(trials)
    summaries = []
    for variant in VARIANTS:
        values = [row for row in trials if row["variant"] == variant]
        summaries.append({"variant": variant, "trials": 5, **{f"median_{key}": statistics.median(row[key] for row in values) for key in ("idle_power_w", "run_power_w", "dynamic_power_w", "total_energy_per_inference_j", "dynamic_energy_per_inference_j", "throughput_per_second", "throughput_per_total_watt", "throughput_per_dynamic_watt")}})
    summary_csv = args.output_dir / "energy_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    payload = {
        "schema": "journal_phase6_energy_summary_v1", "status": "PASSED", "node": args.node,
        "hardware": "海光 K100 AI 加速卡", "power_cap_w": 400, "sampling_hz": 10,
        "trials": trials, "summaries": summaries, "excluded_trials": excluded,
        "definitions": {"idle_power": "mean during 10-second pre-container device-idle window", "run_power": "trapezoidal mean over runner's exact measured window", "dynamic_power": "run minus paired idle", "energy_per_inference": "integrated measured-window energy divided by 1000"},
        "exclusive_kfd_cgroup_gate": True, "source_files": audit,
    }
    (args.output_dir / "energy_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASSED", "trials": len(trials), "excluded": excluded, "summary": summaries})); return 0

if __name__ == "__main__":
    raise SystemExit(main())
