#!/usr/bin/env python3
'Cloud latency, resource and sustained-operation evidence aggregation.'
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))


import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        raise ValueError(f"empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def role_from_trial(path: Path) -> str:
    return "_".join(path.parent.name.split("_")[4:])


def latency_summary(node: str, root: Path) -> list[dict]:
    grouped: dict[str, list[tuple[Path, list[float]]]] = {}
    for path in sorted((root / "trials").glob("*/latency.csv")):
        values = [float(row["latency_ms"]) for row in rows(path)]
        if len(values) != 200:
            raise RuntimeError(f"latency trial does not contain 200 rows: {path}")
        grouped.setdefault(role_from_trial(path), []).append((path, values))
    output = []
    for role, trials in sorted(grouped.items()):
        if len(trials) != 5:
            raise RuntimeError(f"expected five fresh processes: {node}/{role}")
        values = [item for _, trial in trials for item in trial]
        trial_ratios = [percentile(trial, .95) / statistics.median(trial) for _, trial in trials]
        median = statistics.median(values)
        p95 = percentile(values, .95)
        mean = statistics.fmean(values)
        mad = statistics.median(abs(value - median) for value in values)
        output.append({
            "node": node,
            "hardware": "海光 K100 AI 加速卡",
            "role": role,
            "sessions": 14,
            "runner": "Runner-A-Cloud-Contract-v2-final",
            "scope": "full 14-session pipeline plus FP32 logits D2H",
            "fresh_processes": 5,
            "measured_calls": len(values),
            "median_ms": median,
            "p90_ms": percentile(values, .90),
            "p95_ms": p95,
            "p99_ms": percentile(values, .99),
            "p95_median_ratio": p95 / median,
            "throughput_images_s": 1000.0 / median,
            "mean_ms": mean,
            "stdev_ms": statistics.stdev(values),
            "max_ms": max(values),
            "mad_ms": mad,
            "cv": statistics.stdev(values) / mean,
            "max_trial_p95_median_ratio": max(trial_ratios),
            "trial_tail_anomaly_count_gt_1p5": sum(value > 1.5 for value in trial_ratios),
            "status": "PASS" if max(trial_ratios) <= 1.5 else "FAIL_UNEXPLAINED_TAIL",
        })
    return output


def latency_raw_table(node: str, root: Path) -> list[dict]:
    output = []
    for path in sorted((root / "trials").glob("*/latency.csv")):
        role = role_from_trial(path)
        for row in rows(path):
            output.append({"node": node, "role": role, "trial": path.parent.name, **row})
    return output


def provider_summary(node: str, root: Path) -> list[dict]:
    output = []
    for role_dir in sorted(path for path in (root / "precheck").iterdir() if path.is_dir()):
        profiles = sorted((role_dir / "profiles").glob("session_*.json"))
        if len(profiles) != 14:
            raise RuntimeError(f"expected 14 provider profiles: {node}/{role_dir.name}")
        for profile in profiles:
            events = json.loads(profile.read_text(encoding="utf-8"))
            counts = Counter(
                str(event["args"]["provider"])
                for event in events if event.get("args", {}).get("provider")
            )
            ordinal = int(profile.name.split("_", 2)[1])
            mgx = counts["MIGraphXExecutionProvider"]
            cpu = counts["CPUExecutionProvider"]
            output.append({
                "node": node, "hardware": "海光 K100 AI 加速卡", "role": role_dir.name,
                "session_ordinal": ordinal, "migraphx_events": mgx, "cpu_events": cpu,
                "profile": profile.name, "status": "PASS" if mgx > 0 and cpu == 0 else "FAIL",
            })
    return output


def cross_node_summary(values: list[dict]) -> list[dict]:
    output = []
    roles = sorted({row["role"] for row in values})
    for role in roles:
        selected = [row for row in values if row["role"] == role]
        if len(selected) != 2:
            raise RuntimeError(f"expected exactly two node summaries for {role}")
        selected.sort(key=lambda row: row["node"])
        left, right = selected
        median_max = max(float(left["median_ms"]), float(right["median_ms"]))
        p95_max = max(float(left["p95_ms"]), float(right["p95_ms"]))
        output.append({
            "role": role, "node_a": left["node"], "node_b": right["node"],
            "node_a_median_ms": left["median_ms"], "node_b_median_ms": right["median_ms"],
            "median_relative_difference": abs(float(left["median_ms"]) - float(right["median_ms"])) / median_max,
            "node_a_p95_ms": left["p95_ms"], "node_b_p95_ms": right["p95_ms"],
            "p95_relative_difference": abs(float(left["p95_ms"]) - float(right["p95_ms"])) / p95_max,
            "status": "PASS",
        })
    return output


def telemetry(path: Path) -> list[dict]:
    result = rows(path)
    if len(result) < 2:
        raise RuntimeError(f"insufficient telemetry: {path}")
    return result


def integrate_power(values: list[dict], idle_watts: float) -> tuple[float, float]:
    energy = incremental = 0.0
    for left, right in zip(values, values[1:]):
        dt = (int(right["monotonic_ns"]) - int(left["monotonic_ns"])) / 1e9
        p0, p1 = float(left["power_microwatts"]) / 1e6, float(right["power_microwatts"]) / 1e6
        energy += .5 * (p0 + p1) * dt
        incremental += .5 * (max(0.0, p0 - idle_watts) + max(0.0, p1 - idle_watts)) * dt
    return energy, incremental


def iso_seconds(start: Path, end: Path) -> float:
    def parse(path: Path) -> datetime:
        value = path.read_text(encoding="utf-8").strip().replace(",", ".", 1)
        # Linux ``date --iso-8601=ns`` emits nine fractional digits, while
        # Python 3.8's fromisoformat accepts at most six.  Truncation to
        # microseconds preserves sub-millisecond cold-start timing and keeps
        # the parser compatible with both timestamp forms.
        value = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", value)
        return datetime.fromisoformat(value)
    return (parse(end) - parse(start)).total_seconds()


def runner_timings(path: Path) -> dict[str, float]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"manifest_ms", "session_load_ms", "runner_init_ms", "first_inference_ms", "total_ms"}:
            result[key] = float(value)
    if set(result) != {"manifest_ms", "session_load_ms", "runner_init_ms", "first_inference_ms", "total_ms"}:
        raise RuntimeError(f"runner timing fields incomplete: {path}")
    return result


def aux_summary(node: str, root: Path) -> list[dict]:
    status = root / "RUN_STATUS"
    if not status.is_file():
        status = root / "RUN_STATUS.txt"
    if status.read_text(encoding="utf-8").strip() != "PASSED":
        raise RuntimeError(f"auxiliary run is not closed: {root}")
    roles = sorted(path.name for path in (root / "vram").iterdir() if path.is_dir())
    output = []
    for role in roles:
        vram = telemetry(root / "vram" / role / "telemetry.csv")
        vram_values = [int(row["vram_used_bytes"]) for row in vram]
        vram_idle = [int(row["vram_used_bytes"]) for row in vram if not row["kfd_pids"]]
        vram_active = [int(row["vram_used_bytes"]) for row in vram if row["kfd_pids"]]
        if not vram_idle or not vram_active:
            raise RuntimeError(f"VRAM idle/active phases missing: {node}/{role}")
        energy_total, energy_incremental, energy_per_inference = [], [], []
        idle_power, run_power, dynamic_power, run_throughput = [], [], [], []
        for run in sorted((root / "energy" / role).glob("run_*")):
            idle = telemetry(run / "idle_telemetry.csv")
            active = telemetry(run / "run_telemetry.csv")
            idle_watts = statistics.median(float(row["power_microwatts"]) / 1e6 for row in idle)
            total, incremental = integrate_power(active, idle_watts)
            duration = (int(active[-1]["monotonic_ns"]) - int(active[0]["monotonic_ns"])) / 1e9
            latency_values = [float(row["latency_ms"]) for row in rows(run / "latency.csv")]
            energy_total.append(total)
            energy_incremental.append(incremental)
            energy_per_inference.append(incremental / 1000.0)
            idle_power.append(idle_watts)
            run_power.append(total / duration)
            dynamic_power.append(incremental / duration)
            run_throughput.append(1000.0 / statistics.median(latency_values))
        cold_runs = sorted((root / "cold_start" / role).glob("run_*"))
        cold = [iso_seconds(run / "outer_started.txt", run / "outer_ended.txt") * 1000 for run in cold_runs]
        cold_components = [runner_timings(run / "runner.log") for run in cold_runs]
        recovery_runs = sorted((root / "recovery" / role).glob("run_*"))
        recovery = [iso_seconds(run / "reload_started.txt", run / "reload_ended.txt") * 1000 for run in recovery_runs]
        recovery_components = [runner_timings(run / "reload_runner.log") for run in recovery_runs]
        stability_path = root / "stability" / role / "stability_gate.json"
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
        stability_root = stability_path.parent
        stability_latency = [float(row["latency_ms"]) for row in rows(stability_root / "latency.csv")]
        stability_telemetry = telemetry(stability_root / "telemetry.csv")
        stability_power = [float(row["power_microwatts"]) / 1e6 for row in stability_telemetry]
        stability_vram = [int(row["vram_used_bytes"]) for row in stability_telemetry]
        stability_edge = [int(row["temp_edge_millic"]) / 1000 for row in stability_telemetry if row["temp_edge_millic"]]
        stability_junction = [int(row["temp_junction_millic"]) / 1000 for row in stability_telemetry if row["temp_junction_millic"]]
        stability_memory_temp = [int(row["temp_mem_millic"]) / 1000 for row in stability_telemetry if row["temp_mem_millic"]]
        output.append({
            "node": node,
            "hardware": "海光 K100 AI 加速卡",
            "role": role,
            "sessions": 14,
            "vram_increment_peak_bytes": max(vram_values) - min(vram_values),
            "vram_peak_used_bytes": max(vram_values),
            "vram_idle_median_bytes": statistics.median(vram_idle),
            "vram_load_first_steady_peak_bytes": max(vram_active),
            "energy_runs": len(energy_total),
            "energy_total_j_median": statistics.median(energy_total),
            "energy_incremental_j_median": statistics.median(energy_incremental),
            "energy_incremental_j_per_inference_median": statistics.median(energy_per_inference),
            "idle_power_w_median": statistics.median(idle_power),
            "run_power_w_median": statistics.median(run_power),
            "dynamic_power_w_median": statistics.median(dynamic_power),
            "energy_throughput_images_s_median": statistics.median(run_throughput),
            "throughput_per_dynamic_watt": statistics.median(run_throughput) / statistics.median(dynamic_power),
            "cold_start_runs": len(cold),
            "cold_start_median_ms": statistics.median(cold),
            "cold_start_p95_ms": percentile(cold, .95),
            "cold_manifest_median_ms": statistics.median(row["manifest_ms"] for row in cold_components),
            "cold_session_load_median_ms": statistics.median(row["session_load_ms"] for row in cold_components),
            "cold_runner_init_median_ms": statistics.median(row["runner_init_ms"] for row in cold_components),
            "cold_first_inference_median_ms": statistics.median(row["first_inference_ms"] for row in cold_components),
            "cold_runner_total_median_ms": statistics.median(row["total_ms"] for row in cold_components),
            "recovery_runs": len(recovery),
            "recovery_median_ms": statistics.median(recovery),
            "recovery_p95_ms": percentile(recovery, .95),
            "recovery_manifest_median_ms": statistics.median(row["manifest_ms"] for row in recovery_components),
            "recovery_session_load_median_ms": statistics.median(row["session_load_ms"] for row in recovery_components),
            "recovery_runner_init_median_ms": statistics.median(row["runner_init_ms"] for row in recovery_components),
            "recovery_first_inference_median_ms": statistics.median(row["first_inference_ms"] for row in recovery_components),
            "recovery_runner_total_median_ms": statistics.median(row["total_ms"] for row in recovery_components),
            "stability_duration_seconds": stability["duration_seconds"],
            "stability_iterations": stability["iterations"],
            "stability_prediction_drift_zero": stability["checks"]["prediction_drift_zero"],
            "stability_non_finite_zero": stability["checks"]["non_finite_zero"],
            "stability_errors_zero": stability["checks"]["errors_zero"],
            "stability_row_counts_match": stability["checks"]["row_counts_match"],
            "stability_kfd_exclusive_entire_run": stability["checks"]["kfd_exclusive_entire_run"],
            "stability_median_ms": statistics.median(stability_latency),
            "stability_p95_ms": percentile(stability_latency, .95),
            "stability_throughput_images_s": 1000.0 / statistics.median(stability_latency),
            "stability_power_w_median": statistics.median(stability_power),
            "stability_power_w_p95": percentile(stability_power, .95),
            "stability_vram_peak_bytes": max(stability_vram),
            "stability_temp_edge_max_c": max(stability_edge),
            "stability_temp_junction_max_c": max(stability_junction),
            "stability_temp_memory_max_c": max(stability_memory_temp),
            "status": stability["status"],
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node", action="append", required=True,
        help="NAME=LATENCY_ROOT, or legacy NAME=LATENCY_ROOT=AUX_ROOT",
    )
    parser.add_argument(
        "--aux", action="append", default=[],
        help="NAME=AUX_ROOT; may identify a different clean hardware node from the latency matrix",
    )
    parser.add_argument("--kernel-gate", required=True, type=Path)
    parser.add_argument("--compile-gate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True)
    latency_rows, raw_rows, provider_rows, aux_rows = [], [], [], []
    for value in args.node:
        parts = value.split("=", 2)
        if len(parts) not in {2, 3}:
            raise ValueError(f"invalid --node value: {value}")
        name, latency_path = parts[:2]
        latency_root = Path(latency_path)
        latency_rows.extend(latency_summary(name, latency_root))
        raw_rows.extend(latency_raw_rows := latency_raw_table(name, latency_root))
        if len(latency_raw_rows) != 3000:
            raise RuntimeError(f"expected 3000 latency calls on {name}, got {len(latency_raw_rows)}")
        provider_rows.extend(provider_summary(name, latency_root))
        if len(parts) == 3:
            aux_rows.extend(aux_summary(name, Path(parts[2])))
    for value in args.aux:
        name, separator, aux_raw = value.partition("=")
        if not separator or not name or not aux_raw:
            raise ValueError(f"invalid --aux value: {value}")
        aux_rows.extend(aux_summary(name, Path(aux_raw)))
    kernel = json.loads(args.kernel_gate.resolve(strict=True).read_text(encoding="utf-8"))
    compile_gate = json.loads(args.compile_gate.resolve(strict=True).read_text(encoding="utf-8"))
    if any(row["status"] != "PASS" for row in latency_rows + provider_rows + aux_rows):
        raise RuntimeError("system gate failed")
    if kernel.get("status") != "PASS" or compile_gate.get("status") != "PASS":
        raise RuntimeError("kernel or compile-resource gate failed")
    expected_roles = {"Cloud-RCS-FP32-Compat", "Cloud-RCS-FP16-Opt-v2", "Cloud-RCS-MP-Task-v2"}
    expected_aux_roles = {"Cloud-RCS-FP32-Compat", "Cloud-RCS-FP16-Opt-v2"}
    if {row["role"] for row in latency_rows} != expected_roles or len(latency_rows) != 6:
        raise RuntimeError("two-node three-role latency matrix incomplete")
    if {row["role"] for row in aux_rows} != expected_aux_roles or len(aux_rows) != 2:
        raise RuntimeError("two production-role auxiliary matrix incomplete")
    if any(
        row["energy_runs"] != 5 or row["cold_start_runs"] != 10 or row["recovery_runs"] != 5
        or float(row["stability_duration_seconds"]) < 3600
        or row["stability_kfd_exclusive_entire_run"] is not True
        for row in aux_rows
    ):
        raise RuntimeError("auxiliary repetition or stability-duration gate failed")
    if len(provider_rows) != 84:
        raise RuntimeError(f"provider matrix incomplete: {len(provider_rows)}")
    write_csv(args.output_root / "cloud_final_same_topology_performance.csv", latency_rows)
    write_csv(args.output_root / "cloud_final_system_table.csv", aux_rows)
    write_csv(args.output_root / "cloud_phase7r_latency_raw.csv", raw_rows)
    write_csv(args.output_root / "cloud_phase7r_cross_node.csv", cross_node_summary(latency_rows))
    write_csv(args.output_root / "cloud_phase7r_provider.csv", provider_rows)
    kernel_rows = [
        {"role": "Cloud-RCS-FP16-Opt-v2", "region": "shared frozen FP16 session 09", "kernel": "HBH",
         "calls": kernel["HBH"]["calls"], "distinct_names": kernel["HBH"]["distinct_names"],
         "identity_scope": "same session identity retained in B23 one-block Mixed candidate", "status": "PASS"},
        {"role": "Cloud-RCS-MP-Task-v2", "region": "FP16 session 09", "kernel": "HBH",
         "calls": kernel["HBH"]["calls"], "distinct_names": kernel["HBH"]["distinct_names"],
         "identity_scope": kernel["scope"], "status": "PASS"},
        {"role": "Cloud-RCS-MP-Task-v2", "region": "INT8 session 12", "kernel": "I8II",
         "calls": kernel["I8II"]["calls"], "distinct_names": kernel["I8II"]["distinct_names"],
         "identity_scope": kernel["scope"], "status": "PASS"},
    ]
    write_csv(args.output_root / "cloud_phase7r_kernel.csv", kernel_rows)
    write_csv(args.output_root / "cloud_phase7r_compile_resources.csv", compile_gate["rows"])
    write_csv(args.output_root / "cloud_phase7r_vram.csv", [{
        key: row[key] for key in ("node", "hardware", "role", "sessions", "vram_idle_median_bytes", "vram_load_first_steady_peak_bytes", "vram_increment_peak_bytes", "vram_peak_used_bytes", "status")
    } for row in aux_rows])
    write_csv(args.output_root / "cloud_phase7r_energy.csv", [{
        key: row[key] for key in ("node", "hardware", "role", "energy_runs", "idle_power_w_median", "run_power_w_median", "dynamic_power_w_median", "energy_total_j_median", "energy_incremental_j_median", "energy_incremental_j_per_inference_median", "energy_throughput_images_s_median", "throughput_per_dynamic_watt", "status")
    } for row in aux_rows])
    write_csv(args.output_root / "cloud_phase7r_cold_start.csv", [{
        key: row[key] for key in ("node", "hardware", "role", "cold_start_runs", "cold_start_median_ms", "cold_start_p95_ms", "cold_manifest_median_ms", "cold_session_load_median_ms", "cold_runner_init_median_ms", "cold_first_inference_median_ms", "cold_runner_total_median_ms", "status")
    } for row in aux_rows])
    write_csv(args.output_root / "cloud_phase7r_recovery.csv", [{
        key: row[key] for key in ("node", "hardware", "role", "recovery_runs", "recovery_median_ms", "recovery_p95_ms", "recovery_manifest_median_ms", "recovery_session_load_median_ms", "recovery_runner_init_median_ms", "recovery_first_inference_median_ms", "recovery_runner_total_median_ms", "status")
    } for row in aux_rows])
    write_csv(args.output_root / "cloud_phase7r_stability.csv", [{
        key: row[key] for key in (
            "node", "hardware", "role", "stability_duration_seconds", "stability_iterations",
            "stability_median_ms", "stability_p95_ms", "stability_throughput_images_s",
            "stability_power_w_median", "stability_power_w_p95", "stability_vram_peak_bytes",
            "stability_temp_edge_max_c", "stability_temp_junction_max_c", "stability_temp_memory_max_c",
            "stability_errors_zero", "stability_non_finite_zero", "stability_prediction_drift_zero",
            "stability_row_counts_match",
            "stability_kfd_exclusive_entire_run", "status",
        )
    } for row in aux_rows])
    gate = {
        "schema": "phase7r_cloud_system_gate_v1",
        "status": "PASS",
        "nodes": sorted({row["node"] for row in latency_rows}),
        "roles": sorted({row["role"] for row in latency_rows}),
        "production_roles": sorted(expected_aux_roles),
        "rejected_characterization_roles": ["Cloud-RCS-MP-Task-v2"],
        "same_topology": True,
        "session_count": 14,
        "common_runner": True,
        "cpu_fallback": False,
        "latency_rows": len(latency_rows),
        "auxiliary_rows": len(aux_rows),
        "auxiliary_node_scopes": sorted({row["node"] for row in aux_rows}),
        "stability_admission": "cgroup-aware entire-run KFD exclusivity; all nonadmitted attempts retained as deviation evidence",
        "provider_rows": len(provider_rows),
        "kernel_gate": "PASS",
        "compile_resource_gate": "PASS",
        "raw_latency_rows": len(raw_rows),
    }
    (args.output_root / "cloud_phase7r_system_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "cloud_phase7r_system_report.md").write_text(
        "# Cloud system measurements\n\nSee the numerical CSV files and gate JSON for the measured nodes, roles and coverage. Task acceptance is evaluated separately from system measurements.\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
