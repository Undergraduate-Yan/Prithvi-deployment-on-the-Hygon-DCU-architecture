#!/usr/bin/env python3
'Summarize flood recovery measurements from raw records.'
from __future__ import annotations
import argparse

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


VARIANTS = ("Mono_FP32", "Mono_FP16_Opt", "MP_RCS_Opt")
EXPECTED_INPUT = "7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80"
EXPECTED_PREDICTION = {
    "Mono_FP32": "0b8f242de4daa45f34f64281a093eacd9ca599c4fbcf95085e7a03a0eda53b76",
    "Mono_FP16_Opt": "63a88d1b2b40c0516f5e637cb7e1e98313c8e04402bd4c49ca364bec4cfa500a",
    "MP_RCS_Opt": "9b887ed3a90322fe6cd45775c87b9903e83cc847100269e477c98a465a5cbf8a",
}
METRICS = (
    "kill_to_device_idle_seconds",
    "reload_launch_to_first_result_seconds",
    "fault_injection_to_first_recovered_result_seconds",
    "recovered_process_time_to_first_result_seconds",
)


def identity(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_ns(directory: Path, name: str) -> int:
    return int((directory / name).read_text(encoding="utf8").strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path, action='append')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    roots = [p.resolve(strict=True) for p in args.input_root]
    root = roots[0]
    phase_root = args.output_dir
    for root in roots:
        if (root / "RUN_STATUS").read_text(encoding="utf8").strip() != "PASSED":
            raise RuntimeError(f"recovery RUN_STATUS failed: {root}")
    accepted, excluded, sources = [], [], []
    for variant in VARIANTS:
        directories = sorted(directory for root in roots for directory in (root / variant).glob("trial_*"))
        if len(directories) < 5:
            raise RuntimeError(f"{variant} has only {len(directories)} recovery cycles")
        for directory in directories:
            trial = int(directory.name.split("_")[-1])
            result_path = directory / "recovered_run/result.json"
            telemetry_path = directory / "telemetry.csv"
            result = json.loads(result_path.read_text(encoding="utf8"))
            fault_id = (directory / "fault_container_id.txt").read_text().strip()
            reload_id = (directory / "reload_container_id.txt").read_text().strip()
            fault_launch = read_ns(directory, "fault_container_launch_unix_ns.txt")
            fault_injection = read_ns(directory, "fault_injection_unix_ns.txt")
            fault_exit = read_ns(directory, "fault_exit_observed_unix_ns.txt")
            post_kill_idle = read_ns(directory, "post_kill_device_idle_unix_ns.txt")
            reload_launch = read_ns(directory, "reload_container_launch_unix_ns.txt")
            reload_exit = read_ns(directory, "reload_exit_unix_ns.txt")
            first_result = int(result["timing"]["first_result_unix_ns"])
            source_paths = (
                result_path, telemetry_path, directory / "fault_container_id.txt", directory / "reload_container_id.txt",
                directory / "fault_container_launch_unix_ns.txt", directory / "fault_injection_unix_ns.txt",
                directory / "fault_exit_observed_unix_ns.txt", directory / "post_kill_device_idle_unix_ns.txt",
                directory / "reload_container_launch_unix_ns.txt", directory / "reload_exit_unix_ns.txt",
                directory / "fault_exit_code.txt", directory / "reload_exit_code.txt",
                directory / "docker_kill_stdout.txt", directory / "faulted_run/stage_events.jsonl",
            )
            sources.extend(identity(path) for path in source_paths)

            stages = [json.loads(line) for line in (directory / "faulted_run/stage_events.jsonl").read_text().splitlines()]
            stage_names = [row["stage"] for row in stages]
            steady = next((row["unix_ns"] for row in stages if row["stage"] == "steady_state"), None)
            if (
                stage_names[:4] != ["process_started", "sessions_loaded", "first_inference", "steady_state"]
                or steady is None or steady > fault_injection
                or (directory / "fault_exit_code.txt").read_text().strip() != "137"
                or (directory / "reload_exit_code.txt").read_text().strip() != "0"
                or (directory / "docker_kill_stdout.txt").read_text().strip() != fault_id
                or not (fault_launch < steady <= fault_injection <= fault_exit < post_kill_idle <= reload_launch < first_result <= reload_exit)
            ):
                raise RuntimeError(f"active-termination timeline gate failed: {directory}")
            if (
                result["status"] != "PASSED" or not all(result["gates"].values())
                or result["input"]["sha256"] != EXPECTED_INPUT
                or result["baseline"]["prediction_sha256"] != EXPECTED_PREDICTION[variant]
                or result["runtime"]["provider"] != "MIGraphXExecutionProvider"
                or result["runtime"]["cpu_fallback_disabled"] is not True
            ):
                raise RuntimeError(f"recovered result gate failed: {directory}")

            rows = []
            with telemetry_path.open(newline="", encoding="utf8") as stream:
                for row in csv.DictReader(stream):
                    row["unix_ns"] = int(row["unix_ns"]); rows.append(row)
            pre = [row for row in rows if row["unix_ns"] < fault_launch]
            fault_window = [row for row in rows if fault_launch <= row["unix_ns"] <= fault_injection]
            reload_window = [row for row in rows if reload_launch <= row["unix_ns"] <= first_result]
            reasons, foreign = [], set()
            if not pre or any(row["kfd_pids"] for row in pre):
                reasons.append("missing or nonempty paired pre-fault idle window")
            for name, window, allowed in (("fault", fault_window, fault_id), ("reload", reload_window, reload_id)):
                if not window:
                    reasons.append(f"missing {name} telemetry window")
                    continue
                own = 0
                for row in window:
                    for pid, cgroup in json.loads(row["kfd_cgroups_json"]).items():
                        if allowed in cgroup: own += 1
                        else: foreign.add(pid)
                if own == 0: reasons.append(f"no own KFD sample in {name} window")
            if foreign: reasons.append("foreign KFD cgroup in exact fault/reload window")
            if reasons:
                excluded.append({"variant": variant, "trial": trial, "reasons": reasons, "foreign_pids": sorted(foreign)})
                continue

            accepted.append({
                "node": args.node, "hardware": "海光 K100 AI 加速卡", "variant": variant, "trial": trial,
                "active_termination": True, "fault_exit_code": 137, "reload_exit_code": 0,
                "steady_state_reached_before_kill": True,
                "kill_to_device_idle_seconds": (post_kill_idle - fault_injection) / 1e9,
                "reload_launch_to_first_result_seconds": (first_result - reload_launch) / 1e9,
                "fault_injection_to_first_recovered_result_seconds": (first_result - fault_injection) / 1e9,
                "recovered_process_time_to_first_result_seconds": result["timing"]["process_time_to_first_result_seconds"],
                "recovered_prediction_sha256": result["baseline"]["prediction_sha256"],
                "prediction_hash_restored": True, "exclusive_kfd_cgroup_gate": True,
                "configuration_validation_input_sha256": EXPECTED_INPUT,
                
            })

    counts = {variant: sum(row["variant"] == variant for row in accepted) for variant in VARIANTS}
    if any(counts[variant] < 5 for variant in VARIANTS):
        raise RuntimeError(json.dumps({"insufficient_clean_cycles": counts, "excluded": excluded}))
    with (phase_root / "recovery_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accepted[0])); writer.writeheader(); writer.writerows(accepted)
    summaries = []
    for variant in VARIANTS:
        rows = [row for row in accepted if row["variant"] == variant]
        summary = {"variant": variant, "accepted_cycles": len(rows), "active_kills": len(rows), "successful_reloads": len(rows)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"median_{metric}"] = float(np.median(values))
            summary[f"p95_{metric}"] = float(np.percentile(values, 95))
            summary[f"maximum_{metric}"] = float(np.max(values))
            summary[f"cv_{metric}"] = float(statistics.stdev(values) / statistics.fmean(values))
        summaries.append(summary)
    with (phase_root / "recovery_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    payload = {
        "schema": "journal_phase6_active_kill_reload_summary_v1", "status": "PASSED",
        "node": args.node, "hardware": "海光 K100 AI 加速卡", "accepted_counts": counts,
        "definition": "Each cycle actively docker-killed a steady-state container, required exit code 137 and an empty-device gate, then launched a new container and verified the configuration-specific prediction hash.",
        "timing_boundaries": {
            "kill_to_device_idle": "fault injection command timestamp to completion of the ten-second global device-idle gate",
            "reload_to_first_result": "new detached-container launch timestamp to recovered runner first-result timestamp",
            "end_to_end": "fault injection command timestamp to recovered runner first-result timestamp; includes controlled device-idle gate",
        },
        "summaries": summaries, "excluded_cycles": excluded, "exclusive_kfd_cgroup_gate": True,
        "configuration_validation_input_sha256": EXPECTED_INPUT,
        "source_files": sources,
    }
    (phase_root / "recovery_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASSED", "accepted_counts": counts, "excluded": excluded, "summaries": summaries}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
