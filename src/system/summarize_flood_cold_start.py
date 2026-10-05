#!/usr/bin/env python3
'Summarize flood cold start measurements from raw records.'
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
METRICS = (
    "container_launch_to_first_result_seconds",
    "process_time_to_first_result_seconds",
    "config_and_input_identity_gate_seconds",
    "artifact_hash_verification_seconds",
    "model_hash_verification_seconds",
    "mxr_hash_verification_seconds",
    "runtime_plus_cached_session_initialization_seconds",
    "runner_load_seconds",
    "first_inference_ms",
)


def identity(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


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
            raise RuntimeError(f"cold-start RUN_STATUS failed: {root}")

    accepted: list[dict] = []
    excluded: list[dict] = []
    sources: list[dict] = []
    for variant in VARIANTS:
        directories = sorted(directory for root in roots for directory in (root / variant).glob("trial_*"))
        if len(directories) < 10:
            raise RuntimeError(f"{variant} has only {len(directories)} trials")
        for directory in directories:
            trial = int(directory.name.split("_")[-1])
            result_path = directory / "run/result.json"
            telemetry_path = directory / "telemetry.csv"
            result = json.loads(result_path.read_text(encoding="utf8"))
            launch_ns = int((directory / "container_launch_unix_ns.txt").read_text())
            first_ns = int(result["timing"]["first_result_unix_ns"])
            container_id = (directory / "container_id.txt").read_text().strip()
            sources.extend(identity(path) for path in (
                result_path,
                telemetry_path,
                directory / "container_id.txt",
                directory / "container_launch_unix_ns.txt",
                directory / "container_exit_unix_ns.txt",
            ))

            if (
                result["schema"] != "journal_phase6_process_cold_start_v1"
                or result["status"] != "PASSED"
                or not all(result["gates"].values())
                or result["input"]["sha256"] != EXPECTED_INPUT
                or result["data_role"] != "configuration-validation; formal test not accessed"
                or result["runtime"]["provider"] != "MIGraphXExecutionProvider"
                or result["runtime"]["cpu_fallback_disabled"] is not True
            ):
                raise RuntimeError(f"functional/lineage gate failed: {directory}")
            if first_ns <= launch_ns:
                raise RuntimeError(f"invalid timestamps: {directory}")

            rows = []
            with telemetry_path.open(newline="", encoding="utf8") as stream:
                for row in csv.DictReader(stream):
                    row["unix_ns"] = int(row["unix_ns"])
                    rows.append(row)
            before = [row for row in rows if row["unix_ns"] < launch_ns]
            window = [row for row in rows if launch_ns <= row["unix_ns"] <= first_ns]
            reasons = []
            if not before or any(row["kfd_pids"] for row in before):
                reasons.append("missing or nonempty paired pre-launch idle window")
            if not window:
                reasons.append("telemetry did not sample launch-to-first-result window")
            foreign_pids = set()
            own_samples = 0
            for row in window:
                cgroups = json.loads(row["kfd_cgroups_json"])
                for pid, cgroup in cgroups.items():
                    if container_id not in cgroup:
                        foreign_pids.add(pid)
                    else:
                        own_samples += 1
            if foreign_pids:
                reasons.append("foreign KFD cgroup in launch-to-first-result window")
            if own_samples == 0:
                reasons.append("no trial-container KFD sample in launch-to-first-result window")
            if reasons:
                excluded.append({
                    "variant": variant,
                    "trial": trial,
                    "reasons": reasons,
                    "foreign_pids": sorted(foreign_pids),
                })
                continue

            timing = result["timing"]
            accepted.append({
                "node": args.node,
                "hardware": "海光 K100 AI 加速卡",
                "variant": variant,
                "trial": trial,
                "independent_process": True,
                "cold_start_definition": "process cold; filesystem page cache not flushed",
                "container_launch_to_first_result_seconds": (first_ns - launch_ns) / 1e9,
                "process_time_to_first_result_seconds": timing["process_time_to_first_result_seconds"],
                "config_and_input_identity_gate_seconds": timing["config_and_input_identity_gate_seconds"],
                "artifact_hash_verification_seconds": timing["artifact_hash_verification_seconds"],
                "model_hash_verification_seconds": timing["model_hash_verification_seconds"],
                "mxr_hash_verification_seconds": timing["mxr_hash_verification_seconds"],
                "runtime_plus_cached_session_initialization_seconds": timing["runtime_plus_cached_session_initialization_seconds"],
                "runner_load_seconds": result["runtime"]["load_seconds"],
                "first_inference_ms": result["runtime"]["first_inference_ms"],
                "session_count": result["runtime"]["session_count"],
                "prediction_sha256": result["baseline"]["prediction_sha256"],
                "configuration_validation_input_sha256": EXPECTED_INPUT,
                "exclusive_kfd_cgroup_gate": True,
                
            })

    counts = {variant: sum(row["variant"] == variant for row in accepted) for variant in VARIANTS}
    if any(counts[variant] < 10 for variant in VARIANTS):
        raise RuntimeError(json.dumps({"insufficient_clean_trials": counts, "excluded": excluded}))

    result_csv = phase_root / "cold_start_results.csv"
    with result_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accepted[0]))
        writer.writeheader()
        writer.writerows(accepted)

    summaries = []
    for variant in VARIANTS:
        rows = [row for row in accepted if row["variant"] == variant]
        summary = {"variant": variant, "accepted_trials": len(rows)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"median_{metric}"] = float(np.median(values))
            summary[f"p95_{metric}"] = float(np.percentile(values, 95))
            summary[f"p99_{metric}"] = float(np.percentile(values, 99))
            summary[f"cv_{metric}"] = float(statistics.stdev(values) / statistics.fmean(values))
        summaries.append(summary)
    summary_csv = phase_root / "cold_start_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "schema": "journal_phase6_process_cold_start_summary_v1",
        "status": "PASSED",
        "node": args.node,
        "hardware": "海光 K100 AI 加速卡",
        "definition": "Each accepted trial used a new container and process. Filesystem page cache was not flushed; therefore this is process-cold, not disk-cold.",
        "timing_boundary": "Host launch-to-first-result includes detached container launch overhead; process time-to-first-result begins in the runner. MXR loading is not separable from ORT/MIGraphX cached Session initialization.",
        "accepted_counts": counts,
        "excluded_trials": excluded,
        "summaries": summaries,
        "exclusive_kfd_cgroup_gate": True,
        "configuration_validation_input_sha256": EXPECTED_INPUT,
        
        "source_files": sources,
    }
    (phase_root / "cold_start_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({"status": "PASSED", "accepted_counts": counts, "excluded": excluded, "summaries": summaries}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
