#!/usr/bin/env python3
"""Aggregate the rotated 3x100 task-equivalence performance campaign."""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

import phase11_mixed_precision_common as common


EXPECTED_TRIAL_SCRIPT_SHA256 = "836b890a6fecbaf39658578dd3b0d7bf22c6b1d60ee5cacc6a0a4c3d514fd113"
EXPECTED_COMMON_MODULE_SHA256 = "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8"
EXPECTED_TRIALS = 3
EXPECTED_MEASURED = 100
MEDIAN_CV_MAX_PERCENT = 5.0
SCOPES = ("model_only", "end_to_end_logits")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_FP32_AGGREGATE_SCRIPT_SHA256 = "25bfeaefa2982c33e304bacae188fd4e23b30097b46c97f158d21acab2df9706"


def parse_iso8601(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} must include timezone information")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-schedule", default="M0,M1,M2,M3,M4")
    parser.add_argument("--fp32-summary", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if (root / "summary.json").exists() or (root / "summary.csv").exists():
        raise RuntimeError("refusing to overwrite an existing frozen performance summary")
    schedule = common.parse_schedule(args.candidate_schedule)
    fp32_identity = common.identity(args.fp32_summary)
    fp32 = common.load_json(args.fp32_summary)
    if fp32.get("schema") != "phase11_fp32_segment25_headbarrier_performance_summary_v1":
        raise RuntimeError("FP32 performance summary schema drift")
    if fp32.get("status") != "passed":
        raise RuntimeError("same-protocol FP32 summary did not pass")
    if fp32.get("identities", {}).get("aggregate_script", {}).get("sha256") != (
        EXPECTED_FP32_AGGREGATE_SCRIPT_SHA256
    ):
        raise RuntimeError("FP32 aggregate script identity drift")
    fp32_claims = fp32.get("claims", {})
    if not fp32_claims.get("same_protocol_fp32_baseline_valid") or not fp32_claims.get(
        "formal_speedup_denominator_allowed"
    ):
        raise RuntimeError("FP32 summary is not an admitted formal speedup denominator")
    if fp32.get("protocol", {}).get("track") != "same_protocol_fp32_reference":
        raise RuntimeError("FP32 protocol track drift")
    fp32_scopes = fp32.get("scope_summaries", {})
    for scope in SCOPES:
        row = fp32_scopes.get(scope, {})
        if float(row.get("median_of_trial_medians_ms", 0)) <= 0:
            raise RuntimeError(f"invalid FP32 median for {scope}")
        if float(row.get("trial_median_cv_percent", float("inf"))) > MEDIAN_CV_MAX_PERCENT:
            raise RuntimeError(f"unstable FP32 denominator for {scope}")
    records = []
    csv_rows = []
    manifest_by_candidate = {}
    runtime_fingerprint_identity = None
    container_image_id = None

    for trial_index in range(1, EXPECTED_TRIALS + 1):
        order = common.rotated_schedule(schedule, trial_index)
        for position, candidate_id in enumerate(order, start=1):
            output_dir = root / f"trial_{trial_index:02d}" / f"position_{position:02d}_{candidate_id}" / "output"
            result_path = output_dir / "result.json"
            result = common.load_json(result_path)
            if result.get("schema") != "phase11_mixed_precision_task_equivalence_performance_trial_v1":
                raise RuntimeError(f"trial result schema drift: {result_path}")
            if result.get("status") != "passed" or not result.get("claims", {}).get(
                "task_equivalence_performance_trial_valid"
            ):
                raise RuntimeError(f"failed performance trial: {result_path}")
            if result.get("trial_index") != trial_index:
                raise RuntimeError(f"trial index drift: {result_path}")
            lineage = result.get("candidate_lineage", {})
            if lineage.get("candidate_id") != candidate_id:
                raise RuntimeError(f"candidate ID drift: {result_path}")
            schedule_result = result.get("schedule", {})
            if schedule_result.get("rotated_trial_order") != list(order):
                raise RuntimeError(f"rotated order drift: {result_path}")
            if schedule_result.get("candidate_position_1_based") != position:
                raise RuntimeError(f"candidate position drift: {result_path}")
            if result.get("identities", {}).get("benchmark_script", {}).get("sha256") != EXPECTED_TRIAL_SCRIPT_SHA256:
                raise RuntimeError(f"benchmark script identity drift: {result_path}")
            if result.get("identities", {}).get("common_module", {}).get("sha256") != EXPECTED_COMMON_MODULE_SHA256:
                raise RuntimeError(f"common module identity drift: {result_path}")
            trial_fingerprint = result.get("identities", {}).get("runtime_fingerprint", {})
            if not isinstance(trial_fingerprint, dict) or not trial_fingerprint.get("sha256"):
                raise RuntimeError(f"runtime fingerprint identity missing: {result_path}")
            fingerprint_pair = (
                int(trial_fingerprint.get("size_bytes", -1)),
                str(trial_fingerprint.get("sha256", "")),
            )
            if runtime_fingerprint_identity is None:
                runtime_fingerprint_identity = trial_fingerprint
            elif fingerprint_pair != (
                int(runtime_fingerprint_identity["size_bytes"]),
                str(runtime_fingerprint_identity["sha256"]),
            ):
                raise RuntimeError(f"runtime fingerprint drift: {result_path}")
            trial_image_id = result.get("runtime", {}).get("container_image_id")
            if not isinstance(trial_image_id, str) or not IMAGE_ID_PATTERN.fullmatch(
                trial_image_id
            ):
                raise RuntimeError(f"full container image ID missing: {result_path}")
            if container_image_id is None:
                container_image_id = trial_image_id
            elif trial_image_id != container_image_id:
                raise RuntimeError(f"container image ID drift: {result_path}")
            process = result.get("process", {})
            if not isinstance(process.get("pid"), int) or process["pid"] <= 0:
                raise RuntimeError(f"invalid benchmark PID: {result_path}")
            started_at = parse_iso8601(process.get("started_at_utc"), "process.started_at_utc")
            ended_at = parse_iso8601(process.get("ended_at_utc"), "process.ended_at_utc")
            if ended_at < started_at or float(process.get("elapsed_seconds", -1)) <= 0:
                raise RuntimeError(f"invalid benchmark process interval: {result_path}")
            manifest_pair = (
                lineage["manifest"]["size_bytes"],
                lineage["manifest"]["sha256"],
            )
            previous = manifest_by_candidate.setdefault(candidate_id, manifest_pair)
            if manifest_pair != previous:
                raise RuntimeError(f"candidate manifest drift across trials: {candidate_id}")

            raw_arrays = {}
            recomputed = {}
            for scope in SCOPES:
                latency_path = output_dir / f"{scope}_latencies_ms.npy"
                common.identity(latency_path, result["artifacts"][f"{scope}_latencies_ms"])
                values = np.load(latency_path, allow_pickle=False).astype(np.float64, copy=False)
                if values.shape != (EXPECTED_MEASURED,):
                    raise RuntimeError(f"latency vector shape drift: {latency_path}")
                statistics = common.statistics(values)
                for key in (
                    "median_ms",
                    "p95_ms",
                    "p99_ms",
                    "mean_ms",
                    "throughput_samples_per_second",
                ):
                    if not np.isclose(
                        statistics[key], result["statistics"][scope][key], rtol=0, atol=1.0e-12
                    ):
                        raise RuntimeError(f"stored statistic drift: {result_path}/{scope}/{key}")
                raw_arrays[scope] = values
                recomputed[scope] = statistics
                csv_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "trial": trial_index,
                        "position": position,
                        "scope": scope,
                        "median_ms": statistics["median_ms"],
                        "p95_ms": statistics["p95_ms"],
                        "p99_ms": statistics["p99_ms"],
                        "mean_ms": statistics["mean_ms"],
                        "throughput_samples_per_second": statistics[
                            "throughput_samples_per_second"
                        ],
                    }
                )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "trial_index": trial_index,
                    "position": position,
                    "manifest": lineage["manifest"],
                    "fp16_backbone_blocks": lineage["fp16_backbone_blocks"],
                    "int8_backbone_blocks": lineage["int8_backbone_blocks"],
                    "result_artifact": common.identity(result_path),
                    "statistics": recomputed,
                    "raw_arrays": raw_arrays,
                    "process_max_rss_kib": result["runtime"].get("process_max_rss_kib"),
                    "process": process,
                    "container_image_id": trial_image_id,
                    "runtime_fingerprint": trial_fingerprint,
                }
            )

    candidate_summaries = {}
    all_candidate_gates = []
    for candidate_id in schedule:
        candidate_records = [row for row in records if row["candidate_id"] == candidate_id]
        if len(candidate_records) != 3:
            raise RuntimeError(f"candidate {candidate_id} does not have exactly three trials")
        scope_summaries = {}
        scope_gates = {}
        for scope in SCOPES:
            medians = [row["statistics"][scope]["median_ms"] for row in candidate_records]
            p95s = [row["statistics"][scope]["p95_ms"] for row in candidate_records]
            p99s = [row["statistics"][scope]["p99_ms"] for row in candidate_records]
            throughputs = [
                row["statistics"][scope]["throughput_samples_per_second"]
                for row in candidate_records
            ]
            pooled = np.concatenate([row["raw_arrays"][scope] for row in candidate_records])
            median_cv = common.cv_percent(medians)
            scope_summaries[scope] = {
                "median_of_trial_medians_ms": float(np.median(medians)),
                "trial_median_min_ms": float(np.min(medians)),
                "trial_median_max_ms": float(np.max(medians)),
                "trial_median_cv_percent": median_cv,
                "median_of_trial_p95_ms": float(np.median(p95s)),
                "median_of_trial_p99_ms": float(np.median(p99s)),
                "median_trial_throughput_samples_per_second": float(np.median(throughputs)),
                "pooled_300_calls_descriptive_only": common.statistics(pooled),
            }
            scope_gates[f"{scope}_trial_median_cv_le_5pct"] = (
                median_cv <= MEDIAN_CV_MAX_PERCENT
            )
        result_hashes_unique = len(
            {row["result_artifact"]["sha256"] for row in candidate_records}
        ) == 3
        gates = {
            "three_fresh_process_trials_present": len(candidate_records) == 3
            and result_hashes_unique,
            "each_scope_has_100_measurements_per_trial": all(
                row["raw_arrays"][scope].shape == (100,)
                for row in candidate_records
                for scope in SCOPES
            ),
            **scope_gates,
        }
        candidate_passed = all(gates.values())
        all_candidate_gates.append(candidate_passed)
        candidate_summaries[candidate_id] = {
            "status": "passed" if candidate_passed else "unstable",
            "manifest": candidate_records[0]["manifest"],
            "fp16_backbone_blocks": candidate_records[0]["fp16_backbone_blocks"],
            "int8_backbone_blocks": candidate_records[0]["int8_backbone_blocks"],
            "scopes": scope_summaries,
            "gates": gates,
            "speedup_vs_same_protocol_fp32": {
                scope: {
                    "fp32_median_of_trial_medians_ms": float(
                        fp32_scopes[scope]["median_of_trial_medians_ms"]
                    ),
                    "candidate_median_of_trial_medians_ms": float(
                        scope_summaries[scope]["median_of_trial_medians_ms"]
                    ),
                    "formal_median_speedup_x": float(
                        fp32_scopes[scope]["median_of_trial_medians_ms"]
                        / scope_summaries[scope]["median_of_trial_medians_ms"]
                    ),
                    "p95_ratio_x_descriptive": float(
                        fp32_scopes[scope]["median_of_trial_p95_ms"]
                        / scope_summaries[scope]["median_of_trial_p95_ms"]
                    ),
                    "p99_ratio_x_descriptive": float(
                        fp32_scopes[scope]["median_of_trial_p99_ms"]
                        / scope_summaries[scope]["median_of_trial_p99_ms"]
                    ),
                    "throughput_ratio_x_descriptive": float(
                        scope_summaries[scope][
                            "median_trial_throughput_samples_per_second"
                        ]
                        / fp32_scopes[scope][
                            "median_trial_throughput_samples_per_second"
                        ]
                    ),
                    "formal_claim_allowed": candidate_passed,
                }
                for scope in SCOPES
            },
            "claims": {
                "absolute_task_equivalence_latency_claim_allowed": candidate_passed,
                "same_protocol_fp32_speedup_claim_allowed": candidate_passed,
                "native_int8_kernel_verified": False,
                "strict_numeric_equivalence_confirmed": False,
            },
        }

    passed = all(all_candidate_gates)
    summary = {
        "schema": "phase11_mixed_precision_task_equivalence_performance_summary_v1",
        "status": "passed" if passed else "unstable",
        "identities": {
            "aggregate_script": common.identity(Path(__file__)),
            "common_module": common.identity(Path(common.__file__)),
            "expected_trial_script_sha256": EXPECTED_TRIAL_SCRIPT_SHA256,
            "runtime_fingerprint": runtime_fingerprint_identity,
            "same_protocol_fp32_summary": fp32_identity,
        },
        "runtime_lock": {
            "container_image_id": container_image_id,
            "runtime_fingerprint_identical_across_all_trials": True,
            "trial_pid_and_started_at_recorded": True,
        },
        "protocol": {
            "track": "task_equivalence_performance",
            "batch": 1,
            "warmup_per_scope_per_trial": 30,
            "measurements_per_scope_per_trial": 100,
            "fresh_process_trials_per_candidate": 3,
            "candidate_schedule": list(schedule),
            "trial_orders": {
                str(index): list(common.rotated_schedule(schedule, index))
                for index in range(1, 4)
            },
            "trial_median_cv_max_percent": MEDIAN_CV_MAX_PERCENT,
            "fp32_denominator_track": "same_protocol_fp32_reference",
            "fp32_denominator_scope_matching": list(SCOPES),
        },
        "candidate_summaries": candidate_summaries,
        "trial_artifacts": [
            {
                key: value
                for key, value in row.items()
                if key not in {"raw_arrays"}
            }
            for row in records
        ],
        "claims": {
            "all_candidate_absolute_task_equivalence_latency_claims_allowed": passed,
            "same_protocol_fp32_speedup_claim_allowed_for_all_stable_candidates": passed,
            "strict_numeric_equivalence_confirmed": False,
            "historical_m0_strict_failure_overridden": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
        "evidence_boundary": {
            "speedup_uses_separately_admitted_same_protocol_fp32_summary": True,
            "formal_speedup_is_ratio_of_median_of_three_trial_medians_in_matching_scope": True,
            "absolute_latency_does_not_prove_native_int8_kernels": True,
            "task_equivalence_does_not_rewrite_strict_numeric_failure": True,
            "process_max_rss_is_not_K100_device_memory": True,
            "K100_vram_is_measured_by_a_separate_untimed_tool": True,
        },
    }
    common.json_dump(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
