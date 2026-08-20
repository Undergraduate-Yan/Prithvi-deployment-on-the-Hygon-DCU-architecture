#!/usr/bin/env python3
"""Join task, capacity, latency, and isolated VRAM evidence into a Pareto summary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import phase11_mixed_precision_common as common


SCHEMA = "phase11_mixed_precision_capacity_performance_pareto_v1"
EXPECTED_PERFORMANCE_AGGREGATE_SCRIPT_SHA256 = "8ef4d2ecf23052d5e49398311a374a14a3468aa7b2c32baf66cc74902cc0c80a"
EXPECTED_TEST90_AGGREGATE_SCRIPT_SHA256 = "717aa6c5e10b889c082acb4626105ad708c9006acf70380793f0b93c71fbd159"
EXPECTED_VRAM_MEASUREMENT_SCRIPT_SHA256 = "4a554b78a67055e869d3663e87b80ab27ae8ffead209f18531c41075a3bf8a50"
EXPECTED_COMMON_MODULE_SHA256 = "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8"
EXPECTED_FP32_VRAM_WRAPPER_SHA256 = "402c43803e80defaa32c110e28b1ce9c3cd360889b55d3d1c2c3abca72b93428"


def assignments(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise RuntimeError(f"{label} assignment must be CANDIDATE=PATH: {raw!r}")
        candidate, path = raw.split("=", 1)
        if candidate in result or not candidate:
            raise RuntimeError(f"duplicate/empty {label} candidate: {candidate!r}")
        result[candidate] = Path(path)
    return result


def same_identity(left: dict, right: dict) -> bool:
    return (int(left.get("size_bytes", -1)), str(left.get("sha256", ""))) == (
        int(right.get("size_bytes", -2)),
        str(right.get("sha256", "")),
    )


def median_task_metrics(summary: dict) -> dict[str, float]:
    keys = ("miou", "water_iou", "boundary_water_iou_corrected", "pixel_accuracy")
    rows = summary.get("runs", [])
    if len(rows) != 3:
        raise RuntimeError("test90 summary must contain exactly three runs")
    result = {
        key: float(np.median([float(row["metrics"][key]) for row in rows]))
        for key in keys
    }
    result["prediction_agreement_vs_fp32"] = float(
        np.median([float(row["prediction_agreement_vs_fp32"]) for row in rows])
    )
    return result


def capacity(rows: list[dict]) -> dict:
    model_identities = [row["verified_model_identity"] for row in rows]
    cache_identities = [row["verified_cache_identity"] for row in rows]
    onnx_logical = sum(int(item["size_bytes"]) for item in model_identities)
    mxr_logical = sum(int(item["size_bytes"]) for item in cache_identities)
    unique_models = {
        (int(item["size_bytes"]), str(item["sha256"])) for item in model_identities
    }
    unique_caches = {
        (int(item["size_bytes"]), str(item["sha256"])) for item in cache_identities
    }
    onnx_unique = sum(item[0] for item in unique_models)
    mxr_unique = sum(item[0] for item in unique_caches)
    return {
        "onnx_logical_bytes_25_segments": onnx_logical,
        "mxr_logical_bytes_25_segments": mxr_logical,
        "combined_logical_bytes": onnx_logical + mxr_logical,
        "onnx_unique_payload_bytes_within_candidate": onnx_unique,
        "mxr_unique_payload_bytes_within_candidate": mxr_unique,
        "combined_unique_payload_bytes_within_candidate": onnx_unique + mxr_unique,
        "onnx_unique_payload_count": len(unique_models),
        "mxr_unique_payload_count": len(unique_caches),
        "capacity_scope": (
            "logical deployed payload for one candidate; filesystem allocation, container image, "
            "Python/runtime dependencies, and cross-candidate hard-link savings excluded"
        ),
    }


def dominates(left: dict, right: dict) -> bool:
    maximize = (
        "miou",
        "water_iou",
        "boundary_water_iou_corrected",
        "prediction_agreement_vs_fp32",
    )
    minimize = ("combined_logical_bytes", "end_to_end_median_ms")
    no_worse = all(left[key] >= right[key] for key in maximize) and all(
        left[key] <= right[key] for key in minimize
    )
    strictly_better = any(left[key] > right[key] for key in maximize) or any(
        left[key] < right[key] for key in minimize
    )
    return no_worse and strictly_better


def normalized_regrets(rows: dict[str, dict]) -> dict[str, float]:
    directions = {
        "miou": "maximize",
        "water_iou": "maximize",
        "boundary_water_iou_corrected": "maximize",
        "prediction_agreement_vs_fp32": "maximize",
        "combined_logical_bytes": "minimize",
        "end_to_end_median_ms": "minimize",
        "incremental_peak_vram_bytes": "minimize",
    }
    per_candidate = {candidate: [] for candidate in rows}
    for key, direction in directions.items():
        values = [float(row[key]) for row in rows.values()]
        low, high = min(values), max(values)
        for candidate, row in rows.items():
            if high == low:
                regret = 0.0
            elif direction == "minimize":
                regret = (float(row[key]) - low) / (high - low)
            else:
                regret = (high - float(row[key])) / (high - low)
            per_candidate[candidate].append(regret)
    return {
        candidate: float(np.mean(values)) for candidate, values in per_candidate.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--performance-summary", type=Path, required=True)
    parser.add_argument("--fp32-performance-summary", type=Path, required=True)
    parser.add_argument("--fp32-vram-result", type=Path)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--test90-summary", action="append", required=True)
    parser.add_argument("--vram-result", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = assignments(args.manifest, "manifest")
    test90_summaries = assignments(args.test90_summary, "test90 summary")
    vram_results = assignments(args.vram_result, "VRAM result")

    performance_identity = common.identity(args.performance_summary)
    performance = common.load_json(args.performance_summary)
    if performance.get("schema") != "phase11_mixed_precision_task_equivalence_performance_summary_v1":
        raise RuntimeError("performance summary schema drift")
    if performance.get("status") not in {"passed", "unstable"}:
        raise RuntimeError("performance campaign did not complete")
    if performance.get("identities", {}).get("aggregate_script", {}).get("sha256") != (
        EXPECTED_PERFORMANCE_AGGREGATE_SCRIPT_SHA256
    ):
        raise RuntimeError("performance aggregate script identity drift")
    if performance.get("identities", {}).get("common_module", {}).get("sha256") != (
        EXPECTED_COMMON_MODULE_SHA256
    ):
        raise RuntimeError("performance common module identity drift")
    fp32_performance_identity = common.identity(args.fp32_performance_summary)
    if not same_identity(
        fp32_performance_identity,
        performance.get("identities", {}).get("same_protocol_fp32_summary", {}),
    ):
        raise RuntimeError("Pareto/performance FP32 summary identity drift")
    fp32_performance = common.load_json(args.fp32_performance_summary)
    if fp32_performance.get("status") != "passed" or not fp32_performance.get(
        "claims", {}
    ).get("formal_speedup_denominator_allowed"):
        raise RuntimeError("FP32 summary is not an admitted formal denominator")
    candidates = tuple(performance.get("protocol", {}).get("candidate_schedule", ()))
    if not candidates:
        raise RuntimeError("performance candidate schedule is empty")
    expected_set = set(candidates)
    for label, mapping in (
        ("manifest", manifests),
        ("test90 summary", test90_summaries),
        ("VRAM result", vram_results),
    ):
        if set(mapping) != expected_set:
            raise RuntimeError(f"{label} candidate set does not match performance schedule")

    joined = {}
    artifact_rows = []
    for candidate in candidates:
        manifest, manifest_identity, rows = common.load_candidate(
            args.bundle, manifests[candidate]
        )
        if manifest["candidate_id"] != candidate:
            raise RuntimeError(f"manifest candidate mismatch: {candidate}")
        perf_row = performance["candidate_summaries"][candidate]
        if not same_identity(perf_row["manifest"], manifest_identity):
            raise RuntimeError(f"performance/manifest lineage mismatch: {candidate}")

        test90_identity = common.identity(test90_summaries[candidate])
        test90 = common.load_json(test90_summaries[candidate])
        if test90.get("schema") != "phase11_mixed_precision_test90_three_run_summary_v1":
            raise RuntimeError(f"test90 summary schema drift: {candidate}")
        if test90.get("status") != "passed":
            raise RuntimeError(f"test90 summary did not pass: {candidate}")
        if test90.get("identities", {}).get("aggregate_script", {}).get("sha256") != (
            EXPECTED_TEST90_AGGREGATE_SCRIPT_SHA256
        ):
            raise RuntimeError(f"test90 aggregate script identity drift: {candidate}")
        common.validate_result_candidate(test90, manifest_identity, candidate)

        vram_identity = common.identity(vram_results[candidate])
        vram = common.load_json(vram_results[candidate])
        if vram.get("schema") != "phase11_mixed_precision_k100_vram_v1":
            raise RuntimeError(f"VRAM result schema drift: {candidate}")
        if vram.get("status") != "passed" or not vram.get("claims", {}).get(
            "isolated_k100_device_vram_peak_measured"
        ):
            raise RuntimeError(f"VRAM evidence did not pass: {candidate}")
        if vram.get("identities", {}).get("measurement_script", {}).get("sha256") != (
            EXPECTED_VRAM_MEASUREMENT_SCRIPT_SHA256
        ):
            raise RuntimeError(f"VRAM measurement script identity drift: {candidate}")
        if vram.get("identities", {}).get("common_module", {}).get("sha256") != (
            EXPECTED_COMMON_MODULE_SHA256
        ):
            raise RuntimeError(f"VRAM common module identity drift: {candidate}")
        common.validate_result_candidate(vram, manifest_identity, candidate)
        if vram.get("runtime", {}).get("container_image_id") != performance.get(
            "runtime_lock", {}
        ).get("container_image_id"):
            raise RuntimeError(f"VRAM/performance image drift: {candidate}")
        if not same_identity(
            vram.get("identities", {}).get("runtime_fingerprint", {}),
            performance.get("identities", {}).get("runtime_fingerprint", {}),
        ):
            raise RuntimeError(f"VRAM/performance runtime fingerprint drift: {candidate}")

        task = median_task_metrics(test90)
        storage = capacity(rows)
        end_to_end = perf_row["scopes"]["end_to_end_logits"]
        model_only = perf_row["scopes"]["model_only"]
        joined[candidate] = {
            "candidate_id": candidate,
            "performance_status": perf_row["status"],
            "fp16_backbone_blocks": manifest["fp16_backbone_blocks"],
            "int8_backbone_blocks": manifest["int8_backbone_blocks"],
            **task,
            **storage,
            "end_to_end_median_ms": float(end_to_end["median_of_trial_medians_ms"]),
            "end_to_end_p95_ms": float(end_to_end["median_of_trial_p95_ms"]),
            "end_to_end_p99_ms": float(end_to_end["median_of_trial_p99_ms"]),
            "end_to_end_throughput_samples_per_second": float(
                end_to_end["median_trial_throughput_samples_per_second"]
            ),
            "end_to_end_trial_median_cv_percent": float(
                end_to_end["trial_median_cv_percent"]
            ),
            "model_only_median_ms": float(model_only["median_of_trial_medians_ms"]),
            "model_only_trial_median_cv_percent": float(
                model_only["trial_median_cv_percent"]
            ),
            "formal_model_only_speedup_vs_fp32_x": float(
                perf_row["speedup_vs_same_protocol_fp32"]["model_only"][
                    "formal_median_speedup_x"
                ]
            ),
            "formal_end_to_end_speedup_vs_fp32_x": float(
                perf_row["speedup_vs_same_protocol_fp32"]["end_to_end_logits"][
                    "formal_median_speedup_x"
                ]
            ),
            "peak_vram_used_bytes": int(vram["measurements"]["peak_vram_used_bytes"]),
            "baseline_vram_used_bytes": int(
                vram["measurements"]["baseline_vram_used_bytes"]
            ),
            "incremental_peak_vram_bytes": int(
                vram["measurements"]["incremental_peak_over_baseline_bytes"]
            ),
            "identities": {
                "manifest": manifest_identity,
                "test90_summary": test90_identity,
                "vram_result": vram_identity,
            },
        }
        artifact_rows.append(
            {
                "candidate_id": candidate,
                "manifest": manifest_identity,
                "test90_summary": test90_identity,
                "vram_result": vram_identity,
            }
        )

    fp32_vram = {
        "status": "unavailable_not_supplied",
        "reason": (
            "pass --fp32-vram-result from the separate same-scope wrapper run; "
            "FP32 process RSS and timed-run telemetry are not substituted"
        ),
    }
    if args.fp32_vram_result is not None:
        fp32_vram_identity = common.identity(args.fp32_vram_result)
        fp32_vram_result = common.load_json(args.fp32_vram_result)
        if fp32_vram_result.get("schema") != "phase11_command_wrapped_k100_vram_v1":
            raise RuntimeError("FP32 VRAM result schema drift")
        if fp32_vram_result.get("status") != "passed" or not fp32_vram_result.get(
            "claims", {}
        ).get("isolated_k100_device_vram_peak_measured"):
            raise RuntimeError("FP32 VRAM result did not pass")
        if fp32_vram_result.get("label") != "fp32_25segment_same_protocol":
            raise RuntimeError("FP32 VRAM label drift")
        if fp32_vram_result.get("identities", {}).get("measurement_script", {}).get(
            "sha256"
        ) != EXPECTED_FP32_VRAM_WRAPPER_SHA256:
            raise RuntimeError("FP32 VRAM wrapper identity drift")
        fp32_vram = {
            "status": "available",
            "identity": fp32_vram_identity,
            "runtime": fp32_vram_result["runtime"],
            "protocol": fp32_vram_result["protocol"],
            "measurements": fp32_vram_result["measurements"],
            "comparison_scope_matches_mixed": (
                "25 resident sessions exercised in model-only and end-to-end scopes; "
                "30 warmups plus 100 exercises per scope; 10 ms device-wide sysfs sampling"
            ),
        }

    eligible_candidates = tuple(
        candidate
        for candidate in candidates
        if joined[candidate]["performance_status"] == "passed"
    )
    if not eligible_candidates:
        raise RuntimeError("no candidate passed the trial-median CV gate")
    eligible_rows = {candidate: joined[candidate] for candidate in eligible_candidates}
    pareto_frontier = [
        candidate
        for candidate, row in eligible_rows.items()
        if not any(
            other != candidate and dominates(other_row, row)
            for other, other_row in eligible_rows.items()
        )
    ]
    regrets = normalized_regrets(eligible_rows)
    for candidate in candidates:
        joined[candidate]["balanced_normalized_mean_regret"] = regrets.get(candidate)
        joined[candidate]["pareto_eligible_after_cv_gate"] = candidate in eligible_candidates
        joined[candidate]["pareto_nondominated"] = candidate in pareto_frontier

    accuracy_best = max(
        eligible_candidates,
        key=lambda candidate: (
            joined[candidate]["miou"],
            joined[candidate]["water_iou"],
            joined[candidate]["boundary_water_iou_corrected"],
            joined[candidate]["prediction_agreement_vs_fp32"],
            -int(candidate[1:]),
        ),
    )
    file_smallest = min(
        eligible_candidates,
        key=lambda candidate: (joined[candidate]["combined_logical_bytes"], candidate),
    )
    latency_lowest = min(
        eligible_candidates,
        key=lambda candidate: (joined[candidate]["end_to_end_median_ms"], candidate),
    )
    balanced = min(
        pareto_frontier,
        key=lambda candidate: (regrets[candidate], candidate),
    )
    roles = {
        "accuracy_best": accuracy_best,
        "file_smallest": file_smallest,
        "latency_lowest": latency_lowest,
        "balanced_pareto": balanced,
    }
    summary = {
        "schema": SCHEMA,
        "status": (
            "completed"
            if len(eligible_candidates) == len(candidates)
            else "completed_with_unstable_candidates_excluded"
        ),
        "identities": {
            "summary_script": common.identity(Path(__file__)),
            "common_module": common.identity(Path(common.__file__)),
            "performance_summary": performance_identity,
            "same_protocol_fp32_performance_summary": fp32_performance_identity,
        },
        "runtime_lock": performance["runtime_lock"],
        "objectives": {
            "maximize": [
                "median miou",
                "median water_iou",
                "median corrected boundary_water_iou",
                "median prediction_agreement_vs_fp32",
            ],
            "minimize": ["combined logical ONNX+MXR bytes", "end-to-end median latency"],
            "reported_but_not_used_for_dominance": ["incremental peak device VRAM"],
            "balanced_score": (
                "equal-weight mean of min-max normalized regret over four task metrics, "
                "combined bytes, end-to-end latency, and incremental VRAM; lower is better"
            ),
        },
        "candidate_rows": joined,
        "same_protocol_fp32_baseline": {
            "performance": {
                "model_only_median_ms": fp32_performance["scope_summaries"]["model_only"][
                    "median_of_trial_medians_ms"
                ],
                "end_to_end_median_ms": fp32_performance["scope_summaries"][
                    "end_to_end_logits"
                ]["median_of_trial_medians_ms"],
                "identity": fp32_performance_identity,
            },
            "vram": fp32_vram,
        },
        "pareto_eligible_candidates": list(eligible_candidates),
        "pareto_frontier": pareto_frontier,
        "role_selections": roles,
        "next_formal_candidate": balanced,
        "recommendation": {
            "paper_candidate_for_kernel_and_deployment_validation": balanced,
            "replace_fp16_full_now": False,
            "reason": (
                "Pareto selection completes only task/capacity/performance/VRAM comparison. "
                "I8II kernel proof, FP16-full size/CLI comparison, and deployment stability "
                "remain mandatory before changing the deployment recommendation."
            ),
        },
        "artifacts": artifact_rows,
        "claims": {
            "mixed_candidate_pareto_selection_complete": True,
            "formal_same_protocol_fp32_speedups_computed": True,
            "same_protocol_fp32_vram_available": fp32_vram["status"] == "available",
            "native_int8_kernel_verified": False,
            "strict_numeric_equivalence_confirmed": False,
            "historical_m0_strict_failure_overridden": False,
            "fp16_full_replaced_as_deployment_recommendation": False,
            "deployment_ready": False,
        },
    }
    common.json_dump(args.output_dir / "pareto_summary.json", summary)
    csv_fields = [
        "candidate_id",
        "performance_status",
        "fp16_backbone_blocks",
        "int8_backbone_blocks",
        "miou",
        "water_iou",
        "boundary_water_iou_corrected",
        "prediction_agreement_vs_fp32",
        "onnx_logical_bytes_25_segments",
        "mxr_logical_bytes_25_segments",
        "combined_logical_bytes",
        "end_to_end_median_ms",
        "end_to_end_p95_ms",
        "end_to_end_p99_ms",
        "end_to_end_throughput_samples_per_second",
        "end_to_end_trial_median_cv_percent",
        "model_only_median_ms",
        "model_only_trial_median_cv_percent",
        "formal_model_only_speedup_vs_fp32_x",
        "formal_end_to_end_speedup_vs_fp32_x",
        "peak_vram_used_bytes",
        "incremental_peak_vram_bytes",
        "balanced_normalized_mean_regret",
        "pareto_eligible_after_cv_gate",
        "pareto_nondominated",
    ]
    with (args.output_dir / "pareto_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for candidate in candidates:
            row = joined[candidate]
            writer.writerow({key: row[key] for key in csv_fields})
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
