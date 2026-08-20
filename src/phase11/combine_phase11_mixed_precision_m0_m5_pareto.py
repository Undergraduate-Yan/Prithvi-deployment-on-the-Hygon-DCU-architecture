#!/usr/bin/env python3
"""Read-only merger for the frozen Phase 11 M0--M5 evidence campaigns.

The merger never edits an input artifact and refuses to overwrite an output
directory.  M0's original unstable performance campaign is immutable: an
optional confirmation campaign is recorded as a separate observation and is
never substituted into the M0 row or Pareto eligibility decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "phase11_mixed_precision_combined_m0_m5_pareto_v1"
PERFORMANCE_SCHEMA = "phase11_mixed_precision_task_equivalence_performance_summary_v1"
PARETO_SCHEMA = "phase11_mixed_precision_capacity_performance_pareto_v1"
TEST90_SCHEMA = "phase11_mixed_precision_test90_three_run_summary_v1"
FP32_PERFORMANCE_SCHEMA = "phase11_fp32_segment25_headbarrier_performance_summary_v1"
FP32_VRAM_SCHEMA = "phase11_command_wrapped_k100_vram_v1"
CANDIDATES = tuple("M%d" % index for index in range(6))
M0_M4 = CANDIDATES[:5]

# These are the immutable top-level artifacts admitted on 2026-08-18.  The
# merger deliberately fails closed if any source is regenerated or edited.
FROZEN_SOURCE_SHA256 = {
    "m0_m4_performance_summary": "2793259a97963a1ec07fa326236269396c3ec7246eb4fb1d04bb43ff6814c3e1",
    "m0_m4_pareto_summary": "c3647d3e387933a72ca3d306e6df479cca86c55ac8284720a743087e98e4bf8c",
    "m5_performance_summary": "2cd5920bb585a2a2d6253a83c667f81fbb692544cdbabe1da22a4aa8ab1e3e2c",
    "m5_pareto_summary": "f89fc7913e39bcdf2d6b7c6c1dc5a4e291021112c37ccb064f4b14e71bbc39c9",
    "fp32_performance_summary": "2616ca342f2ddf484f6f9a91e7ea18368f0ee7f8340c63cecf1a16feb4e62a87",
    "fp32_vram_result": "270be7b0f926e81731a8148846158301464b61a4377937f3329ee96b3ad92285",
}

FROZEN_TEST90_SHA256 = {
    "M0": "347b74d387d5096a6c283dabe1fe51e3a5635da5738be6212e596e782c8419ab",
    "M1": "e8c4b4c7aacd86b1decfabbd747ff1bf8b8a34da8afb18689a0e1daecfd72173",
    "M2": "25f81c492ff920c2df27a0d354d0617ab78534f71b494f35728465bd61036dfc",
    "M3": "6adeb235faa28eb2bb118f6f3c65fa04501496ed8d5745b849c1a67dde6b1c0d",
    "M4": "d40bbf475bf3dd960d75aabcd5f48d203ce754134e8be29e3b0e55cb602aebcf",
    "M5": "13205c1457bad016e2373c0bee39f722775672c355ec56689acac918f47815f5",
}

MAXIMIZE_OBJECTIVES = (
    "miou",
    "water_iou",
    "boundary_water_iou_corrected",
    "prediction_agreement_vs_fp32",
)
MINIMIZE_OBJECTIVES = (
    "combined_logical_bytes",
    "end_to_end_median_ms",
    "incremental_peak_vram_bytes",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_identity(path: Path) -> dict:
    require(path.is_file(), "input artifact is not a regular file: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot load JSON artifact %s: %s" % (path, exc)) from exc
    require(isinstance(value, dict), "JSON artifact must contain an object: %s" % path)
    return value


def same_identity(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    try:
        return int(left["size_bytes"]) == int(right["size_bytes"]) and str(
            left["sha256"]
        ).lower() == str(right["sha256"]).lower()
    except (KeyError, TypeError, ValueError):
        return False


def require_frozen_identity(label: str, identity: Mapping[str, object]) -> None:
    expected = FROZEN_SOURCE_SHA256[label]
    require(
        str(identity.get("sha256", "")).lower() == expected,
        "%s SHA-256 drift: expected %s, got %s"
        % (label, expected, identity.get("sha256")),
    )


def parse_assignments(values: Sequence[str], label: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for raw in values:
        require("=" in raw, "%s must be CANDIDATE=PATH: %r" % (label, raw))
        candidate, raw_path = raw.split("=", 1)
        require(candidate in CANDIDATES, "unknown %s candidate: %r" % (label, candidate))
        require(candidate not in result, "duplicate %s candidate: %s" % (label, candidate))
        require(bool(raw_path), "empty %s path for %s" % (label, candidate))
        result[candidate] = Path(raw_path)
    require(set(result) == set(CANDIDATES), "%s must cover exactly M0--M5" % label)
    return result


def close(left: object, right: object, label: str) -> None:
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("non-numeric value for %s" % label) from exc
    require(math.isfinite(left_value) and math.isfinite(right_value), "%s is non-finite" % label)
    require(
        math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12),
        "%s drift: %r != %r" % (label, left, right),
    )


def median_task_metrics(summary: Mapping[str, object], candidate: str) -> dict:
    runs = summary.get("runs")
    require(isinstance(runs, list) and len(runs) == 3, "%s test90 must contain three runs" % candidate)
    values: Dict[str, List[float]] = {
        key: [] for key in MAXIMIZE_OBJECTIVES + ("pixel_accuracy",)
    }
    valid_pixels = []
    for index, run in enumerate(runs, 1):
        require(isinstance(run, dict), "%s test90 run %d is not an object" % (candidate, index))
        metrics = run.get("metrics")
        require(isinstance(metrics, dict), "%s test90 run %d metrics missing" % (candidate, index))
        for key in MAXIMIZE_OBJECTIVES[:3] + ("pixel_accuracy",):
            value = float(metrics[key])
            require(math.isfinite(value), "%s %s is non-finite" % (candidate, key))
            values[key].append(value)
        agreement = float(run["prediction_agreement_vs_fp32"])
        require(math.isfinite(agreement), "%s agreement is non-finite" % candidate)
        values["prediction_agreement_vs_fp32"].append(agreement)
        valid_pixels.append(int(metrics["valid_pixels"]))
    require(set(valid_pixels) == {3927398}, "%s valid-pixel population drift" % candidate)
    result = {key: float(statistics.median(series)) for key, series in values.items()}
    result["valid_pixels"] = valid_pixels[0]
    return result


def validate_fp32(performance_path: Path, vram_path: Path) -> Tuple[dict, dict, dict, dict]:
    performance_identity = file_identity(performance_path)
    vram_identity = file_identity(vram_path)
    require_frozen_identity("fp32_performance_summary", performance_identity)
    require_frozen_identity("fp32_vram_result", vram_identity)
    performance = load_json(performance_path)
    vram = load_json(vram_path)
    require(performance.get("schema") == FP32_PERFORMANCE_SCHEMA, "FP32 performance schema drift")
    require(performance.get("status") == "passed", "FP32 performance is not passed")
    require(
        performance.get("claims", {}).get("formal_speedup_denominator_allowed") is True,
        "FP32 performance is not an admitted formal denominator",
    )
    require(vram.get("schema") == FP32_VRAM_SCHEMA, "FP32 VRAM schema drift")
    require(vram.get("status") == "passed", "FP32 VRAM is not passed")
    require(vram.get("label") == "fp32_25segment_same_protocol", "FP32 VRAM label drift")
    require(
        vram.get("claims", {}).get("isolated_k100_device_vram_peak_measured") is True,
        "FP32 VRAM measurement claim is not admitted",
    )
    require(
        vram.get("claims", {}).get("wrapped_command_latency_claim_allowed") is False,
        "FP32 wrapped VRAM run must not be used as latency evidence",
    )
    return performance, performance_identity, vram, vram_identity


def validate_campaign(
    label: str,
    performance_path: Path,
    pareto_path: Path,
    expected_candidates: Sequence[str],
    fp32_identity: Mapping[str, object],
) -> Tuple[dict, dict, dict, dict]:
    performance_identity = file_identity(performance_path)
    pareto_identity = file_identity(pareto_path)
    require_frozen_identity(label + "_performance_summary", performance_identity)
    require_frozen_identity(label + "_pareto_summary", pareto_identity)
    performance = load_json(performance_path)
    pareto = load_json(pareto_path)
    require(performance.get("schema") == PERFORMANCE_SCHEMA, "%s performance schema drift" % label)
    require(performance.get("status") in {"passed", "unstable"}, "%s performance incomplete" % label)
    require(pareto.get("schema") == PARETO_SCHEMA, "%s Pareto schema drift" % label)
    require(
        set(performance.get("candidate_summaries", {})) == set(expected_candidates),
        "%s performance candidate set drift" % label,
    )
    require(
        set(pareto.get("candidate_rows", {})) == set(expected_candidates),
        "%s Pareto candidate set drift" % label,
    )
    require(
        tuple(performance.get("protocol", {}).get("candidate_schedule", ())) == tuple(expected_candidates),
        "%s candidate schedule drift" % label,
    )
    require(
        same_identity(pareto.get("identities", {}).get("performance_summary", {}), performance_identity),
        "%s Pareto/performance identity mismatch" % label,
    )
    require(
        same_identity(performance.get("identities", {}).get("same_protocol_fp32_summary", {}), fp32_identity),
        "%s performance/FP32 identity mismatch" % label,
    )
    require(
        same_identity(
            pareto.get("identities", {}).get("same_protocol_fp32_performance_summary", {}),
            fp32_identity,
        ),
        "%s Pareto/FP32 identity mismatch" % label,
    )
    require(
        performance.get("claims", {}).get("historical_m0_strict_failure_overridden") is False,
        "%s illegally overrides historical M0 strict failure" % label,
    )
    require(
        pareto.get("claims", {}).get("historical_m0_strict_failure_overridden") is False,
        "%s Pareto illegally overrides historical M0 strict failure" % label,
    )
    return performance, performance_identity, pareto, pareto_identity


def validate_protocol_compatibility(left: Mapping[str, object], right: Mapping[str, object]) -> dict:
    keys = (
        "track",
        "batch",
        "warmup_per_scope_per_trial",
        "measurements_per_scope_per_trial",
        "fresh_process_trials_per_candidate",
        "trial_median_cv_max_percent",
        "fp32_denominator_track",
        "fp32_denominator_scope_matching",
    )
    left_protocol = left["protocol"]
    right_protocol = right["protocol"]
    for key in keys:
        require(left_protocol.get(key) == right_protocol.get(key), "campaign protocol drift: %s" % key)
    left_runtime = left["runtime_lock"]
    right_runtime = right["runtime_lock"]
    require(left_runtime == right_runtime, "M0--M4 and M5 runtime locks differ")
    require(
        same_identity(
            left.get("identities", {}).get("runtime_fingerprint", {}),
            right.get("identities", {}).get("runtime_fingerprint", {}),
        ),
        "M0--M4 and M5 runtime fingerprints differ",
    )
    return {
        "shared_protocol_fields": {key: left_protocol.get(key) for key in keys},
        "runtime_lock": left_runtime,
        "runtime_fingerprint": left.get("identities", {}).get("runtime_fingerprint"),
        "campaign_trial_orders_kept_separate": True,
    }


def validate_test90(
    candidate: str,
    path: Path,
    performance_row: Mapping[str, object],
    pareto_row: Mapping[str, object],
) -> Tuple[dict, dict, dict]:
    identity = file_identity(path)
    require(
        identity["sha256"] == FROZEN_TEST90_SHA256[candidate],
        "%s test90 SHA-256 drift" % candidate,
    )
    require(
        same_identity(pareto_row.get("identities", {}).get("test90_summary", {}), identity),
        "%s Pareto/test90 identity mismatch" % candidate,
    )
    summary = load_json(path)
    require(summary.get("schema") == TEST90_SCHEMA, "%s test90 schema drift" % candidate)
    require(summary.get("status") == "passed", "%s test90 did not pass" % candidate)
    lineage = summary.get("candidate_lineage", {})
    require(lineage.get("candidate_id") == candidate, "%s test90 candidate lineage drift" % candidate)
    require(
        same_identity(lineage.get("manifest", {}), performance_row.get("manifest", {})),
        "%s test90/performance manifest mismatch" % candidate,
    )
    require(
        same_identity(lineage.get("manifest", {}), pareto_row.get("identities", {}).get("manifest", {})),
        "%s test90/Pareto manifest mismatch" % candidate,
    )
    for block_key in ("fp16_backbone_blocks", "int8_backbone_blocks"):
        expected = list(performance_row.get(block_key, ()))
        require(list(lineage.get(block_key, ())) == expected, "%s %s drift" % (candidate, block_key))
        require(list(pareto_row.get(block_key, ())) == expected, "%s Pareto %s drift" % (candidate, block_key))
    gates = summary.get("gates", {})
    require(gates.get("exactly_three_independent_results") is True, "%s lacks three test90 runs" % candidate)
    require(gates.get("all_individual_task_gates_passed") is True, "%s task gate failed" % candidate)
    require(
        summary.get("claims", {}).get("historical_m0_strict_failure_overridden") is False,
        "%s test90 illegally overrides historical M0 strict failure" % candidate,
    )
    metrics = median_task_metrics(summary, candidate)
    for key in MAXIMIZE_OBJECTIVES + ("pixel_accuracy",):
        close(metrics[key], pareto_row[key], "%s %s Pareto/test90" % (candidate, key))
    return summary, identity, metrics


def dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    no_worse = all(float(left[key]) >= float(right[key]) for key in MAXIMIZE_OBJECTIVES)
    no_worse = no_worse and all(
        float(left[key]) <= float(right[key]) for key in MINIMIZE_OBJECTIVES
    )
    strictly_better = any(
        float(left[key]) > float(right[key]) for key in MAXIMIZE_OBJECTIVES
    ) or any(float(left[key]) < float(right[key]) for key in MINIMIZE_OBJECTIVES)
    return no_worse and strictly_better


def normalized_regrets(rows: Mapping[str, Mapping[str, object]]) -> Dict[str, float]:
    regrets: Dict[str, List[float]] = {candidate: [] for candidate in rows}
    for key in MAXIMIZE_OBJECTIVES + MINIMIZE_OBJECTIVES:
        values = [float(row[key]) for row in rows.values()]
        low, high = min(values), max(values)
        for candidate, row in rows.items():
            if high == low:
                regret = 0.0
            elif key in MINIMIZE_OBJECTIVES:
                regret = (float(row[key]) - low) / (high - low)
            else:
                regret = (high - float(row[key])) / (high - low)
            regrets[candidate].append(regret)
    return {
        candidate: float(statistics.fmean(values))
        for candidate, values in regrets.items()
    }


def confirmation_observation(path: Optional[Path], fp32_identity: Mapping[str, object], m0_manifest: Mapping[str, object]) -> dict:
    if path is None:
        return {
            "status": "not_supplied",
            "used_to_overwrite_original_m0_performance": False,
            "used_for_combined_pareto": False,
        }
    identity = file_identity(path)
    summary = load_json(path)
    require(summary.get("schema") == PERFORMANCE_SCHEMA, "M0 confirmation schema drift")
    require(set(summary.get("candidate_summaries", {})) == {"M0"}, "M0 confirmation candidate set drift")
    require(
        tuple(summary.get("protocol", {}).get("candidate_schedule", ())) == ("M0",),
        "M0 confirmation schedule drift",
    )
    require(
        same_identity(summary.get("identities", {}).get("same_protocol_fp32_summary", {}), fp32_identity),
        "M0 confirmation/FP32 identity mismatch",
    )
    row = summary["candidate_summaries"]["M0"]
    require(same_identity(row.get("manifest", {}), m0_manifest), "M0 confirmation manifest mismatch")
    scopes = row.get("scopes", {})
    result = {
        "status": "supplied_as_append_only_observation",
        "identity": identity,
        "campaign_status": summary.get("status"),
        "candidate_status": row.get("status"),
        "model_only": scopes.get("model_only"),
        "end_to_end_logits": scopes.get("end_to_end_logits"),
        "gates": row.get("gates"),
        "claims": row.get("claims"),
        "used_to_overwrite_original_m0_performance": False,
        "used_for_combined_pareto": False,
        "reason": (
            "The frozen original M0 campaign remains status=unstable.  A later confirmation "
            "is an append-only observation and cannot rewrite the original row or eligibility."
        ),
    }
    return result


def build_row(
    candidate: str,
    performance_row: Mapping[str, object],
    pareto_row: Mapping[str, object],
    task_metrics: Mapping[str, object],
    test90_identity: Mapping[str, object],
    source_campaign: str,
) -> dict:
    for key in (
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
        "peak_vram_used_bytes",
        "baseline_vram_used_bytes",
        "incremental_peak_vram_bytes",
    ):
        require(key in pareto_row, "%s Pareto row missing %s" % (candidate, key))
    require(
        int(pareto_row["combined_logical_bytes"])
        == int(pareto_row["onnx_logical_bytes_25_segments"])
        + int(pareto_row["mxr_logical_bytes_25_segments"]),
        "%s combined ONNX+MXR size is inconsistent" % candidate,
    )
    perf_scopes = performance_row["scopes"]
    close(
        pareto_row["end_to_end_median_ms"],
        perf_scopes["end_to_end_logits"]["median_of_trial_medians_ms"],
        "%s end-to-end median Pareto/performance" % candidate,
    )
    close(
        pareto_row["model_only_median_ms"],
        perf_scopes["model_only"]["median_of_trial_medians_ms"],
        "%s model-only median Pareto/performance" % candidate,
    )
    original_status = str(performance_row.get("status"))
    eligible = original_status == "passed"
    eligible = eligible and performance_row.get("gates", {}).get(
        "model_only_trial_median_cv_le_5pct"
    ) is True
    eligible = eligible and performance_row.get("gates", {}).get(
        "end_to_end_logits_trial_median_cv_le_5pct"
    ) is True
    require(
        bool(pareto_row.get("pareto_eligible_after_cv_gate")) == eligible,
        "%s old Pareto eligibility does not match the original performance gates" % candidate,
    )
    row = {
        "candidate_id": candidate,
        "source_campaign": source_campaign,
        "fp16_backbone_blocks": list(performance_row["fp16_backbone_blocks"]),
        "int8_backbone_blocks": list(performance_row["int8_backbone_blocks"]),
        **task_metrics,
        "onnx_logical_bytes_25_segments": int(pareto_row["onnx_logical_bytes_25_segments"]),
        "mxr_logical_bytes_25_segments": int(pareto_row["mxr_logical_bytes_25_segments"]),
        "combined_logical_bytes": int(pareto_row["combined_logical_bytes"]),
        "capacity_scope": pareto_row.get("capacity_scope"),
        "end_to_end_median_ms": float(pareto_row["end_to_end_median_ms"]),
        "end_to_end_p95_ms": float(pareto_row["end_to_end_p95_ms"]),
        "end_to_end_p99_ms": float(pareto_row["end_to_end_p99_ms"]),
        "end_to_end_throughput_samples_per_second": float(
            pareto_row["end_to_end_throughput_samples_per_second"]
        ),
        "end_to_end_trial_median_cv_percent": float(
            pareto_row["end_to_end_trial_median_cv_percent"]
        ),
        "model_only_median_ms": float(pareto_row["model_only_median_ms"]),
        "model_only_trial_median_cv_percent": float(
            pareto_row["model_only_trial_median_cv_percent"]
        ),
        "formal_model_only_speedup_vs_fp32_x": float(
            pareto_row["formal_model_only_speedup_vs_fp32_x"]
        ),
        "formal_end_to_end_speedup_vs_fp32_x": float(
            pareto_row["formal_end_to_end_speedup_vs_fp32_x"]
        ),
        "peak_vram_used_bytes": int(pareto_row["peak_vram_used_bytes"]),
        "baseline_vram_used_bytes": int(pareto_row["baseline_vram_used_bytes"]),
        "incremental_peak_vram_bytes": int(pareto_row["incremental_peak_vram_bytes"]),
        "original_performance_status": original_status,
        "original_performance_gates": performance_row.get("gates"),
        "pareto_eligible_from_original_campaign": eligible,
        "identities": {
            "manifest": performance_row["manifest"],
            "test90_summary": dict(test90_identity),
            "vram_result": pareto_row.get("identities", {}).get("vram_result"),
        },
        "strict_numeric_equivalence_confirmed": False,
        "native_int8_kernel_verified": False,
        "deployment_ready": False,
    }
    if candidate == "M0":
        require(original_status == "unstable", "frozen original M0 performance must remain unstable")
        require(not eligible, "frozen original M0 must remain Pareto-ineligible")
        row["immutable_m0_policy"] = {
            "original_run_status": "unstable",
            "original_model_only_cv_gate_passed": False,
            "confirmation_may_overwrite": False,
            "reason": "original model-only trial-median CV exceeded 5%",
        }
    else:
        require(original_status == "passed", "%s frozen performance must remain passed" % candidate)
        require(eligible, "%s must pass both original CV gates" % candidate)
    return row


def write_outputs(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "combined_m0_m5_pareto.json"
    csv_path = output_dir / "combined_m0_m5_pareto.csv"
    with json_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    csv_fields = [
        "candidate_id",
        "source_campaign",
        "original_performance_status",
        "pareto_eligible_from_original_campaign",
        "pareto_nondominated_combined",
        "fp16_backbone_blocks",
        "int8_backbone_blocks",
        "miou",
        "water_iou",
        "boundary_water_iou_corrected",
        "pixel_accuracy",
        "prediction_agreement_vs_fp32",
        "valid_pixels",
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
        "strict_numeric_equivalence_confirmed",
        "native_int8_kernel_verified",
        "deployment_ready",
    ]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for candidate in CANDIDATES:
            row = dict(summary["candidate_rows"][candidate])
            row["fp16_backbone_blocks"] = json.dumps(row["fp16_backbone_blocks"])
            row["int8_backbone_blocks"] = json.dumps(row["int8_backbone_blocks"])
            writer.writerow({key: row.get(key) for key in csv_fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge frozen M0--M4 and M5 campaigns without modifying sources; "
            "M0 confirmation is append-only and never replaces the unstable original run."
        )
    )
    parser.add_argument("--m0-m4-performance-summary", type=Path, required=True)
    parser.add_argument("--m0-m4-pareto-summary", type=Path, required=True)
    parser.add_argument("--m5-performance-summary", type=Path, required=True)
    parser.add_argument("--m5-pareto-summary", type=Path, required=True)
    parser.add_argument(
        "--test90-summary",
        action="append",
        required=True,
        metavar="CANDIDATE=PATH",
        help="repeat exactly once for each of M0 through M5",
    )
    parser.add_argument("--fp32-performance-summary", type=Path, required=True)
    parser.add_argument("--fp32-vram-result", type=Path, required=True)
    parser.add_argument(
        "--m0-confirmation-performance-summary",
        type=Path,
        help="optional append-only observation; never replaces the frozen M0 run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    test90_paths = parse_assignments(args.test90_summary, "test90 summary")
    fp32, fp32_identity, fp32_vram, fp32_vram_identity = validate_fp32(
        args.fp32_performance_summary, args.fp32_vram_result
    )
    m0_m4_perf, m0_m4_perf_identity, m0_m4_pareto, m0_m4_pareto_identity = validate_campaign(
        "m0_m4",
        args.m0_m4_performance_summary,
        args.m0_m4_pareto_summary,
        M0_M4,
        fp32_identity,
    )
    m5_perf, m5_perf_identity, m5_pareto, m5_pareto_identity = validate_campaign(
        "m5",
        args.m5_performance_summary,
        args.m5_pareto_summary,
        ("M5",),
        fp32_identity,
    )
    protocol_lock = validate_protocol_compatibility(m0_m4_perf, m5_perf)

    rows = {}
    test90_identities = {}
    for candidate in CANDIDATES:
        performance = m0_m4_perf if candidate in M0_M4 else m5_perf
        pareto = m0_m4_pareto if candidate in M0_M4 else m5_pareto
        performance_row = performance["candidate_summaries"][candidate]
        pareto_row = pareto["candidate_rows"][candidate]
        _, test90_identity, task_metrics = validate_test90(
            candidate, test90_paths[candidate], performance_row, pareto_row
        )
        test90_identities[candidate] = test90_identity
        rows[candidate] = build_row(
            candidate,
            performance_row,
            pareto_row,
            task_metrics,
            test90_identity,
            "M0-M4_original_v2" if candidate in M0_M4 else "M5_original_v1",
        )

    confirmation = confirmation_observation(
        args.m0_confirmation_performance_summary,
        fp32_identity,
        rows["M0"]["identities"]["manifest"],
    )
    rows["M0"]["append_only_confirmation_observation"] = confirmation

    eligible = tuple(
        candidate
        for candidate in CANDIDATES
        if rows[candidate]["pareto_eligible_from_original_campaign"]
    )
    require(eligible == ("M1", "M2", "M3", "M4", "M5"), "frozen eligible set drift")
    eligible_rows = {candidate: rows[candidate] for candidate in eligible}
    frontier = [
        candidate
        for candidate, row in eligible_rows.items()
        if not any(
            other != candidate and dominates(other_row, row)
            for other, other_row in eligible_rows.items()
        )
    ]
    regrets = normalized_regrets(eligible_rows)
    for candidate in CANDIDATES:
        rows[candidate]["balanced_normalized_mean_regret"] = regrets.get(candidate)
        rows[candidate]["pareto_nondominated_combined"] = candidate in frontier

    def candidate_index(candidate: str) -> int:
        return int(candidate[1:])

    roles = {
        "task_accuracy_lexicographic_best": max(
            eligible,
            key=lambda candidate: tuple(float(rows[candidate][key]) for key in MAXIMIZE_OBJECTIVES)
            + (-candidate_index(candidate),),
        ),
        "onnx_plus_mxr_smallest": min(
            eligible, key=lambda candidate: (rows[candidate]["combined_logical_bytes"], candidate_index(candidate))
        ),
        "end_to_end_latency_lowest": min(
            eligible, key=lambda candidate: (rows[candidate]["end_to_end_median_ms"], candidate_index(candidate))
        ),
        "incremental_vram_lowest": min(
            eligible, key=lambda candidate: (rows[candidate]["incremental_peak_vram_bytes"], candidate_index(candidate))
        ),
        "balanced_pareto": min(
            frontier, key=lambda candidate: (regrets[candidate], candidate_index(candidate))
        ),
    }

    summary = {
        "schema": SCHEMA,
        "status": "completed_with_frozen_m0_original_unstable_excluded",
        "generated_by": file_identity(Path(__file__)),
        "source_artifacts": {
            "m0_m4_performance_summary": m0_m4_perf_identity,
            "m0_m4_pareto_summary": m0_m4_pareto_identity,
            "m5_performance_summary": m5_perf_identity,
            "m5_pareto_summary": m5_pareto_identity,
            "test90_summaries": test90_identities,
            "fp32_performance_summary": fp32_identity,
            "fp32_vram_result": fp32_vram_identity,
            "m0_confirmation_performance_summary": confirmation.get("identity"),
        },
        "identity_policy": {
            "all_required_top_level_sources_sha256_frozen": True,
            "all_six_test90_sources_sha256_frozen": True,
            "cross_document_manifest_test90_performance_pareto_lineage_verified": True,
            "inputs_modified": False,
            "output_overwrite_allowed": False,
        },
        "protocol_lock": protocol_lock,
        "objectives": {
            "maximize_for_dominance": list(MAXIMIZE_OBJECTIVES),
            "minimize_for_dominance": list(MINIMIZE_OBJECTIVES),
            "capacity_definition": "logical bytes of 25 ONNX segments plus 25 MXR caches",
            "latency_definition": "end-to-end input-to-logits median of trial medians",
            "vram_definition": "isolated incremental peak device VRAM over baseline",
            "balanced_score": (
                "equal-weight mean of min-max normalized regret over four task metrics, "
                "ONNX+MXR bytes, end-to-end median latency, and incremental peak VRAM"
            ),
        },
        "same_protocol_fp32_baseline": {
            "performance": {
                "status": fp32["status"],
                "model_only_median_ms": fp32["scope_summaries"]["model_only"][
                    "median_of_trial_medians_ms"
                ],
                "end_to_end_median_ms": fp32["scope_summaries"]["end_to_end_logits"][
                    "median_of_trial_medians_ms"
                ],
                "identity": fp32_identity,
            },
            "vram": {
                "status": fp32_vram["status"],
                "measurements": fp32_vram["measurements"],
                "identity": fp32_vram_identity,
                "wrapped_command_latency_used": False,
            },
        },
        "candidate_rows": rows,
        "pareto_eligible_candidates": list(eligible),
        "pareto_excluded_candidates": {
            "M0": "frozen original model-only trial-median CV exceeded 5%; append-only confirmation cannot overwrite it"
        },
        "pareto_frontier": frontier,
        "role_selections": roles,
        "claims": {
            "m0_m5_task_capacity_end_to_end_vram_comparison_complete": True,
            "m0_original_run_remains_unstable": True,
            "m0_confirmation_overrode_original": False,
            "formal_same_protocol_fp32_denominator_available": True,
            "strict_numeric_equivalence_confirmed": False,
            "historical_m0_strict_failure_overridden": False,
            "native_int8_kernel_verified": False,
            "fp16_full_replaced_as_deployment_recommendation": False,
            "deployment_ready": False,
        },
        "evidence_boundary": (
            "This merge proves only the joined frozen task, logical-capacity, same-protocol "
            "performance, and isolated-VRAM comparison.  It does not prove native INT8 "
            "kernels, strict logits equivalence, FP16-full replacement, or deployment readiness."
        ),
    }
    write_outputs(args.output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
