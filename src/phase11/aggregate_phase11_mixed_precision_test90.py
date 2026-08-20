#!/usr/bin/env python3
"""Aggregate three independent fixed-90 admissions for one mixed candidate."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import phase11_mixed_precision_common as common


MIOU_WATER_SPAN_MAX_PP = 0.01
BOUNDARY_SPAN_MAX_PP = 0.02
EXPECTED_TEST90_SCRIPT_SHA256 = "8bc54ffe04719cce53d22648b0ffc2fd1e8daaac8e2e4acb61c12910c61d4091"
EXPECTED_COMMON_MODULE_SHA256 = "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 3:
        raise RuntimeError("exactly three independently launched test90 run directories are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest, manifest_identity, _ = common.load_candidate(args.bundle, args.manifest)

    runs = []
    for trial_index, run_dir in enumerate(args.run_dir, start=1):
        run_dir = run_dir.resolve(strict=True)
        result_path = run_dir / "result.json"
        arrays_path = run_dir / "predictions_and_targets.npz"
        csv_path = run_dir / "per_sample_metrics.csv"
        result = common.load_json(result_path)
        if result.get("schema") != "phase11_mixed_precision_test90_v1":
            raise RuntimeError(f"run {trial_index} schema drift")
        if result.get("status") != "diagnostic_completed":
            raise RuntimeError(f"run {trial_index} did not complete diagnostically")
        if result.get("identities", {}).get("evaluation_script", {}).get("sha256") != EXPECTED_TEST90_SCRIPT_SHA256:
            raise RuntimeError(f"run {trial_index} test90 evaluator identity drift")
        if result.get("identities", {}).get("common_module", {}).get("sha256") != EXPECTED_COMMON_MODULE_SHA256:
            raise RuntimeError(f"run {trial_index} common module identity drift")
        common.validate_result_candidate(result, manifest_identity, manifest["candidate_id"])
        if not result.get("task_equivalence_admission_passed"):
            raise RuntimeError(f"run {trial_index} failed its task gate")
        if result.get("claims", {}).get("historical_m0_strict_failure_overridden"):
            raise RuntimeError("historical M0 strict failure was rewritten")
        common.identity(arrays_path, result["artifacts"]["predictions_and_targets"])
        common.identity(csv_path, result["artifacts"]["per_sample_metrics"])
        with np.load(arrays_path, allow_pickle=False) as pack:
            predictions = np.ascontiguousarray(pack["predictions"], dtype=np.uint8)
            targets = np.ascontiguousarray(pack["targets"], dtype=np.int64)
            sample_ids = [str(item) for item in pack["sample_ids"].tolist()]
        if predictions.shape != (90, 224, 224) or targets.shape != (90, 224, 224):
            raise RuntimeError(f"run {trial_index} tensor shape drift")
        runs.append(
            {
                "trial_index": trial_index,
                "result": result,
                "predictions": predictions,
                "targets": targets,
                "sample_ids": sample_ids,
                "artifacts": {
                    "result": common.identity(result_path),
                    "predictions_and_targets": common.identity(arrays_path),
                    "per_sample_metrics": common.identity(csv_path),
                },
            }
        )

    reference = runs[0]
    dataset_keys = (
        "samples",
        "valid_pixels",
        "sample_ids_sha256",
        "targets_array_sha256",
    )
    dataset_equal = all(
        all(
            run["result"]["dataset"][key] == reference["result"]["dataset"][key]
            for key in dataset_keys
        )
        and all(
            run["result"]["dataset"][artifact][field]
            == reference["result"]["dataset"][artifact][field]
            for artifact in ("frozen_inputs", "frozen_input_manifest")
            for field in ("size_bytes", "sha256")
        )
        for run in runs[1:]
    )
    targets_equal = all(np.array_equal(run["targets"], reference["targets"]) for run in runs[1:])
    ids_equal = all(run["sample_ids"] == reference["sample_ids"] for run in runs[1:])
    metric_keys = ("miou", "water_iou", "pixel_accuracy", "boundary_water_iou_corrected")
    spans_pp = {
        key: 100.0
        * (
            max(float(run["result"]["metrics"][key]) for run in runs)
            - min(float(run["result"]["metrics"][key]) for run in runs)
        )
        for key in metric_keys
    }
    pairwise = []
    for left, right in combinations(runs, 2):
        valid = (left["targets"] >= 0) & (left["targets"] < 2)
        changed = int(
            np.count_nonzero(left["predictions"][valid] != right["predictions"][valid])
        )
        pairwise.append(
            {
                "left_trial": left["trial_index"],
                "right_trial": right["trial_index"],
                "changed_valid_pixels": changed,
                "valid_pixel_agreement": float(
                    np.mean(left["predictions"][valid] == right["predictions"][valid])
                ),
            }
        )
    gates = {
        "exactly_three_independent_results": len(runs) == 3
        and len({run["artifacts"]["result"]["sha256"] for run in runs}) == 3,
        "all_individual_task_gates_passed": all(
            run["result"]["task_equivalence_admission_passed"] for run in runs
        ),
        "all_dataset_identities_equal": dataset_equal,
        "all_targets_exact": targets_equal,
        "all_sample_ids_exact": ids_equal,
        "miou_span_le_0_01pp": spans_pp["miou"] <= MIOU_WATER_SPAN_MAX_PP,
        "water_iou_span_le_0_01pp": spans_pp["water_iou"] <= MIOU_WATER_SPAN_MAX_PP,
        "boundary_span_le_0_02pp": spans_pp["boundary_water_iou_corrected"]
        <= BOUNDARY_SPAN_MAX_PP,
    }
    passed = all(gates.values())
    trial_rows = []
    for run in runs:
        trial_rows.append(
            {
                "candidate_id": manifest["candidate_id"],
                "trial": run["trial_index"],
                "miou": run["result"]["metrics"]["miou"],
                "water_iou": run["result"]["metrics"]["water_iou"],
                "boundary_water_iou_corrected": run["result"]["metrics"][
                    "boundary_water_iou_corrected"
                ],
                "pixel_accuracy": run["result"]["metrics"]["pixel_accuracy"],
                "prediction_agreement_vs_fp32": run["result"][
                    "valid_prediction_agreement_vs_frozen_fp32"
                ],
            }
        )
    summary = {
        "schema": "phase11_mixed_precision_test90_three_run_summary_v1",
        "status": "passed" if passed else "failed",
        "identities": {
            "aggregate_script": common.identity(Path(__file__)),
            "common_module": common.identity(Path(common.__file__)),
        },
        "candidate_lineage": {
            "candidate_id": manifest["candidate_id"],
            "manifest": manifest_identity,
            "fp16_backbone_blocks": manifest["fp16_backbone_blocks"],
            "int8_backbone_blocks": manifest["int8_backbone_blocks"],
        },
        "thresholds": {
            "individual_task_gates": common.TASK_GATES,
            "miou_span_max_pp": MIOU_WATER_SPAN_MAX_PP,
            "water_iou_span_max_pp": MIOU_WATER_SPAN_MAX_PP,
            "corrected_boundary_water_iou_span_max_pp": BOUNDARY_SPAN_MAX_PP,
        },
        "runs": [
            {
                "trial_index": run["trial_index"],
                "metrics": run["result"]["metrics"],
                "deltas_vs_frozen_fp32_pp": run["result"]["deltas_vs_frozen_fp32_pp"],
                "prediction_agreement_vs_fp32": run["result"][
                    "valid_prediction_agreement_vs_frozen_fp32"
                ],
                "artifacts": run["artifacts"],
            }
            for run in runs
        ],
        "metric_spans_pp": spans_pp,
        "pairwise_prediction_comparisons_descriptive": pairwise,
        "exact_prediction_determinism_descriptive": all(
            row["changed_valid_pixels"] == 0 for row in pairwise
        ),
        "gates": gates,
        "claims": {
            "task_equivalence_repeatability_confirmed": passed,
            "task_equivalence_performance_track_may_proceed": passed,
            "strict_numeric_equivalence_confirmed": False,
            "historical_m0_strict_failure_overridden": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
        "evidence_boundary": {
            "strict_numeric_equivalence_is_not_inferred_from_task_repeatability": True,
            "pairwise_exact_determinism_is_descriptive_not_a_gate": True,
            "performance_not_measured": True,
        },
    }
    common.json_dump(args.output_dir / "summary.json", summary)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
