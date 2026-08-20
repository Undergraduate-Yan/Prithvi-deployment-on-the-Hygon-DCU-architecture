#!/usr/bin/env python3
"""Fresh-process confirmation of FP16 task-level equivalence.

The frozen single-sample logits gate remains failed.  This confirmation only
establishes deterministic task-level accuracy for the compatibility variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EVALUATOR_SIZE = 11_488
EVALUATOR_SHA256 = "5817db57933914e1d560bfb614d20d047c5a3301ff3482e0c8b00f6f3a6b83ef"
FROZEN_RESULT_SIZE = 5_993
FROZEN_RESULT_SHA256 = "48de1f62acf10116254574851d5974528c4ce3df7363504b2fe59424de3a72dc"
FROZEN_CSV_SIZE = 25_666
FROZEN_CSV_SHA256 = "69ffbf22eb2b7380a3675ebe5650fbde12d1af951ecf3e168fe22126666cd530"
SINGLE_RESULT_SHA256 = "5e93e828ad44ca703c1eac5b08fd2c332d70002cec13c6d3c3c1b688f5a35a89"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, size: int | None = None, digest: str | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if size is not None and item["size_bytes"] != size:
        raise RuntimeError(f"size mismatch: {item}")
    if digest is not None and item["sha256"] != digest:
        raise RuntimeError(f"SHA mismatch: {item}")
    return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--frozen-result", type=Path, required=True)
    ap.add_argument("--frozen-csv", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--single-result", type=Path, required=True)
    ap.add_argument("--base-evaluator", type=Path, required=True)
    ap.add_argument("--helper", type=Path, required=True)
    ap.add_argument("--fp32-result", type=Path, required=True)
    ap.add_argument("--fp32-csv", type=Path, required=True)
    ap.add_argument("--fp32-tensors", type=Path, required=True)
    ap.add_argument("--frozen-inputs", type=Path, required=True)
    ap.add_argument("--frozen-input-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluator_file = record(args.evaluator, EVALUATOR_SIZE, EVALUATOR_SHA256)
    frozen_result_file = record(args.frozen_result, FROZEN_RESULT_SIZE, FROZEN_RESULT_SHA256)
    frozen_csv_file = record(args.frozen_csv, FROZEN_CSV_SIZE, FROZEN_CSV_SHA256)
    single_file = record(args.single_result, digest=SINGLE_RESULT_SHA256)
    frozen = json.loads(args.frozen_result.read_text(encoding="utf-8"))
    single = json.loads(args.single_result.read_text(encoding="utf-8"))
    if frozen.get("status") != "diagnostic_completed" or not frozen["descriptive_task_gates"]["passed"]:
        raise RuntimeError("frozen 90-sample diagnostic semantics mismatch")
    if single.get("status") != "failed" or single["claims"]["formal_exact_transform_equivalence"]:
        raise RuntimeError("frozen failed logits gate was not preserved")

    raw_dir = args.output_dir / "raw_confirmation"
    command = [
        sys.executable,
        str(args.evaluator),
        "--candidate", str(args.candidate),
        "--cache", str(args.cache),
        "--single-result", str(args.single_result),
        "--base-evaluator", str(args.base_evaluator),
        "--helper", str(args.helper),
        "--fp32-result", str(args.fp32_result),
        "--fp32-csv", str(args.fp32_csv),
        "--fp32-tensors", str(args.fp32_tensors),
        "--frozen-inputs", str(args.frozen_inputs),
        "--frozen-input-manifest", str(args.frozen_input_manifest),
        "--output-dir", str(raw_dir),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output_dir / "confirmation_evaluator.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"frozen evaluator failed with exit {completed.returncode}")

    new_result_path = raw_dir / "result.json"
    new_csv_path = raw_dir / "per_sample_metrics.csv"
    new_result_file = record(new_result_path)
    new_csv_file = record(new_csv_path)
    current = json.loads(new_result_path.read_text(encoding="utf-8"))
    gates = {
        "fresh_evaluator_exit_zero": completed.returncode == 0,
        "diagnostic_status_completed": current.get("status") == "diagnostic_completed",
        "task_gates_passed": bool(current["descriptive_task_gates"]["passed"]),
        "sample_and_input_contract_exact": current["dataset"] == frozen["dataset"],
        "metrics_exact_reproduction": current["metrics"] == frozen["metrics"],
        "confusion_exact_reproduction": current["confusion_matrix"] == frozen["confusion_matrix"],
        "deltas_exact_reproduction": current["deltas_percentage_points"] == frozen["deltas_percentage_points"],
        "agreement_exact_reproduction": current["valid_pixel_prediction_agreement"] == frozen["valid_pixel_prediction_agreement"],
        "changed_pixels_exact_reproduction": current["changed_valid_pixels"] == frozen["changed_valid_pixels"],
        "per_sample_csv_exact_reproduction": new_csv_file["size_bytes"] == FROZEN_CSV_SIZE and new_csv_file["sha256"] == FROZEN_CSV_SHA256,
        "fresh_migraphx_events_eq_90": current["placement"]["provider_event_counts"].get("MIGraphXExecutionProvider") == 90,
        "fresh_cpu_events_zero": current["placement"]["provider_event_counts"].get("CPUExecutionProvider", 0) == 0,
        "formal_logits_gate_remains_failed": single.get("status") == "failed" and not single["claims"]["formal_exact_transform_equivalence"],
    }
    passed = all(gates.values())
    summary = {
        "status": "task_level_confirmation_passed" if passed else "failed",
        "protocol": {
            "fresh_container_process_required": True,
            "fixed_90_sample_test_set": True,
            "exact_per_sample_prediction_reproduction_required": True,
            "strict_logits_equivalence_is_not_reclassified": True,
        },
        "frozen_evidence": {"evaluator": evaluator_file, "diagnostic_result": frozen_result_file, "per_sample_csv": frozen_csv_file, "failed_single_gate": single_file},
        "fresh_evidence": {"result": new_result_file, "per_sample_csv": new_csv_file},
        "metrics": current["metrics"],
        "deltas_percentage_points": current["deltas_percentage_points"],
        "valid_pixel_prediction_agreement": current["valid_pixel_prediction_agreement"],
        "placement": current["placement"],
        "gates": gates,
        "claims": {
            "strict_logits_numeric_equivalence": False,
            "task_level_accuracy_confirmed": passed,
            "compatibility_variant_only": True,
            "paired_performance_experiment_allowed": passed,
            "performance_result_available": False,
            "deployment_complete": False,
        },
    }
    summary_path = args.output_dir / "confirmation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
