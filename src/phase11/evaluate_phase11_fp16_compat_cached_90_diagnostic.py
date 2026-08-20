#!/usr/bin/env python3
"""Diagnostic-only 90-sample evaluation after the FP16 single gate failed."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort


CANDIDATE_SIZE = 638_970_735
CANDIDATE_SHA256 = "8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7"
CACHE_SIZE = 708_193_607
CACHE_SHA256 = "9fe8985dc8ce4a3cf9d829ad66a91de41bb67b22e01b8181c43bf15c74f7dd0a"
SINGLE_SIZE = 3_316
SINGLE_SHA256 = "5e93e828ad44ca703c1eac5b08fd2c332d70002cec13c6d3c3c1b688f5a35a89"
BASE_SIZE = 16_657
BASE_SHA256 = "a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f"
EXPECTED_SAMPLES = 90
EXPECTED_VALID = 3_927_398
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path: Path, size: int | None = None, digest: str | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if size is not None and item["size_bytes"] != size:
        raise RuntimeError(f"size mismatch: {item}")
    if digest is not None and item["sha256"] != digest:
        raise RuntimeError(f"SHA mismatch: {item}")
    return item


def load_base(path: Path):
    locked = artifact(path, BASE_SIZE, BASE_SHA256)
    spec = importlib.util.spec_from_file_location("phase11_fp16_diag_base", locked["path"])
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Phase-11 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, locked


def write_csv(path: Path, rows: list[dict]) -> dict:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return artifact(path)


def main() -> None:
    ap = argparse.ArgumentParser()
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

    candidate_file = artifact(args.candidate, CANDIDATE_SIZE, CANDIDATE_SHA256)
    cache_file = artifact(args.cache, CACHE_SIZE, CACHE_SHA256)
    single_file = artifact(args.single_result, SINGLE_SIZE, SINGLE_SHA256)
    single = json.loads(args.single_result.read_text(encoding="utf-8"))
    if single.get("status") != "failed":
        raise RuntimeError("this diagnostic must preserve the frozen failed single gate")
    if not single["claims"]["strict_migraphx_single_sample_placement"]:
        raise RuntimeError("single-sample strict placement did not pass")
    if single["claims"]["eligible_for_90_sample_task_evaluation"]:
        raise RuntimeError("unexpected single-gate semantics")
    if single["compiled_cache"]["sha256"] != CACHE_SHA256:
        raise RuntimeError("single-result/cache lineage mismatch")

    base, base_file = load_base(args.base_evaluator)
    helper, helper_file = base.load_helper(args.helper)
    fp32_csv_file, frozen_rows = helper.read_locked_fp32_csv(args.fp32_csv)
    dataset, inputs_file, manifest_file = base.load_frozen_dataset(
        args.frozen_inputs, args.frozen_input_manifest, frozen_rows, helper
    )
    frozen = helper.load_fp32_reference(
        args.fp32_result, args.fp32_tensors, fp32_csv_file, dataset
    )
    if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
        raise RuntimeError("official ORT/MIGraphX runtime mismatch")

    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(args.cache.resolve())
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(args.cache.resolve())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output_dir / "fp16_cached_90_profile")
    started = perf_counter()
    session = ort.InferenceSession(
        str(args.candidate), sess_options=options, providers=[(MGX, {"device_id": 0})]
    )
    session.disable_fallback()
    creation_seconds = perf_counter() - started
    registered = session.get_providers()
    io_inputs = [{"name": x.name, "shape": list(x.shape), "type": x.type} for x in session.get_inputs()]
    io_outputs = [{"name": x.name, "shape": list(x.shape), "type": x.type} for x in session.get_outputs()]
    if io_inputs != [{"name": "image", "shape": [1, 6, 224, 224], "type": "tensor(float)"}]:
        raise RuntimeError(f"input contract drift: {io_inputs}")
    if len(io_outputs) != 2 or io_outputs[0] != {"name": "logits", "shape": [1, 2, 224, 224], "type": "tensor(float)"}:
        raise RuntimeError(f"output contract drift: {io_outputs}")

    confusion = np.zeros((2, 2), dtype=np.int64)
    predictions, rows = [], []
    total_loss = 0.0
    total_valid = 0
    started = perf_counter()
    try:
        for index, (raw, target) in enumerate(zip(dataset["raw_inputs"], dataset["targets"], strict=True)):
            logits = helper.validate_logits(
                session.run(["logits"], {"image": raw})[0], f"FP16 MIGraphX sample {index}"
            )
            pred = np.argmax(logits, axis=1).astype(np.uint8)
            current, valid = helper.sample_confusion(pred, target)
            confusion += current
            total_valid += valid
            total_loss += helper.cross_entropy_sum(logits, target)
            predictions.append(np.ascontiguousarray(pred[0]))
            rows.append({
                "sample_index": index,
                "sample_id": dataset["sample_ids"][index],
                "raw_scaled_fp32_sha256": dataset["manifest_rows"][index]["raw_scaled_fp32_sha256"],
                "target_int64_sha256": dataset["manifest_rows"][index]["target_int64_sha256"],
                "logits_fp32_sha256": helper.array_sha256(logits),
                "prediction_uint8_sha256": helper.array_sha256(pred),
                "valid_pixels": valid,
            })
            print(f"FP16 MIGraphX {index + 1}/{EXPECTED_SAMPLES} {dataset['sample_ids'][index]}", flush=True)
    finally:
        evaluation_seconds = perf_counter() - started
        profile_path = Path(session.end_profiling()).resolve(strict=True)
        del session
        gc.collect()

    if len(predictions) != EXPECTED_SAMPLES or total_valid != EXPECTED_VALID:
        raise RuntimeError("90-sample completeness failure")
    prediction_array = np.stack(predictions).astype(np.uint8, copy=False)
    target_array = np.concatenate(dataset["targets"], axis=0).astype(np.int64, copy=False)
    metrics = helper.confusion_metrics(confusion)
    metrics["loss"] = float(total_loss / total_valid)
    metrics["valid_pixels"] = total_valid
    boundary = helper.corrected_boundary_counts(prediction_array, target_array, thickness=2)
    metrics["boundary_water_iou_corrected"] = boundary["water_iou_corrected"]
    frozen_metrics = frozen["metrics"]
    deltas_pp = {
        key: 100.0 * (float(metrics[key]) - float(frozen_metrics[key]))
        for key in ("miou", "water_iou", "pixel_accuracy", "boundary_water_iou_corrected")
    }
    valid_mask = (target_array >= 0) & (target_array < 2)
    frozen_predictions = frozen["predictions"]
    agreement = float(np.mean(prediction_array[valid_mask] == frozen_predictions[valid_mask]))
    changed = int(np.count_nonzero(prediction_array[valid_mask] != frozen_predictions[valid_mask]))
    placement = base.parse_profile(profile_path)
    descriptive_gates = {
        "sample_count_eq_90": len(predictions) == EXPECTED_SAMPLES,
        "valid_pixels_exact": total_valid == EXPECTED_VALID,
        "migraphx_events_positive": placement["strict_migraphx_positive"],
        "cpu_events_zero": placement["cpu_zero"],
        "miou_drop_le_0_5pp": deltas_pp["miou"] >= -0.5,
        "water_iou_drop_le_1_0pp": deltas_pp["water_iou"] >= -1.0,
        "boundary_drop_le_1_0pp": deltas_pp["boundary_water_iou_corrected"] >= -1.0,
        "valid_prediction_agreement_ge_99_5pct": agreement >= 0.995,
    }
    tensors_path = args.output_dir / "predictions_and_targets.npz"
    np.savez_compressed(tensors_path, predictions=prediction_array, targets=target_array, sample_ids=np.asarray(dataset["sample_ids"]))
    csv_file = write_csv(args.output_dir / "per_sample_metrics.csv", rows)
    result = {
        "status": "diagnostic_completed",
        "lineage": {"candidate": candidate_file, "compiled_cache": cache_file, "failed_single_gate": single_file, "base_evaluator": base_file, "metric_helper": helper_file, "frozen_inputs": inputs_file, "frozen_manifest": manifest_file},
        "runtime": {"onnxruntime": ort.__version__, "registered_providers": registered, "cache_load_and_session_seconds": creation_seconds, "evaluation_seconds_diagnostic_only": evaluation_seconds, "performance_claim_allowed": False},
        "io_contract": {"inputs": io_inputs, "outputs": io_outputs, "requested_outputs": ["logits"]},
        "dataset": {"sample_count": len(predictions), "valid_pixels": total_valid, "sample_ids_sha256": dataset["sample_ids_sha256"], "raw_hash_list_sha256": dataset["raw_hash_list_sha256"], "target_hash_list_sha256": dataset["target_hash_list_sha256"]},
        "metrics": metrics,
        "frozen_fp32_metrics": frozen_metrics,
        "deltas_percentage_points": deltas_pp,
        "valid_pixel_prediction_agreement": agreement,
        "changed_valid_pixels": changed,
        "confusion_matrix": metrics["confusion_matrix"],
        "placement": placement,
        "descriptive_task_gates": {"passed": all(descriptive_gates.values()), "gates": descriptive_gates},
        "artifacts": {"per_sample_csv": csv_file, "predictions_and_targets": artifact(tensors_path)},
        "claims": {"formal_single_numeric_admission": False, "diagnostic_90_completed": True, "formal_fp16_deployment_passed": False, "performance": False, "deployment_complete": False},
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
