#!/usr/bin/env python3
'Research implementation: evaluate precision maps.'

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np


EXPECTED_ORT = "1.19.2"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
RETAIN_AFTER = (5, 11, 17, 23)
FP32_MODEL = (1_277_126_412, "ac1a502b79a6551788dabc0fb25dda81e1e5c3f747197f955aaf8e0861fb0fec")
FP32_CACHE = (1_413_746_440, "43318e7555063c7733603e16c3c27b6223105423b530a7b70d3f01130ef295df")
TASK_GATES = {
    "miou_drop_pp_max": 0.5,
    "water_iou_drop_pp_max": 1.0,
    "boundary_water_iou_drop_pp_max": 1.0,
    "prediction_agreement_min": 0.995,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, expected: dict[str, Any] | tuple[int, str] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None:
        pair = (
            int(expected["size_bytes"]), str(expected["sha256"])
        ) if isinstance(expected, dict) else expected
        if (item["size_bytes"], item["sha256"]) != pair:
            raise RuntimeError(f"artifact identity drift: {item}; expected={pair}")
    return item


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def resolve_payload(manifest: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(strict=True)


def load_pack(pack_path: Path, manifest_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "journal_stage1_input_pack_v1" or manifest.get("status") != "passed":
        raise RuntimeError("input manifest is not the passed Stage-1 pack")
    if manifest.get("selection_role") != "configuration-validation only; precision-map selection permitted":
        raise RuntimeError("input role is not configuration-validation only")
    gates = manifest.get("leakage_gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise RuntimeError("not every leakage gate is true")
    identity(pack_path, manifest["output_pack"])
    with np.load(pack_path, allow_pickle=False) as pack:
        if set(pack.files) != {"inputs", "targets", "sample_ids"}:
            raise RuntimeError("unexpected input-pack members")
        inputs = np.ascontiguousarray(pack["inputs"], dtype=np.float32)
        targets = np.ascontiguousarray(pack["targets"], dtype=np.int64)
        sample_ids = [str(value) for value in pack["sample_ids"].tolist()]
    expected = manifest["configuration_validation"]
    if inputs.shape != (64, 6, 224, 224) or targets.shape != (64, 224, 224):
        raise RuntimeError("configuration-validation tensor shape drift")
    if sample_ids != expected["sample_ids"]:
        raise RuntimeError("configuration-validation ID order drift")
    if array_sha256(inputs) != expected["inputs_sha256"] or array_sha256(targets) != expected["targets_sha256"]:
        raise RuntimeError("configuration-validation tensor digest drift")
    return inputs, targets, sample_ids, manifest


def session_options(ort: Any, profile_prefix: Path | None = None) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def set_cache(path: Path) -> None:
    path = path.resolve(strict=True)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(path)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(path)


def make_session(ort: Any, asset: dict[str, Any], profile_prefix: Path | None = None) -> Any:
    set_cache(asset["cache_path"])
    session = ort.InferenceSession(
        str(asset["model_path"]),
        sess_options=session_options(ort, profile_prefix),
        providers=[(MGX, {"device_id": 0})],
    )
    session.disable_fallback()
    if not session.get_providers() or session.get_providers()[0] != MGX:
        raise RuntimeError(f"provider priority drift for {asset['label']}")
    return session


def profile_counts(path: Path) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    result = {
        "provider_event_counts": dict(sorted(counts.items())),
        "migraphx_events_positive": int(counts.get(MGX, 0)) > 0,
        "cpu_events_zero": int(counts.get(CPU, 0)) == 0,
    }
    result["passed"] = result["migraphx_events_positive"] and result["cpu_events_zero"]
    return result


def load_assets(
    int8_manifest_path: Path,
    all_fp16_manifest_path: Path,
    fp16_compile_result_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    int8_manifest = json.loads(int8_manifest_path.read_text(encoding="utf-8"))
    if int8_manifest.get("status") != "assembled_and_identity_locked" or len(int8_manifest.get("segments", [])) != 25:
        raise RuntimeError("frozen INT8 bundle manifest drift")
    int8_assets = []
    for index, row in enumerate(int8_manifest["segments"][:24]):
        if row.get("index") != index or row.get("label") != f"encoder_block_{index:02d}":
            raise RuntimeError("INT8 block order drift")
        model = resolve_payload(int8_manifest_path, row["model"])
        cache = resolve_payload(int8_manifest_path, row["cache"])
        int8_assets.append(
            {
                "index": index,
                "label": row["label"],
                "precision": "int8_qdq",
                "model_path": model,
                "cache_path": cache,
                "model": identity(model, row["model_identity"]),
                "cache": identity(cache, row["cache_identity"]),
                "inputs": row["inputs"],
                "outputs": row["outputs"],
            }
        )

    fp16_manifest = json.loads(all_fp16_manifest_path.read_text(encoding="utf-8"))
    rows = fp16_manifest.get("segments", [])
    if fp16_manifest.get("candidate_id") != "M4" or fp16_manifest.get("fp16_backbone_blocks") != list(range(24)):
        raise RuntimeError("all-FP16 construction manifest drift")
    compile_result = json.loads(fp16_compile_result_path.read_text(encoding="utf-8"))
    if compile_result.get("status") != "passed" or len(compile_result.get("segments", [])) != 24:
        raise RuntimeError("all-FP16 cache compile result drift")
    cache_rows = {int(row["index"]): row for row in compile_result["segments"]}
    fp16_assets = []
    for index, row in enumerate(rows[:24]):
        receipt = cache_rows[index]
        if not receipt.get("cache_and_placement_passed") or not all(receipt.get("essential_gates", {}).values()):
            raise RuntimeError(f"FP16 cache receipt failed for block {index}")
        model = resolve_payload(all_fp16_manifest_path, row["model"])
        cache = Path(receipt["cache"]["path"]).resolve(strict=True)
        fp16_assets.append(
            {
                "index": index,
                "label": row["label"],
                "precision": "fp16",
                "model_path": model,
                "cache_path": cache,
                "model": identity(model, row["model_identity"]),
                "cache": identity(cache, receipt["cache"]),
                "inputs": row["inputs"],
                "outputs": row["outputs"],
                "placement_receipt": {
                    "migraphx_profile": receipt["migraphx_profile"],
                    "essential_gates": receipt["essential_gates"],
                },
            }
        )
    head_row = rows[24]
    head_model = resolve_payload(all_fp16_manifest_path, head_row["model"])
    head_cache = resolve_payload(all_fp16_manifest_path, head_row["cache"])
    head = {
        "index": 24,
        "label": head_row["label"],
        "precision": "fp32_compat_barrier",
        "model_path": head_model,
        "cache_path": head_cache,
        "model": identity(head_model, head_row["model_identity"]),
        "cache": identity(head_cache, head_row["cache_identity"]),
        "inputs": head_row["inputs"],
        "outputs": head_row["outputs"],
    }
    evidence = {
        "int8_bundle_manifest": identity(int8_manifest_path),
        "all_fp16_manifest": identity(all_fp16_manifest_path),
        "fp16_compile_result": identity(fp16_compile_result_path),
    }
    return int8_assets, fp16_assets, head, evidence


def bind_one(session: Any, inputs: dict[str, Any], output_name: str) -> Any:
    binding = session.io_binding()
    expected = {item.name for item in session.get_inputs()}
    if expected != set(inputs):
        raise RuntimeError(f"session input contract drift: {expected} != {set(inputs)}")
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    binding.bind_output(output_name, "cuda", 0)
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    outputs = binding.get_outputs()
    if len(outputs) != 1 or outputs[0].device_name() != "cuda":
        raise RuntimeError("device output contract drift")
    return outputs[0]


def run_pipeline(
    int8_sessions: list[Any],
    head_session: Any,
    raw_device: list[Any],
    fp16_overrides: dict[int, Any],
    *,
    capture_block_inputs: bool = False,
) -> tuple[np.ndarray, dict[int, list[Any]]]:
    current = list(raw_device)
    retained: dict[str, list[Any]] = {}
    captured: dict[int, list[Any]] = {}
    for index in range(24):
        if capture_block_inputs:
            captured[index] = list(current)
        session = fp16_overrides.get(index, int8_sessions[index])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        current = [bind_one(session, {input_name: value}, output_name) for value in current]
        if index in RETAIN_AFTER:
            retained[output_name] = list(current)
    output_name = next(item.name for item in head_session.get_outputs() if item.name == "logits")
    logits = []
    for sample_index in range(len(raw_device)):
        feed = {item.name: retained[item.name][sample_index] for item in head_session.get_inputs()}
        value = bind_one(head_session, feed, output_name)
        host = np.ascontiguousarray(value.numpy(), dtype=np.float32)
        if host.shape != (1, 2, 224, 224) or not np.isfinite(host).all():
            raise RuntimeError("mixed pipeline logits contract drift")
        logits.append(host[0])
    return np.stack(logits), captured


def run_fp32_reference(ort: Any, model: Path, cache: Path, inputs: np.ndarray, output_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    model_identity = identity(model, FP32_MODEL)
    cache_identity = identity(cache, FP32_CACHE)
    asset = {"label": "Mono-FP32-FPN4-barrier", "model_path": model, "cache_path": cache}
    session = make_session(ort, asset, output_dir / "fp32_reference_profile")
    if session.get_inputs()[0].name != "image" or "logits" not in {item.name for item in session.get_outputs()}:
        raise RuntimeError("FP32 reference I/O contract drift")
    device_inputs = [ort.OrtValue.ortvalue_from_numpy(value[None], "cuda", 0) for value in inputs]
    logits = []
    for value in device_inputs:
        output = bind_one(session, {"image": value}, "logits")
        host = np.ascontiguousarray(output.numpy(), dtype=np.float32)
        if host.shape != (1, 2, 224, 224) or not np.isfinite(host).all():
            raise RuntimeError("FP32 reference output contract drift")
        logits.append(host[0])
    profile = Path(session.end_profiling()).resolve(strict=True)
    placement = profile_counts(profile)
    if not placement["passed"]:
        raise RuntimeError(f"FP32 reference placement failed: {placement}")
    report = {
        "model": model_identity,
        "cache": cache_identity,
        "profile": {**identity(profile), **placement},
        "external_io": {"input": "FP32[1,6,224,224]", "output": "FP32[1,2,224,224]"},
        "sample_count": len(logits),
    }
    del session, device_inputs
    gc.collect()
    return np.stack(logits), report


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    iou = [safe_ratio(true_positive[i], true_positive[i] + false_positive[i] + false_negative[i]) for i in range(2)]
    return {
        "miou": float(np.mean(iou)),
        "background_iou": iou[0],
        "water_iou": iou[1],
        "pixel_accuracy": safe_ratio(true_positive.sum(), matrix.sum()),
        "confusion_matrix": matrix.tolist(),
    }


def corrected_boundary_water_iou(predictions: np.ndarray, targets: np.ndarray, thickness: int = 2) -> float:
    import torch
    import torch.nn.functional as functional

    prediction = torch.from_numpy(np.ascontiguousarray(predictions, dtype=np.int64))
    target = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.int64))
    kernel = 2 * thickness + 1
    prediction_mask = (prediction == 1).float().unsqueeze(1)
    target_mask = (target == 1).float().unsqueeze(1)
    prediction_boundary = (
        functional.max_pool2d(prediction_mask, kernel, stride=1, padding=thickness)
        - (1.0 - functional.max_pool2d(1.0 - prediction_mask, kernel, stride=1, padding=thickness))
    ).clamp_min(0.0) > 0.5
    target_boundary = (
        functional.max_pool2d(target_mask, kernel, stride=1, padding=thickness)
        - (1.0 - functional.max_pool2d(1.0 - target_mask, kernel, stride=1, padding=thickness))
    ).clamp_min(0.0) > 0.5
    valid = (target != -1).unsqueeze(1)
    prediction_boundary &= valid
    target_boundary &= valid
    intersection = int((prediction_boundary & target_boundary).sum())
    union = int((prediction_boundary | target_boundary).sum())
    return safe_ratio(intersection, union)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(logits, dtype=np.float64).transpose(0, 2, 3, 1)
    valid = (targets >= 0) & (targets < 2)
    selected = scores[valid]
    labels = targets[valid]
    maxima = selected.max(axis=1)
    log_sum_exp = maxima + np.log(np.exp(selected - maxima[:, None]).sum(axis=1))
    return float(np.mean(log_sum_exp - selected[np.arange(selected.shape[0]), labels]))


def evaluate_logits(logits: np.ndarray, targets: np.ndarray, reference_predictions: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    if logits.shape != (64, 2, 224, 224) or not np.isfinite(logits).all():
        raise RuntimeError("evaluation logits grid drift")
    predictions = np.argmax(logits, axis=1).astype(np.uint8)
    confusion = np.zeros((2, 2), dtype=np.int64)
    scene_rows = []
    for index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        valid = (target >= 0) & (target < 2)
        encoded = target[valid] * 2 + prediction[valid]
        scene_confusion = np.bincount(encoded, minlength=4).reshape(2, 2).astype(np.int64)
        confusion += scene_confusion
        values = confusion_metrics(scene_confusion)
        scene_rows.append({"sample_index": index, "valid_pixels": int(valid.sum()), **values})
    metrics = confusion_metrics(confusion)
    metrics.update(
        {
            "loss": cross_entropy(logits, targets),
            "boundary_water_iou_corrected": corrected_boundary_water_iou(predictions, targets),
            "valid_pixels": int(np.count_nonzero((targets >= 0) & (targets < 2))),
            "nonfinite_outputs": 0,
        }
    )
    if reference_predictions is not None:
        valid = (targets >= 0) & (targets < 2)
        metrics["prediction_agreement_vs_fp32"] = float(np.mean(predictions[valid] == reference_predictions[valid]))
        metrics["changed_valid_pixels_vs_fp32"] = int(np.count_nonzero(predictions[valid] != reference_predictions[valid]))
    return metrics, predictions, scene_rows


def timed_block(session: Any, values: list[Any], passes: int, precision: str, block: int, sample_ids: list[str]) -> tuple[list[float], list[dict[str, Any]]]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(10):
        bind_one(session, {input_name: values[0]}, output_name)
    latencies = []
    rows = []
    for pass_index in range(passes):
        order = range(len(values)) if pass_index % 2 == 0 else range(len(values) - 1, -1, -1)
        for position, sample_index in enumerate(order):
            binding = session.io_binding()
            binding.bind_ortvalue_input(input_name, values[sample_index])
            binding.bind_output(output_name, "cuda", 0)
            started = perf_counter_ns()
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            elapsed = (perf_counter_ns() - started) / 1_000_000.0
            outputs = binding.get_outputs()
            if len(outputs) != 1 or outputs[0].device_name() != "cuda" or not np.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"timed block contract failed: {block}/{precision}")
            latencies.append(elapsed)
            rows.append(
                {
                    "block_index": block,
                    "precision": precision,
                    "pass_index": pass_index,
                    "position": position,
                    "sample_index": sample_index,
                    "sample_id": sample_ids[sample_index],
                    "latency_ms": elapsed,
                }
            )
    return latencies, rows


def map_payload(name: str, fp16_blocks: list[int], source: str, evidence: dict[str, Any]) -> dict[str, Any]:
    blocks = sorted(fp16_blocks)
    return {
        "schema": "journal_precision_map_v1",
        "name": name,
        "status": "selected_and_configuration_validation_evaluated",
        "seed": 42,
        "selection_source": source,
        "selection_evidence": evidence,
        "external_io": {"input": "FP32[1,6,224,224]", "output": "FP32[1,2,224,224]"},
        "backbone": {"fp16_blocks": blocks, "int8_blocks": [i for i in range(24) if i not in blocks]},
        "head_precision": "FP32",
        "acceptance": {
            "configuration_selection_split": "configuration-validation only",
            "formal_test_split_may_not_select_configuration": True,
        },
    }


def load_maps(path: Path) -> dict[str, list[int]]:
    maps = {}
    for config in sorted(path.glob("*.json")):
        payload = json.loads(config.read_text(encoding="utf-8"))
        if payload.get("schema") != "journal_precision_map_v1" or payload.get("status") == "BLOCKED":
            continue
        blocks = [int(value) for value in payload.get("backbone", {}).get("fp16_blocks", [])]
        if len(blocks) != len(set(blocks)) or any(value not in range(24) for value in blocks):
            raise RuntimeError(f"invalid precision map: {config}")
        maps[str(payload["name"])] = sorted(blocks)
    required = {"Legacy-M5", "Early-9", "Late-9", *(f"Random-9-seed-{seed}" for seed in range(42, 47)), *(f"M{i}-R" for i in range(6))}
    if not required.issubset(maps):
        raise RuntimeError(f"precision map directory is incomplete: {sorted(required - set(maps))}")
    return maps


def admission(metrics: dict[str, Any], reference: dict[str, Any]) -> tuple[dict[str, float], dict[str, bool]]:
    deltas = {
        key: 100.0 * (float(metrics[key]) - float(reference[key]))
        for key in ("miou", "water_iou", "boundary_water_iou_corrected")
    }
    gates = {
        "miou_drop_le_0_5pp": deltas["miou"] >= -TASK_GATES["miou_drop_pp_max"],
        "water_iou_drop_le_1_0pp": deltas["water_iou"] >= -TASK_GATES["water_iou_drop_pp_max"],
        "boundary_water_iou_drop_le_1_0pp": deltas["boundary_water_iou_corrected"] >= -TASK_GATES["boundary_water_iou_drop_pp_max"],
        "prediction_agreement_vs_fp32_ge_99_5pct": metrics["prediction_agreement_vs_fp32"] >= TASK_GATES["prediction_agreement_min"],
        "nonfinite_outputs_eq_0": metrics["nonfinite_outputs"] == 0,
        "configuration_validation_only": True,
        "formal_test_not_used": True,
        "strict_migraphx_no_cpu_fallback_session_mode": True,
    }
    return deltas, gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pack", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--int8-manifest", required=True, type=Path)
    parser.add_argument("--all-fp16-manifest", required=True, type=Path)
    parser.add_argument("--fp16-compile-result", required=True, type=Path)
    parser.add_argument("--precision-map-dir", required=True, type=Path)
    parser.add_argument("--fp32-model", required=True, type=Path)
    parser.add_argument("--fp32-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-passes", type=int, default=3)
    args = parser.parse_args()
    if args.runtime_passes < 3:
        raise ValueError("runtime-passes must be at least 3")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "run_report.json"
    result: dict[str, Any] = {
        "schema": "journal_stage1_precision_map_configuration_validation_v1",
        "status": "failed",
        "selection_split": "configuration-validation only",
        "formal_test_used_for_selection": False,
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != EXPECTED_ORT or MGX not in ort.get_available_providers():
            raise RuntimeError(f"runtime identity drift: {ort.__version__}, {ort.get_available_providers()}")
        inputs, targets, sample_ids, input_manifest = load_pack(args.input_pack, args.input_manifest)
        int8_assets, fp16_assets, head_asset, asset_evidence = load_assets(
            args.int8_manifest.resolve(strict=True),
            args.all_fp16_manifest.resolve(strict=True),
            args.fp16_compile_result.resolve(strict=True),
        )
        maps = load_maps(args.precision_map_dir.resolve(strict=True))

        fp32_logits, fp32_runtime = run_fp32_reference(
            ort, args.fp32_model.resolve(strict=True), args.fp32_cache.resolve(strict=True), inputs, args.output_dir
        )
        fp32_metrics, fp32_predictions, fp32_scene = evaluate_logits(fp32_logits, targets, None)
        del fp32_logits
        gc.collect()

        raw_device = [ort.OrtValue.ortvalue_from_numpy(value[None], "cuda", 0) for value in inputs]
        int8_sessions = []
        for index, asset in enumerate(int8_assets):
            int8_sessions.append(make_session(ort, asset))
            print(f"INT8 session {index:02d}/23 loaded", flush=True)
        head_session = make_session(ort, head_asset)
        print("FP32 compatibility head session loaded", flush=True)

        baseline_logits, block_inputs = run_pipeline(
            int8_sessions, head_session, raw_device, {}, capture_block_inputs=True
        )
        baseline_metrics, baseline_predictions, baseline_scene = evaluate_logits(
            baseline_logits, targets, fp32_predictions
        )
        del baseline_logits

        sensitivity_rows = []
        runtime_rows = []
        runtime_raw_rows = []
        for block in range(24):
            fp16_session = make_session(ort, fp16_assets[block])
            first = ("int8", int8_sessions[block], "fp16", fp16_session) if block % 2 == 0 else ("fp16", fp16_session, "int8", int8_sessions[block])
            measured: dict[str, list[float]] = {}
            for offset in (0, 2):
                precision, session = first[offset], first[offset + 1]
                values, raw_rows = timed_block(
                    session, block_inputs[block], args.runtime_passes, precision, block, sample_ids
                )
                measured[precision] = values
                runtime_raw_rows.extend(raw_rows)
            int8_values = np.asarray(measured["int8"], dtype=np.float64)
            fp16_values = np.asarray(measured["fp16"], dtype=np.float64)
            runtime_rows.append(
                {
                    "block_index": block,
                    "measurement_count_per_precision": int(int8_values.size),
                    "int8_median_ms": float(np.median(int8_values)),
                    "int8_p95_ms": float(np.percentile(int8_values, 95)),
                    "fp16_median_ms": float(np.median(fp16_values)),
                    "fp16_p95_ms": float(np.percentile(fp16_values, 95)),
                    "net_fp16_benefit_median_ms": float(np.median(int8_values) - np.median(fp16_values)),
                    "net_fp16_benefit_p95_ms": float(np.percentile(int8_values, 95) - np.percentile(fp16_values, 95)),
                }
            )
            logits, _ = run_pipeline(int8_sessions, head_session, raw_device, {block: fp16_session})
            metrics, predictions, _ = evaluate_logits(logits, targets, fp32_predictions)
            sensitivity_rows.append(
                {
                    "block_index": block,
                    "miou": metrics["miou"],
                    "water_iou": metrics["water_iou"],
                    "boundary_water_iou_corrected": metrics["boundary_water_iou_corrected"],
                    "loss": metrics["loss"],
                    "miou_gain_vs_all_int8_pp": 100.0 * (metrics["miou"] - baseline_metrics["miou"]),
                    "water_iou_gain_vs_all_int8_pp": 100.0 * (metrics["water_iou"] - baseline_metrics["water_iou"]),
                    "prediction_agreement_vs_fp32": metrics["prediction_agreement_vs_fp32"],
                }
            )
            del fp16_session, logits
            gc.collect()
            print(f"single-block sensitivity/runtime {block:02d}/23 complete", flush=True)

        sensitivity_rank = sorted(
            sensitivity_rows,
            key=lambda row: (
                -float(row["miou"]),
                -float(row["water_iou"]),
                -float(row["boundary_water_iou_corrected"]),
                float(row["loss"]),
                int(row["block_index"]),
            ),
        )
        runtime_rank = sorted(
            runtime_rows,
            key=lambda row: (
                -float(row["net_fp16_benefit_median_ms"]),
                -float(row["net_fp16_benefit_p95_ms"]),
                int(row["block_index"]),
            ),
        )
        sensitivity_blocks = sorted(int(row["block_index"]) for row in sensitivity_rank[:9])
        runtime_blocks = sorted(int(row["block_index"]) for row in runtime_rank[:9])
        generated_dir = args.output_dir / "selected_configs"
        generated_dir.mkdir()
        sensitivity_config = map_payload(
            "Sensitivity-9",
            sensitivity_blocks,
            "descending singleton configuration-validation mIoU with locked tie-breakers",
            {"ranking": [int(row["block_index"]) for row in sensitivity_rank], "formal_test_used": False},
        )
        runtime_config = map_payload(
            "Runtime-9",
            runtime_blocks,
            "descending K100 FP16-over-INT8 median block latency benefit",
            {
                "ranking": [int(row["block_index"]) for row in runtime_rank],
                "runtime_passes": args.runtime_passes,
                "scenes_per_pass": 64,
                "formal_test_used": False,
            },
        )
        write_json(generated_dir / "Sensitivity-9.json", sensitivity_config)
        write_json(generated_dir / "Runtime-9.json", runtime_config)
        maps["Sensitivity-9"] = sensitivity_blocks
        maps["Runtime-9"] = runtime_blocks

        configuration_results = []
        prediction_artifacts: dict[str, np.ndarray] = {"Mono_FP32": fp32_predictions, "M0_R": baseline_predictions}
        per_scene_rows = []
        evaluated_cache: dict[tuple[int, ...], tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]] = {
            tuple(): (baseline_metrics, baseline_predictions, baseline_scene)
        }
        for name, blocks in sorted(maps.items()):
            key = tuple(blocks)
            if key in evaluated_cache:
                metrics, predictions, scene_values = evaluated_cache[key]
            else:
                overrides = {block: make_session(ort, fp16_assets[block]) for block in blocks}
                logits, _ = run_pipeline(int8_sessions, head_session, raw_device, overrides)
                metrics, predictions, scene_values = evaluate_logits(logits, targets, fp32_predictions)
                evaluated_cache[key] = (metrics, predictions, scene_values)
                del overrides, logits
                gc.collect()
            deltas, gates = admission(metrics, fp32_metrics)
            configuration_results.append(
                {
                    "name": name,
                    "fp16_blocks": blocks,
                    "int8_blocks": [index for index in range(24) if index not in blocks],
                    "metrics": metrics,
                    "deltas_vs_fp32_percentage_points": deltas,
                    "task_gates": gates,
                    "task_admission_passed": all(gates.values()),
                }
            )
            prediction_artifacts[name.replace("-", "_")] = predictions
            for scene, sample_id in zip(scene_values, sample_ids, strict=True):
                per_scene_rows.append({"configuration": name, "sample_id": sample_id, **scene})
            print(f"configuration {name} evaluated: FP16 blocks={blocks}", flush=True)

        def write_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
            if not rows:
                raise RuntimeError(f"refusing empty CSV: {path}")
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return identity(path)

        sensitivity_csv = write_csv(args.output_dir / "single_block_sensitivity.csv", sensitivity_rows)
        runtime_csv = write_csv(args.output_dir / "per_block_runtime.csv", runtime_rows)
        runtime_raw_csv = write_csv(args.output_dir / "per_block_runtime_raw.csv", runtime_raw_rows)
        scene_csv = write_csv(args.output_dir / "per_scene_configuration_metrics.csv", per_scene_rows)
        configuration_csv_rows = []
        for row in configuration_results:
            configuration_csv_rows.append(
                {
                    "name": row["name"],
                    "fp16_blocks": ";".join(map(str, row["fp16_blocks"])),
                    "miou": row["metrics"]["miou"],
                    "water_iou": row["metrics"]["water_iou"],
                    "boundary_water_iou_corrected": row["metrics"]["boundary_water_iou_corrected"],
                    "loss": row["metrics"]["loss"],
                    "prediction_agreement_vs_fp32": row["metrics"]["prediction_agreement_vs_fp32"],
                    "miou_delta_pp": row["deltas_vs_fp32_percentage_points"]["miou"],
                    "water_iou_delta_pp": row["deltas_vs_fp32_percentage_points"]["water_iou"],
                    "boundary_delta_pp": row["deltas_vs_fp32_percentage_points"]["boundary_water_iou_corrected"],
                    "task_admission_passed": row["task_admission_passed"],
                }
            )
        configuration_csv = write_csv(args.output_dir / "configuration_metrics.csv", configuration_csv_rows)
        predictions_path = args.output_dir / "configuration_predictions_and_targets.npz"
        np.savez_compressed(
            predictions_path,
            targets=targets,
            sample_ids=np.asarray(sample_ids),
            **prediction_artifacts,
        )

        result.update(
            {
                "status": "passed",
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "onnxruntime": ort.__version__,
                    "provider": MGX,
                    "intersegment_transport": "direct device OrtValue IOBinding",
                    "strict_cpu_fallback_disabled": True,
                },
                "dataset": {
                    "sample_count": 64,
                    "valid_pixels": fp32_metrics["valid_pixels"],
                    "sample_ids_sha256": input_manifest["configuration_validation"]["sample_ids_sha256"],
                    "inputs_sha256": input_manifest["configuration_validation"]["inputs_sha256"],
                    "targets_sha256": input_manifest["configuration_validation"]["targets_sha256"],
                    "selection_role": "configuration-validation only",
                    "formal_test_used": False,
                },
                "identities": {
                    "runner": identity(Path(__file__)),
                    "input_pack": identity(args.input_pack, input_manifest["output_pack"]),
                    "input_manifest": identity(args.input_manifest),
                    **asset_evidence,
                },
                "fp32_reference": {"runtime": fp32_runtime, "metrics": fp32_metrics},
                "all_int8_baseline_metrics": baseline_metrics,
                "selection": {
                    "Sensitivity-9": sensitivity_config,
                    "Runtime-9": runtime_config,
                },
                "configuration_results": configuration_results,
                "artifacts": {
                    "single_block_sensitivity": sensitivity_csv,
                    "per_block_runtime": runtime_csv,
                    "per_block_runtime_raw": runtime_raw_csv,
                    "per_scene_configuration_metrics": scene_csv,
                    "configuration_metrics": configuration_csv,
                    "configuration_predictions_and_targets": identity(predictions_path),
                    "Sensitivity-9": identity(generated_dir / "Sensitivity-9.json"),
                    "Runtime-9": identity(generated_dir / "Runtime-9.json"),
                },
                "gates": {
                    "configuration_validation_only": True,
                    "formal_test_not_used": True,
                    "exact_64_scenes": True,
                    "all_24_singleton_sensitivity_runs_complete": len(sensitivity_rows) == 24,
                    "all_24_runtime_comparisons_complete": len(runtime_rows) == 24,
                    "five_random_seeds_evaluated": all(f"Random-9-seed-{seed}" in maps for seed in range(42, 47)),
                    "revised_sequence_evaluated": all(f"M{i}-R" in maps for i in range(6)),
                    "all_outputs_finite": all(row["metrics"]["nonfinite_outputs"] == 0 for row in configuration_results),
                    "fp32_profile_migraphx_positive_cpu_zero": fp32_runtime["profile"]["passed"],
                    "all_segment_sessions_strict_no_fallback": True,
                },
                "claim_boundary": "Configuration selection/admission only; diagnostic block timings are not end-to-end performance claims.",
            }
        )
        if not all(result["gates"].values()):
            raise RuntimeError(f"Stage-1 configuration evaluation gates failed: {result['gates']}")
        write_json(result_path, result)
        return 0
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
