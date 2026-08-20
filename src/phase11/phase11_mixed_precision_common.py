#!/usr/bin/env python3
"""Shared, identity-locked helpers for Phase 11 M0--M5 experiments.

The helpers deliberately keep task-level admission separate from strict numeric
equivalence.  Importing this module never changes the frozen M0 strict-failure
record.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
FINAL_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_cache_final_v1"
RESULT_SCHEMA_PREFIX = "phase11_mixed_precision"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
EXPECTED_ORT_VERSION = "1.19.2"
EXPECTED_SEGMENTS = 25
EXPECTED_SAMPLES = 90
EXPECTED_VALID_PIXELS = 3_927_398
RETAIN_AFTER = (5, 11, 17, 23)
ALLOWED_PRECISIONS = {"fp16", "int8_qdq", "fp32_compat_barrier"}
EXPECTED_VARIANTS = {
    "M0": (),
    "M1": (0,),
    "M2": (0, 14),
    "M3": (0, 14, 17),
    "M4": (0, 14, 17, 18),
    "M5": (0, 14, 15, 16, 17, 18, 19, 20, 21),
}
TASK_GATES = {
    "miou_drop_pp_max": 0.5,
    "water_iou_drop_pp_max": 1.0,
    "corrected_boundary_water_iou_drop_pp_max": 1.0,
    "prediction_agreement_min": 0.995,
    "nonfinite_outputs_max": 0,
}
STRICT_DIAGNOSTIC_GATES = {
    "mae_max": 1.0e-3,
    "max_abs_max": 5.0e-2,
    "pixel_class_agreement_min": 0.999,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def identity(path: Path, expected: dict[str, Any] | tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    row = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None:
        if isinstance(expected, dict):
            expected_pair = (int(expected["size_bytes"]), str(expected["sha256"]))
        else:
            expected_pair = (int(expected[0]), str(expected[1]))
        if (row["size_bytes"], row["sha256"]) != expected_pair:
            raise RuntimeError(f"artifact identity drift: {row}; expected={expected_pair}")
    return row


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return identity(path)


def _contained(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def resolve_payload(bundle: Path, manifest_parent: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"manifest payload path must be non-empty and relative: {relative!r}")
    root = bundle.resolve(strict=True)
    path = (manifest_parent.resolve(strict=True) / relative).resolve(strict=True)
    if not _contained(root, path):
        raise RuntimeError(f"payload escapes bundle: {relative!r}")
    return path


def _contract_rows(value: Any, context: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty list")
    normalized = []
    for item in value:
        if isinstance(item, str):
            normalized.append({"name": item, "elem_type": None, "shape": None})
        elif isinstance(item, dict):
            name = item.get("name")
            shape = item.get("shape")
            elem_type = item.get("elem_type")
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"invalid {context} name: {item!r}")
            if not isinstance(shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise RuntimeError(f"invalid {context} shape: {item!r}")
            if int(elem_type) != 1:
                raise RuntimeError(f"{context} boundary is not ONNX float32: {item!r}")
            normalized.append({"name": name, "elem_type": int(elem_type), "shape": list(shape)})
        else:
            raise RuntimeError(f"invalid {context} row: {item!r}")
    if len({item["name"] for item in normalized}) != len(normalized):
        raise RuntimeError(f"duplicate names in {context}")
    return normalized


def load_candidate(bundle: Path, manifest_path: Path) -> tuple[dict, dict, list[dict]]:
    bundle = bundle.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    if not _contained(bundle, manifest_path):
        raise RuntimeError("candidate manifest must be inside --bundle")
    manifest_identity = identity(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {SCHEMA, FINAL_SCHEMA}:
        raise RuntimeError(f"candidate manifest schema drift: {manifest.get('schema')!r}")
    if manifest.get("status") != "cache_finalized_static_pass":
        raise RuntimeError(f"candidate construction status is not admissible: {manifest.get('status')!r}")
    if manifest.get("schema") == SCHEMA and manifest.get("manifest_stage") != FINAL_SCHEMA:
        raise RuntimeError("candidate manifest was not finalized after all 25 caches were admitted")
    candidate_id = str(manifest.get("candidate_id", ""))
    if candidate_id not in EXPECTED_VARIANTS:
        raise RuntimeError(f"unsupported candidate_id: {candidate_id!r}")
    if int(manifest.get("segment_count", -1)) != EXPECTED_SEGMENTS:
        raise RuntimeError("candidate segment_count must be exactly 25")
    if tuple(manifest.get("retain_encoder_outputs_after_segments", ())) != RETAIN_AFTER:
        raise RuntimeError("retained encoder-output contract drift")
    if manifest.get("head_precision") != "fp32_compat_barrier":
        raise RuntimeError("mixed candidates must use the repaired FP32 compatibility head")

    rows = manifest.get("segments")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SEGMENTS:
        raise RuntimeError("manifest must contain exactly 25 ordered segments")
    if [row.get("index") for row in rows] != list(range(EXPECTED_SEGMENTS)):
        raise RuntimeError("manifest segment order/index drift")

    fp16_blocks = tuple(int(value) for value in manifest.get("fp16_backbone_blocks", ()))
    int8_blocks = tuple(int(value) for value in manifest.get("int8_backbone_blocks", ()))
    if fp16_blocks != EXPECTED_VARIANTS[candidate_id]:
        raise RuntimeError(
            f"{candidate_id} FP16 mapping drift: {fp16_blocks}; expected={EXPECTED_VARIANTS[candidate_id]}"
        )
    expected_int8 = tuple(index for index in range(24) if index not in fp16_blocks)
    if int8_blocks != expected_int8:
        raise RuntimeError(f"{candidate_id} INT8 mapping drift")

    resolved = []
    for index, row in enumerate(rows):
        expected_label = (
            f"encoder_block_{index:02d}" if index < 24 else "upernet_decoder_head_fpn4barrier"
        )
        if row.get("label") != expected_label:
            raise RuntimeError(
                f"segment label drift at {index}: {row.get('label')!r}; expected={expected_label!r}"
            )
        precision = row.get("precision")
        if precision not in ALLOWED_PRECISIONS:
            raise RuntimeError(f"invalid precision at segment {index}: {precision!r}")
        expected_precision = (
            "fp32_compat_barrier"
            if index == 24
            else "fp16"
            if index in fp16_blocks
            else "int8_qdq"
        )
        if precision != expected_precision:
            raise RuntimeError(
                f"precision mapping drift at segment {index}: {precision}; expected={expected_precision}"
            )
        model = resolve_payload(bundle, manifest_path.parent, row.get("model"))
        model_identity = identity(model, row.get("model_identity"))
        cache_value = row.get("cache")
        cache_meta = row.get("cache_identity")
        if cache_value is None or cache_meta is None:
            raise RuntimeError(
                f"segment {index} has no compiled cache; build/admit caches before K100 evaluation"
            )
        cache = resolve_payload(bundle, manifest_path.parent, cache_value)
        cache_identity = identity(cache, cache_meta)
        inputs = _contract_rows(row.get("inputs"), f"segment {index} inputs")
        outputs = _contract_rows(row.get("outputs"), f"segment {index} outputs")
        resolved.append(
            {
                **row,
                "model_path": model,
                "cache_path": cache,
                "verified_model_identity": model_identity,
                "verified_cache_identity": cache_identity,
                "input_contracts": inputs,
                "output_contracts": outputs,
                "input_names": [item["name"] for item in inputs],
                "output_names": [item["name"] for item in outputs],
            }
        )
    return manifest, manifest_identity, resolved


def _onnx_shape(value_info) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dimensions.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dimensions.append(str(dim.dim_param))
        else:
            dimensions.append(None)
    return dimensions


def validate_onnx_contracts(rows: list[dict]) -> list[dict]:
    import onnx

    checked = []
    for row in rows:
        model = onnx.load(str(row["model_path"]), load_external_data=False)
        onnx.checker.check_model(model)
        graph_inputs = {item.name: item for item in model.graph.input}
        graph_outputs = {item.name: item for item in model.graph.output}
        expected_inputs = {item["name"]: item for item in row["input_contracts"]}
        expected_outputs = {item["name"]: item for item in row["output_contracts"]}
        if set(graph_inputs) != set(expected_inputs):
            raise RuntimeError(f"ONNX input-name drift at segment {row['index']}")
        if set(graph_outputs) != set(expected_outputs):
            raise RuntimeError(f"ONNX output-name drift at segment {row['index']}")
        tensors = list(graph_inputs.values()) + list(graph_outputs.values())
        non_float = [item.name for item in tensors if item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT]
        if non_float:
            raise RuntimeError(
                f"all intersegment boundaries must be float32; segment {row['index']}: {non_float}"
            )
        for name, value_info in {**graph_inputs, **graph_outputs}.items():
            expected = expected_inputs.get(name, expected_outputs.get(name))
            if expected["elem_type"] is not None and value_info.type.tensor_type.elem_type != expected["elem_type"]:
                raise RuntimeError(f"declared dtype drift at segment {row['index']}/{name}")
            actual_shape = _onnx_shape(value_info)
            if expected["shape"] is not None and actual_shape != expected["shape"]:
                raise RuntimeError(f"declared shape drift at segment {row['index']}/{name}")
        checked.append(
            {
                "index": row["index"],
                "inputs": {name: _onnx_shape(item) for name, item in graph_inputs.items()},
                "outputs": {name: _onnx_shape(item) for name, item in graph_outputs.items()},
                "onnx_checker_passed": True,
                "boundary_dtype": "float32",
            }
        )
    if list(checked[0]["inputs"].values()) != [[1, 6, 224, 224]]:
        raise RuntimeError("first segment input must be static float32[1,6,224,224]")
    head_outputs = checked[-1]["outputs"]
    if head_outputs.get("logits") != [1, 2, 224, 224]:
        raise RuntimeError("head logits output contract drift")
    retained_names = {next(iter(checked[index]["outputs"])) for index in RETAIN_AFTER}
    if set(checked[-1]["inputs"]) != retained_names:
        raise RuntimeError("head inputs do not exactly match retained encoder outputs")
    return checked


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_runtime():
    import onnxruntime as ort

    if ort.__version__ != EXPECTED_ORT_VERSION:
        raise RuntimeError(f"onnxruntime identity drift: {ort.__version__}")
    if MGX not in ort.get_available_providers():
        raise RuntimeError("MIGraphXExecutionProvider is unavailable")
    return ort


def session_options(ort, profile_prefix: Path | None = None):
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


def cpu_session_options(ort):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return options


def set_cache_environment(cache: Path) -> None:
    cache = cache.resolve(strict=True)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache)


def create_migraphx_session(ort, row: dict, profile_prefix: Path | None = None):
    set_cache_environment(row["cache_path"])
    session = ort.InferenceSession(
        str(row["model_path"]),
        sess_options=session_options(ort, profile_prefix),
        providers=[(MGX, {"device_id": 0})],
    )
    session.disable_fallback()
    if not session.get_providers() or session.get_providers()[0] != MGX:
        raise RuntimeError(f"provider priority drift at segment {row['index']}")
    return session


def parse_profile(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    migraphx = int(counts.get(MGX, 0))
    cpu = int(counts.get(CPU, 0))
    return {
        "provider_event_counts": dict(sorted(counts.items())),
        "migraphx_events_positive": migraphx > 0,
        "cpu_events_zero": cpu == 0,
        "passed": migraphx > 0 and cpu == 0,
    }


def choose_primary_output(index: int, output_map: dict[str, Any]):
    if index == 24:
        if "logits" not in output_map:
            raise RuntimeError("head output does not contain logits")
        return output_map["logits"]
    return output_map[next(iter(output_map))]


def run_cpu_pipeline(ort, rows: list[dict], raw: np.ndarray) -> np.ndarray:
    current = np.ascontiguousarray(raw, dtype=np.float32)
    retained: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        session = ort.InferenceSession(
            str(row["model_path"]),
            sess_options=cpu_session_options(ort),
            providers=[CPU],
        )
        if index < 24:
            feed = {session.get_inputs()[0].name: current}
        else:
            names = {item.name for item in session.get_inputs()}
            if names != set(retained):
                raise RuntimeError("CPU head retained-feature contract drift")
            feed = {name: retained[name] for name in names}
        names = [item.name for item in session.get_outputs()]
        values = session.run(names, feed)
        output_map = dict(zip(names, values, strict=True))
        current = np.ascontiguousarray(choose_primary_output(index, output_map), dtype=np.float32)
        if index in RETAIN_AFTER:
            retained[names[0]] = current
        if not np.isfinite(current).all():
            raise RuntimeError(f"non-finite CPU output at segment {index}")
        del session
    return current


def compare_logits(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise RuntimeError(f"logit shape mismatch: {reference.shape} != {candidate.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    left = np.argmax(reference, axis=1)
    right = np.argmax(candidate, axis=1)
    return {
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "pixel_class_agreement": float(np.mean(left == right)),
        "changed_pixels": int(np.count_nonzero(left != right)),
        "all_finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def strict_diagnostic_gates(comparison: dict) -> dict:
    return {
        "mae_le_1e_3": comparison["mae"] <= STRICT_DIAGNOSTIC_GATES["mae_max"],
        "max_abs_le_5e_2": comparison["max_abs"] <= STRICT_DIAGNOSTIC_GATES["max_abs_max"],
        "pixel_class_agreement_ge_99_9pct": comparison["pixel_class_agreement"]
        >= STRICT_DIAGNOSTIC_GATES["pixel_class_agreement_min"],
        "all_finite": comparison["all_finite"],
    }


def load_raw_sample(path: Path) -> np.ndarray:
    path = path.resolve(strict=True)
    suffix = path.suffix.lower()
    arrays: list[np.ndarray] = []
    if suffix == ".npy":
        arrays.append(np.load(path, allow_pickle=False))
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as pack:
            preferred = ("input", "image", "raw", "raw_input", "raw_inputs")
            key = next((name for name in preferred if name in pack), None)
            if key is None:
                key = next(iter(pack.files), None)
            if key is None:
                raise RuntimeError("empty NPZ sample")
            arrays.append(pack[key])
    elif suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)

        def visit(value) -> None:
            if torch.is_tensor(value):
                arrays.append(value.detach().cpu().numpy())
            elif isinstance(value, dict):
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    visit(nested)

        visit(obj)
    else:
        raise RuntimeError("sample must be .npy, .npz, .pt, or .pth")
    matches = []
    for array in arrays:
        if array.ndim == 3 and array.shape == (6, 224, 224):
            matches.append(array[None])
        elif array.ndim == 4 and tuple(array.shape[1:]) == (6, 224, 224):
            matches.append(array[:1])
    if not matches:
        raise RuntimeError("sample does not contain a 1x6x224x224-compatible tensor")
    raw = np.ascontiguousarray(matches[0], dtype=np.float32)
    if raw.shape != (1, 6, 224, 224) or not np.isfinite(raw).all():
        raise RuntimeError("sample input contract failure")
    return raw


def validate_result_candidate(result: dict, manifest_identity: dict, candidate_id: str) -> None:
    lineage = result.get("candidate_lineage", {})
    if lineage.get("candidate_id") != candidate_id:
        raise RuntimeError("candidate result ID drift")
    recorded = lineage.get("manifest", {})
    pair = (recorded.get("size_bytes"), recorded.get("sha256"))
    expected = (manifest_identity["size_bytes"], manifest_identity["sha256"])
    if pair != expected:
        raise RuntimeError("candidate result manifest identity drift")


def load_json(path: Path) -> dict:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def statistics(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() or np.min(array) <= 0:
        raise RuntimeError("latencies must be a non-empty positive finite vector")
    return {
        "count": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(np.mean(array)),
        "std_ms": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "throughput_samples_per_second": float(array.size * 1000.0 / np.sum(array)),
        "throughput_formula": "N*1000/sum(latencies_ms), batch=1",
    }


def cv_percent(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all() or np.mean(array) <= 0:
        raise RuntimeError("CV requires at least two positive finite values")
    return float(np.std(array, ddof=1) / np.mean(array) * 100.0)


def parse_schedule(value: str) -> tuple[str, ...]:
    schedule = tuple(item.strip() for item in value.split(",") if item.strip())
    if not schedule or len(set(schedule)) != len(schedule):
        raise RuntimeError("candidate schedule must be a non-empty unique comma-separated list")
    if any(item not in EXPECTED_VARIANTS for item in schedule):
        raise RuntimeError(f"schedule contains unsupported candidates: {schedule}")
    return schedule


def rotated_schedule(schedule: tuple[str, ...], trial_index: int) -> tuple[str, ...]:
    if trial_index not in (1, 2, 3):
        raise RuntimeError("trial index must be 1, 2, or 3")
    offset = (trial_index - 1) % len(schedule)
    return schedule[offset:] + schedule[:offset]


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise RuntimeError(f"unsafe identifier: {value!r}")
    return value
