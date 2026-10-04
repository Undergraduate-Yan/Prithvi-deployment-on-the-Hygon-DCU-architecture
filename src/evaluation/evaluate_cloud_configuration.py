#!/usr/bin/env python3
'Research implementation: evaluate cloud configuration.'
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort


MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
CLASS_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--cache", action="append", type=Path)
    parser.add_argument("--provider", choices=("cpu", "migraphx"), required=True)
    parser.add_argument("--deployment-validation-root", required=True, type=Path)
    parser.add_argument(
        "--indices-file",
        type=Path,
        help="Optional JSON array of deployment-validation ordinals for a CPU-only selection screen.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--role", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    valid = (labels != IGNORE_INDEX) & (labels >= 0) & (labels < len(CLASS_NAMES))
    encoded = labels[valid].astype(np.int64) * len(CLASS_NAMES) + predictions[valid].astype(np.int64)
    return np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def metrics(confusion: np.ndarray) -> dict[str, object]:
    values = confusion.astype(np.float64)
    tp = np.diag(values)
    reference = values.sum(axis=1)
    predicted = values.sum(axis=0)
    union = reference + predicted - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    f1_den = reference + predicted
    f1 = np.divide(2 * tp, f1_den, out=np.full_like(tp, np.nan), where=f1_den > 0)
    return {
        "confusion_matrix": confusion.astype(np.int64).tolist(),
        "class_support_pixels": {CLASS_NAMES[i]: int(reference[i]) for i in range(len(CLASS_NAMES))},
        "per_class_iou": {CLASS_NAMES[i]: float(iou[i]) for i in range(len(CLASS_NAMES))},
        "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(len(CLASS_NAMES))},
        "mIoU": float(np.nanmean(iou)),
        "MacroF1": float(np.nanmean(f1)),
        "PixelAccuracy": float(tp.sum() / values.sum()),
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True)
    if args.provider == "migraphx" and (not args.cache or len(args.cache) != len(args.model)):
        raise ValueError("MIGraphX requires one frozen cache per model")
    dv_root = args.deployment_validation_root.resolve(strict=True)
    inputs_path = dv_root / "inputs.normalized.f32.npy"
    labels_path = dv_root / "labels.u8.npy"
    inputs = np.load(inputs_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if inputs.shape != (1280, 6, 224, 224) or inputs.dtype != np.float32:
        raise RuntimeError(f"DV input contract drift: {inputs.shape} {inputs.dtype}")
    if labels.shape != (1280, 224, 224) or labels.dtype != np.uint8:
        raise RuntimeError(f"DV label contract drift: {labels.shape} {labels.dtype}")
    selection = None
    if args.indices_file is not None:
        if args.provider != "cpu":
            raise RuntimeError("--indices-file is restricted to CPU selection screens")
        ordinals = json.loads(args.indices_file.read_text(encoding="utf-8"))
        if not isinstance(ordinals, list) or not ordinals or any(not isinstance(value, int) for value in ordinals):
            raise RuntimeError("--indices-file must contain a non-empty JSON integer array")
        if len(set(ordinals)) != len(ordinals) or min(ordinals) < 0 or max(ordinals) >= 1280:
            raise RuntimeError("selection ordinals are duplicated or out of range")
        selection = np.asarray(ordinals, dtype=np.int64)
        inputs = inputs[selection]
        labels = labels[selection]

    sessions = []
    session_rows = []
    provider_name = MGX if args.provider == "migraphx" else CPU
    for index, model_path in enumerate(args.model):
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1 if args.provider == "migraphx" else 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if args.provider == "migraphx" else ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        cache_identity = None
        if args.provider == "migraphx":
            cache = args.cache[index].resolve(strict=True)
            cache_identity = identity(cache)
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache)
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache)
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.enable_profiling = True
            options.profile_file_prefix = str(args.output_root / f"session_{index:02d}_profile")
            providers: list[object] = [(MGX, {"device_id": 0})]
        else:
            providers = [CPU]
        started = time.monotonic()
        session = ort.InferenceSession(str(model_path.resolve(strict=True)), sess_options=options, providers=providers)
        if args.provider == "migraphx":
            session.disable_fallback()
            if session.get_providers()[0] != MGX:
                raise RuntimeError(f"Session {index} MIGraphX placement drift")
        session_rows.append({
            "ordinal": index,
            "model": identity(model_path),
            "cache": cache_identity,
            "inputs": [item.name for item in session.get_inputs()],
            "outputs": [item.name for item in session.get_outputs()],
            "providers": session.get_providers(),
            "session_create_seconds": time.monotonic() - started,
        })
        sessions.append(session)

    predictions = np.empty((len(inputs), 224, 224), dtype=np.uint8)
    non_finite = 0
    started = time.monotonic()
    for sample_index in range(len(inputs)):
        state: dict[str, np.ndarray] = {"input": np.ascontiguousarray(inputs[sample_index : sample_index + 1])}
        for session in sessions:
            input_names = [item.name for item in session.get_inputs()]
            output_names = [item.name for item in session.get_outputs()]
            outputs = session.run(output_names, {name: state[name] for name in input_names})
            state.update(zip(output_names, outputs))
        logits = state["logits"]
        non_finite += int(logits.size - np.count_nonzero(np.isfinite(logits)))
        predictions[sample_index] = np.argmax(np.nan_to_num(logits, nan=-np.inf), axis=1)[0].astype(np.uint8)
        if (sample_index + 1) % 32 == 0:
            print(json.dumps({"event": "phase7r_full_dv", "role": args.role, "finished": sample_index + 1, "total": len(inputs), "time_utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True), flush=True)
    inference_seconds = time.monotonic() - started

    provider_events: Counter[str] = Counter()
    profile_rows = []
    if args.provider == "migraphx":
        for index, session in enumerate(sessions):
            profile = Path(session.end_profiling()).resolve(strict=True)
            events = json.loads(profile.read_text(encoding="utf-8"))
            counts = Counter(str(event["args"]["provider"]) for event in events if event.get("args", {}).get("provider"))
            provider_events.update(counts)
            profile_rows.append({"ordinal": index, "profile": identity(profile), "provider_event_counts": dict(counts)})
    result = {
        "schema": "phase7r_full_deployment_validation_execution_v1",
        "status": "PASS" if non_finite == 0 and (args.provider == "cpu" or (provider_events[MGX] > 0 and provider_events[CPU] == 0)) else "FAIL",
        "role": args.role,
        "scope": "full 1,280-scene deployment-validation" if selection is None else "frozen deployment-validation CPU selection screen",
        "formal_payload_accessed": False,
        "provider": provider_name,
        "sample_count": len(inputs),
        "session_count": len(sessions),
        "non_finite_logits": non_finite,
        "inference_seconds": inference_seconds,
        "samples_per_second": len(inputs) / inference_seconds,
        "task_metrics": metrics(confusion_matrix(labels, predictions)),
        "provider_event_counts": dict(provider_events),
        "sessions": session_rows,
        "profiles": profile_rows,
        "inputs": identity(inputs_path),
        "labels": identity(labels_path),
        "selection_indices": selection.tolist() if selection is not None else None,
        "selection_indices_file": identity(args.indices_file) if args.indices_file is not None else None,
    }
    predictions_path = args.output_root / "predictions.u8.npy"
    np.save(predictions_path, predictions, allow_pickle=False)
    result["predictions"] = identity(predictions_path)
    (args.output_root / "full_dv_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "role", "sample_count", "session_count", "non_finite_logits", "inference_seconds", "samples_per_second", "task_metrics", "provider_event_counts")}, ensure_ascii=False, indent=2), flush=True)
    del sessions
    gc.collect()
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
