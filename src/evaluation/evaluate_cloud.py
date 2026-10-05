#!/usr/bin/env python3
'Research implementation: evaluate cloud.'
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort


MGX = "MIGraphXExecutionProvider"
CLASS_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
IGNORE = 255
STARTS = (0, 224, 448, 672, 896, 1120, 1344, 1568, 1776)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--cache", action="append", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--input-contract", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--role", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def confusion(label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    valid = (label != IGNORE) & (label < 4)
    encoded = label[valid].astype(np.int64) * 4 + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=16).reshape(4, 4)


def metrics(matrix: np.ndarray) -> dict:
    value = matrix.astype(np.float64)
    tp = np.diag(value)
    reference, predicted = value.sum(axis=1), value.sum(axis=0)
    union = reference + predicted - tp
    iou = np.divide(tp, union, out=np.full(4, np.nan), where=union > 0)
    f1 = np.divide(2 * tp, reference + predicted, out=np.full(4, np.nan), where=(reference + predicted) > 0)
    return {
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "class_support_pixels": {CLASS_NAMES[i]: int(reference[i]) for i in range(4)},
        "per_class_iou": {CLASS_NAMES[i]: float(iou[i]) for i in range(4)},
        "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(4)},
        "mIoU": float(np.nanmean(iou)),
        "MacroF1": float(np.nanmean(f1)),
        "PixelAccuracy": float(tp.sum() / value.sum()),
    }


def main() -> int:
    args = parse_args()
    if len(args.model) != 14 or len(args.cache) != 14:
        raise RuntimeError("formal same-topology candidates require exactly 14 models and 14 caches")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    payload_root = args.payload_root.resolve(strict=True)
    predictions_dir = args.output_root / "predictions"
    predictions_dir.mkdir()
    payload_manifest = json.loads((payload_root / "cloud_phase7r_formal_payload_manifest.json").read_text(encoding="utf-8"))
    if payload_manifest.get("status") not in {"PASS_MATERIALIZED_ONCE", "REPRODUCTION_PAYLOAD_COMPLETE"} or payload_manifest.get("samples") != 300:
        raise RuntimeError("formal payload manifest is not a complete one-time materialization")
    contract = json.loads(args.input_contract.read_text(encoding="utf-8"))
    mean = np.asarray(contract["normalization"]["mean"], dtype=np.float32).reshape(6, 1, 1)
    std = np.asarray(contract["normalization"]["std"], dtype=np.float32).reshape(6, 1, 1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("normalization contract drift")

    sessions = []
    session_rows = []
    for ordinal, (model, cache) in enumerate(zip(args.model, args.cache)):
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
        os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
        os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve(strict=True))
        os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve(strict=True))
        started = time.monotonic()
        session = ort.InferenceSession(str(model.resolve(strict=True)), sess_options=options, providers=[(MGX, {"device_id": 0})])
        session.disable_fallback()
        if session.get_providers()[0] != MGX:
            raise RuntimeError(f"session {ordinal} MIGraphX provider drift")
        session_rows.append({
            "ordinal": ordinal,
            "model": identity(model),
            "cache": identity(cache),
            "inputs": [item.name for item in session.get_inputs()],
            "outputs": [item.name for item in session.get_outputs()],
            "providers": session.get_providers(),
            "cpu_fallback_disabled": True,
            "session_create_seconds": time.monotonic() - started,
        })
        sessions.append(session)

    scene_rows = []
    aggregate = np.zeros((4, 4), dtype=np.int64)
    non_finite = 0
    prediction_hasher = hashlib.sha256()
    started = time.monotonic()
    for index in range(300):
        scene_path = payload_root / "scene_payloads" / f"{index:03d}.npz"
        with np.load(scene_path, allow_pickle=False) as packed:
            image = np.asarray(packed["image"], dtype=np.float32)
            label = np.asarray(packed["label"], dtype=np.uint8)
        if image.shape != (6, 2000, 2000) or label.shape != (2000, 2000):
            raise RuntimeError(f"scene {index} payload contract drift")
        normalized = np.ascontiguousarray((image - mean) / std, dtype=np.float32)
        logits_sum = np.zeros((4, 2000, 2000), dtype=np.float32)
        counts = np.zeros((2000, 2000), dtype=np.uint8)
        for y in STARTS:
            for x in STARTS:
                state = {"input": normalized[None, :, y:y + 224, x:x + 224]}
                for session in sessions:
                    names = [item.name for item in session.get_inputs()]
                    outputs = [item.name for item in session.get_outputs()]
                    values = session.run(outputs, {name: state[name] for name in names})
                    state.update(zip(outputs, values))
                patch_logits = np.asarray(state["logits"][0], dtype=np.float32)
                non_finite += int(patch_logits.size - np.count_nonzero(np.isfinite(patch_logits)))
                # Diagnostic continuation only: NaN suppresses a class in overlap
                # averaging; any non-finite tile logit makes the final status FAIL.
                logits_sum[:, y:y + 224, x:x + 224] += np.nan_to_num(patch_logits, nan=-np.inf)
                counts[y:y + 224, x:x + 224] += 1
        if np.any(counts == 0):
            raise RuntimeError(f"scene {index} tiling coverage gap")
        prediction = np.argmax(logits_sum / counts[None], axis=0).astype(np.uint8)
        prediction_hasher.update(prediction.tobytes(order="C"))
        prediction_path = predictions_dir / f"{index:03d}.u8.npy"
        np.save(prediction_path, prediction, allow_pickle=False)
        matrix = confusion(label, prediction)
        aggregate += matrix
        row_metrics = metrics(matrix)
        scene_rows.append({
            "scene_index": index,
            "payload_sha256": sha256(scene_path),
            "prediction_sha256": hashlib.sha256(prediction.tobytes(order="C")).hexdigest(),
            "prediction_file_sha256": sha256(prediction_path),
            "valid_pixels": int(np.count_nonzero(label != IGNORE)),
            "mIoU": row_metrics["mIoU"],
            "MacroF1": row_metrics["MacroF1"],
            "PixelAccuracy": row_metrics["PixelAccuracy"],
            "confusion_matrix_json": json.dumps(row_metrics["confusion_matrix"], separators=(",", ":")),
        })
        print(json.dumps({"event": "phase7r_formal", "role": args.role, "finished": index + 1, "total": 300, "time_utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True), flush=True)
        del image, label, normalized, logits_sum, counts, prediction
    elapsed = time.monotonic() - started

    scene_csv = args.output_root / "scene_metrics.csv"
    with scene_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)
    result = {
        "schema": "cloud_phase7r_formal_candidate_result_v1",
        "status": "PASS" if non_finite == 0 else "FAIL",
        "role": args.role,
        "scope": "Reproduction of the fixed 300-scene cloud protocol; 81 tiles per scene",
        "sample_count": 300,
        "patch_inference_count": 300 * 81,
        "session_count": 14,
        "cpu_fallback_disabled": True,
        "non_finite_logits": non_finite,
        "inference_seconds": elapsed,
        "patches_per_second": 300 * 81 / elapsed,
        "task_metrics": metrics(aggregate),
        "prediction_set_sha256": prediction_hasher.hexdigest(),
        "scene_metrics": identity(scene_csv),
        "payload_manifest": identity(payload_root / "cloud_phase7r_formal_payload_manifest.json"),
        "input_contract": identity(args.input_contract),
        "sessions": session_rows,
    }
    output = args.output_root / "formal_candidate_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "role", "sample_count", "patch_inference_count", "non_finite_logits", "inference_seconds", "task_metrics")}, ensure_ascii=False, indent=2))
    del sessions
    gc.collect()
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
