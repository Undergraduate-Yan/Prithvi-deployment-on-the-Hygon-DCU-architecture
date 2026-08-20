#!/usr/bin/env python3
"""Build and validate one strict MIGraphX cache for the five-part backbone."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


SEGMENT_REPORT = (
    21_342,
    "e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e",
)
CPU_PARITY = (
    14_138,
    "5c66f94a447a3d08fe0024686e2a0dc2d8e74f16d0f10951cc49e4ed81568a6f",
)
SAMPLE = (
    4_820_344,
    "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62",
)
RAW_INPUT_SHA256 = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24)) + (
    "upernet_decoder_head",
)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock(path: Path, expected: tuple[int, str]) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"identity drift: {item}")
    return item


def load_image(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def visit(value):
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(obj)
    if not found:
        raise RuntimeError("sample image missing")
    image = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if hashlib.sha256(image.tobytes(order="C")).hexdigest() != RAW_INPUT_SHA256:
        raise RuntimeError("raw input identity drift")
    return image


def session_options(ort, strict: bool = False, profile: Path | None = None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    if strict:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    return options


def record_array(value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def compare(left: np.ndarray, right: np.ndarray, logits: bool) -> dict:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    result = {
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p99_abs": float(np.percentile(difference, 99)),
    }
    if logits:
        pred_left = np.argmax(left, axis=1)
        pred_right = np.argmax(right, axis=1)
        result.update(
            {
                "pixel_class_agreement": float(np.mean(pred_left == pred_right)),
                "changed_pixels": int(np.count_nonzero(pred_left != pred_right)),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--segment-report", type=Path, required=True)
    parser.add_argument("--cpu-parity", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, choices=range(25), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "variant": "int8_backbone_compat_25_static_batch1_onnx_segments",
        "target_segment_index": args.segment_index,
        "target_segment_label": LABELS[args.segment_index],
        "claims": {
            "target_segment_strict_migraphx_admission": False,
            "all_25_segments_strict_migraphx_admission": False,
            "end_to_end_task_accuracy": False,
            "device_resident_intersegment_io": False,
            "performance": False,
            "native_int8_kernel_verified": False,
        },
    }
    try:
        identities = {
            "segment_report": lock(args.segment_report, SEGMENT_REPORT),
            "cpu_parity": lock(args.cpu_parity, CPU_PARITY),
            "sample": lock(args.sample, SAMPLE),
        }
        parity = json.loads(args.cpu_parity.read_text(encoding="utf-8"))
        if parity.get("status") != "passed" or not parity.get("claims", {}).get("cpu_sequential_parity"):
            raise RuntimeError("frozen CPU sequential parity is not passed")
        report = json.loads(args.segment_report.read_text(encoding="utf-8"))
        rows = report.get("segments", [])
        if report.get("status") != "created_static_pass" or [row.get("label") for row in rows] != list(LABELS):
            raise RuntimeError("segment report contract drift")

        segment_paths = []
        for row in rows:
            path = (args.segments_dir / "models" / Path(row["path"]).name).resolve(strict=True)
            identities[row["label"]] = lock(path, (int(row["size_bytes"]), str(row["sha256"])))
            segment_paths.append(path)
        result["identities"] = identities

        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("vendor runtime drift")
        image = load_image(args.sample)
        retained = {}
        value = image
        target_feeds = None
        for index in range(args.segment_index + 1):
            if index == args.segment_index:
                break
            session = ort.InferenceSession(
                str(segment_paths[index]),
                sess_options=session_options(ort),
                providers=[CPU],
            )
            output_name = session.get_outputs()[0].name
            value = np.asarray(session.run([output_name], {session.get_inputs()[0].name: value})[0])
            retained[output_name] = value
            del session

        target_cpu = ort.InferenceSession(
            str(segment_paths[args.segment_index]),
            sess_options=session_options(ort),
            providers=[CPU],
        )
        if args.segment_index < 24:
            target_feeds = {target_cpu.get_inputs()[0].name: value}
        else:
            target_feeds = {item.name: retained[item.name] for item in target_cpu.get_inputs()}
        output_name = target_cpu.get_outputs()[0].name
        cpu_started = perf_counter()
        cpu_output = np.asarray(target_cpu.run([output_name], target_feeds)[0])
        result["cpu_target_inference_seconds_diagnostic"] = perf_counter() - cpu_started
        del target_cpu
        gc.collect()

        args.cache.parent.mkdir(parents=True, exist_ok=True)
        if args.cache.exists():
            raise FileExistsError(args.cache)
        os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
        os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
        os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(args.cache.resolve())
        os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(args.cache.resolve())
        profile_prefix = args.output_dir / f"segment_{args.segment_index:02d}_migraphx_profile"
        started = perf_counter()
        target_migraphx = ort.InferenceSession(
            str(segment_paths[args.segment_index]),
            sess_options=session_options(ort, strict=True, profile=profile_prefix),
            providers=[(MGX, {"device_id": 0})],
        )
        target_migraphx.disable_fallback()
        result["migraphx_session_creation_seconds"] = perf_counter() - started
        started = perf_counter()
        migraphx_output = np.asarray(target_migraphx.run([output_name], target_feeds)[0])
        result["migraphx_first_inference_seconds_diagnostic"] = perf_counter() - started
        profile = Path(target_migraphx.end_profiling()).resolve(strict=True)
        events = json.loads(profile.read_text(encoding="utf-8"))
        counts = Counter(
            str(event["args"]["provider"])
            for event in events
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        )
        is_logits = args.segment_index == 24
        comparison = compare(cpu_output, migraphx_output, logits=is_logits)
        gates = {
            "all_finite": bool(np.isfinite(cpu_output).all() and np.isfinite(migraphx_output).all()),
            "migraphx_events_positive": counts[MGX] > 0,
            "cpu_events_zero": counts[CPU] == 0,
            "mae_le_1e_3": comparison["mae"] <= 1e-3,
            "max_abs_le_5e_2": comparison["max_abs"] <= 5e-2,
        }
        if is_logits:
            gates["pixel_class_agreement_ge_99_9pct"] = comparison["pixel_class_agreement"] >= 0.999
        if not args.cache.is_file() or args.cache.stat().st_size <= 0:
            raise RuntimeError("compiled cache missing")
        paired = args.output_dir / "cpu_and_migraphx_output.npz"
        arrays = {f"input_{index}": np.asarray(value) for index, value in enumerate(target_feeds.values())}
        np.savez_compressed(paired, **arrays, cpu_output=cpu_output, migraphx_output=migraphx_output)
        result.update(
            {
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "registered_providers": target_migraphx.get_providers(),
                },
                "inputs": [record_array(value) for value in target_feeds.values()],
                "cpu_output": record_array(cpu_output),
                "migraphx_output": record_array(migraphx_output),
                "comparison": comparison,
                "profile": {
                    "path": str(profile),
                    "size_bytes": profile.stat().st_size,
                    "sha256": sha256(profile),
                    "provider_event_counts": dict(counts),
                },
                "compiled_cache": {
                    "path": str(args.cache.resolve()),
                    "size_bytes": args.cache.stat().st_size,
                    "sha256": sha256(args.cache),
                },
                "paired_output": {
                    "path": str(paired),
                    "size_bytes": paired.stat().st_size,
                    "sha256": sha256(paired),
                },
                "gates": gates,
                "status": "passed" if all(gates.values()) else "failed",
            }
        )
        result["claims"]["target_segment_strict_migraphx_admission"] = all(gates.values())
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
