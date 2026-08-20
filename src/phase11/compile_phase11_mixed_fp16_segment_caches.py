#!/usr/bin/env python3
"""Compile and place-gate the four shared FP16 segment MXR caches on K100."""
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
from typing import Any

import numpy as np


SCHEMA = "phase11_mixed_fp16_shared_cache_compile_v1"
MANIFEST_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
EXPECTED_BLOCKS = (0, 14, 17, 18)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    row = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (row["size_bytes"], row["sha256"]) != expected:
        raise RuntimeError(f"identity drift for {path}: {row}; expected={expected}")
    return row


def resolve_manifest_path(manifest: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(strict=True)


def require_under(path: Path, root: Path, role: str, strict: bool = True) -> None:
    path = path.resolve(strict=strict)
    root = root.resolve(strict=True)
    if Path(os.path.commonpath((str(root), str(path)))) != root:
        raise RuntimeError(f"{role} escapes the authorized build root: {path}")


def session_options(ort: Any, profile_prefix: Path, strict: bool) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    if strict:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    return options


def deterministic_input(shape: list[int], block_index: int) -> np.ndarray:
    if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise RuntimeError(f"input shape is not fully static: {shape}")
    generator = np.random.default_rng(20_260_818 + block_index)
    if block_index == 0:
        # Reflectance-like values after the frozen 1e-4 input scaling contract.
        value = generator.uniform(0.0, 1.2, size=shape)
    else:
        value = generator.normal(0.0, 1.0, size=shape)
    return np.ascontiguousarray(value, dtype=np.float32)


def comparison(cpu: np.ndarray, migraphx: np.ndarray) -> dict[str, Any]:
    difference = np.abs(cpu.astype(np.float64) - migraphx.astype(np.float64))
    return {
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p99_abs": float(np.percentile(difference, 99)),
        "cpu_finite": bool(np.isfinite(cpu).all()),
        "migraphx_finite": bool(np.isfinite(migraphx).all()),
    }


def profile_counts(path: Path) -> dict[str, int]:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return dict(counts)


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    path = path.resolve(strict=True)
    bundle_root = path.parent.parent.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("status") != "created_static_pass":
        raise RuntimeError("construction manifest schema/status drift")
    if payload.get("candidate_id") != "M4" or payload.get("fp16_backbone_blocks") != list(EXPECTED_BLOCKS):
        raise RuntimeError("cache compiler requires the complete M4 construction manifest")
    segments = payload.get("segments", [])
    if len(segments) != 25 or [row.get("index") for row in segments] != list(range(25)):
        raise RuntimeError("construction manifest segment order drift")
    selected = {row["index"]: row for row in segments if row.get("precision") == "fp16"}
    if tuple(sorted(selected)) != EXPECTED_BLOCKS:
        raise RuntimeError("construction manifest FP16 segment set drift")
    for index, row in selected.items():
        if row.get("cache") is not None or row.get("cache_identity") is not None:
            raise RuntimeError(f"FP16 construction segment {index} unexpectedly has a cache")
        if row.get("cache_action") != "compile_new_fp16":
            raise RuntimeError(f"FP16 construction segment {index} cache action drift")
        model_path = resolve_manifest_path(path, str(row["model"]))
        require_under(model_path, bundle_root, f"FP16 model {index}")
        identity(
            model_path,
            (int(row["model_identity"]["size_bytes"]), str(row["model_identity"]["sha256"])),
        )
        inputs = row.get("inputs", [])
        outputs = row.get("outputs", [])
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"FP16 segment {index} must have one input and one output")
        if inputs[0].get("elem_type") != 1 or outputs[0].get("elem_type") != 1:
            raise RuntimeError(f"FP16 segment {index} external I/O must be float32")
    return payload, identity(path), selected


def compile_one(
    ort: Any,
    manifest_path: Path,
    row: dict[str, Any],
    cache_root: Path,
    output_root: Path,
    device_id: int,
) -> dict[str, Any]:
    index = int(row["index"])
    model_path = resolve_manifest_path(manifest_path, str(row["model"]))
    model_ident = identity(
        model_path,
        (int(row["model_identity"]["size_bytes"]), str(row["model_identity"]["sha256"])),
    )
    feed = deterministic_input(list(row["inputs"][0]["shape"]), index)
    input_name = str(row["inputs"][0]["name"])
    output_name = str(row["outputs"][0]["name"])

    cpu_session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options(ort, output_root / f"segment_{index:02d}_cpu_profile", strict=False),
        providers=[CPU],
    )
    cpu_output = np.ascontiguousarray(cpu_session.run([output_name], {input_name: feed})[0])
    cpu_profile = Path(cpu_session.end_profiling()).resolve(strict=True)
    del cpu_session
    gc.collect()

    cache_path = cache_root / f"segment_{index:02d}_fp16.mxr"
    if cache_path.exists():
        raise FileExistsError(cache_path)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache_path.resolve(strict=False))
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache_path.resolve(strict=False))
    profile_prefix = output_root / f"segment_{index:02d}_migraphx_profile"
    creation_started = perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options(ort, profile_prefix, strict=True),
        providers=[(MGX, {"device_id": device_id})],
    )
    session.disable_fallback()
    creation_seconds = perf_counter() - creation_started
    if session.get_providers()[0] != MGX:
        raise RuntimeError(f"MIGraphX provider priority drift at segment {index}")

    device_input = ort.OrtValue.ortvalue_from_numpy(feed, "cuda", device_id)
    binding = session.io_binding()
    binding.bind_ortvalue_input(input_name, device_input)
    binding.bind_output(output_name, "cuda", device_id)
    binding.synchronize_inputs()
    inference_started = perf_counter()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    first_inference_seconds = perf_counter() - inference_started
    device_outputs = binding.get_outputs()
    if len(device_outputs) != 1 or device_outputs[0].device_name() != "cuda":
        raise RuntimeError(f"device output contract drift at FP16 segment {index}")
    migraphx_output = np.ascontiguousarray(device_outputs[0].numpy())
    profile_path = Path(session.end_profiling()).resolve(strict=True)
    counts = profile_counts(profile_path)
    del binding, session, device_input, device_outputs
    gc.collect()

    if not cache_path.is_file() or cache_path.stat().st_size <= 0:
        raise RuntimeError(f"compiled cache missing for FP16 segment {index}")
    compared = comparison(cpu_output, migraphx_output)
    essential_gates = {
        "cache_created_nonempty": cache_path.is_file() and cache_path.stat().st_size > 0,
        "migraphx_events_positive": counts.get(MGX, 0) > 0,
        "cpu_events_zero_in_migraphx_profile": counts.get(CPU, 0) == 0,
        "cpu_output_finite": compared["cpu_finite"],
        "migraphx_output_finite": compared["migraphx_finite"],
        "device_output_cuda_alias": True,
    }
    diagnostic_strict_gates = {
        "mae_le_1e_3": compared["mae"] <= 1e-3,
        "max_abs_le_5e_2": compared["max_abs"] <= 5e-2,
    }
    if not all(essential_gates.values()):
        raise RuntimeError(f"essential cache/placement gate failed for FP16 segment {index}: {essential_gates}")
    return {
        "index": index,
        "label": row["label"],
        "precision": "fp16",
        "model": model_ident,
        "cache": identity(cache_path),
        "cpu_profile": identity(cpu_profile),
        "migraphx_profile": {
            **identity(profile_path),
            "provider_event_counts": counts,
        },
        "runtime": {
            "migraphx_session_creation_seconds": creation_seconds,
            "first_inference_seconds_diagnostic_only": first_inference_seconds,
            "input_shape": list(feed.shape),
            "input_dtype": str(feed.dtype),
            "output_shape": list(migraphx_output.shape),
            "output_dtype": str(migraphx_output.dtype),
        },
        "comparison_cpu_vs_migraphx": compared,
        "essential_gates": essential_gates,
        "diagnostic_strict_gates_not_an_admission_precondition": diagnostic_strict_gates,
        "cache_and_placement_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    manifest_path = args.m4_manifest.resolve(strict=True)
    bundle_root = manifest_path.parent.parent.resolve(strict=True)
    require_under(args.cache_root, bundle_root, "FP16 cache root", strict=False)
    require_under(args.output_dir, bundle_root, "FP16 cache evidence root", strict=False)
    if args.cache_root.exists() or args.output_dir.exists():
        raise FileExistsError("cache-root and output-dir must both be new")
    args.cache_root.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "claims": {
            "four_fp16_caches_compiled": False,
            "four_fp16_segments_strict_migraphx_placement": False,
            "four_fp16_outputs_finite": False,
            "strict_numeric_equivalence": False,
            "task_accuracy_90": False,
            "performance": False,
            "native_fp16_kernel_verified": False,
        },
    }
    try:
        _, manifest_ident, selected = load_manifest(manifest_path)
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError(
                f"vendor runtime drift: ort={ort.__version__}, providers={ort.get_available_providers()}"
            )
        rows = []
        for index in EXPECTED_BLOCKS:
            row = compile_one(
                ort=ort,
                manifest_path=args.m4_manifest.resolve(strict=True),
                row=selected[index],
                cache_root=args.cache_root.resolve(),
                output_root=args.output_dir.resolve(),
                device_id=args.device_id,
            )
            rows.append(row)
            print(f"FP16 cache {index:02d} compiled and placement-gated", flush=True)
        strict_numeric = all(
            all(row["diagnostic_strict_gates_not_an_admission_precondition"].values())
            for row in rows
        )
        result.update(
            {
                "status": "passed",
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "provider": MGX,
                    "device_id": args.device_id,
                    "cache_mode": "save_new_compiled_model",
                    "intersegment_io_contract": "FP32 device OrtValue; full chain tested later",
                },
                "construction_manifest": manifest_ident,
                "segments": rows,
                "evidence_boundary": {
                    "random_deterministic_segment_inputs_used": True,
                    "cache_and_provider_placement_only": True,
                    "strict_numeric_gates_are_diagnostic_only": True,
                    "end_to_end_mixed_pipeline_not_tested": True,
                    "task_accuracy_not_tested": True,
                    "performance_not_measured": True,
                },
            }
        )
        result["claims"].update(
            {
                "four_fp16_caches_compiled": len(rows) == 4,
                "four_fp16_segments_strict_migraphx_placement": True,
                "four_fp16_outputs_finite": True,
                "strict_numeric_equivalence": strict_numeric,
            }
        )
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    result_path = args.output_dir / "cache_compile_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
