#!/usr/bin/env python3
"""One fresh-process latency trial for the all-resident 25-segment pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import traceback
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import torch


REPORT = (21_342, "e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e")
SAMPLE = (4_820_344, "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62")
RESIDENT = (34_455, "45433433e89ae7086a4d653dba80b92d3449c395c51a355c2241af2482f5e2d3")
RESIDENT_LOGITS = (1_607_774, "206000b34f2240ecd398c4313bde19813c4ef7de9791700ab0106050e8fad81c")
DEVICE90 = (40_581, "432f869538b6f381369db5819bb4b042cbf0221be20cd044a99463aabc4b94a4")
RAW_SHA = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS = tuple(f"encoder_block_{i:02d}" for i in range(24)) + ("upernet_decoder_head",)
RETAIN_AFTER = (5, 11, 17, 23)
MGX = "MIGraphXExecutionProvider"
WARMUP = 30
MEASURED = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lock(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_raw(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def visit(value):
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(obj)
    raw = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if hashlib.sha256(raw.tobytes()).hexdigest() != RAW_SHA:
        raise RuntimeError("raw input identity drift")
    return raw


def options(ort):
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    value.intra_op_num_threads = 4
    value.inter_op_num_threads = 1
    value.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return value


def statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(np.mean(array)),
        "std_ms": float(np.std(array, ddof=1)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
        "throughput_samples_per_second": float(array.size * 1000.0 / np.sum(array)),
        "throughput_formula": "N*1000/sum(latencies_ms), batch=1",
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    p0, p1 = np.argmax(reference, axis=1), np.argmax(candidate, axis=1)
    return {
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "pixel_class_agreement": float(np.mean(p0 == p1)),
        "changed_pixels": int(np.count_nonzero(p0 != p1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trial-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "variant": "int8_backbone_25_all_resident_device_iobinding_absolute_latency",
        "trial_index": args.trial_index,
        "claims": {
            "absolute_pipeline_latency_trial_valid": False,
            "int8_speedup_vs_fp32": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("runtime identity drift")
        segment_root = args.root / "segment25_static_build_cpu"
        report_path = segment_root / "segment_build_report.json"
        resident_root = args.root / "segment25_all_sessions_resident_single_v2" / "output"
        device90_path = args.root / "segment25_iobinding_end_to_end_test90" / "output/result.json"
        identities = {
            "benchmark_script": lock(Path(__file__)),
            "segment_report": lock(report_path, REPORT),
            "sample": lock(args.root / "sample_and_logits.pt", SAMPLE),
            "passed_all_resident_single": lock(resident_root / "result.json", RESIDENT),
            "all_resident_single_logits": lock(
                resident_root / "sequential_and_all_resident_device_logits.npz", RESIDENT_LOGITS
            ),
            "passed_device_resident_test90": lock(device90_path, DEVICE90),
        }
        resident_result = json.loads((resident_root / "result.json").read_text(encoding="utf-8"))
        device90_result = json.loads(device90_path.read_text(encoding="utf-8"))
        if resident_result.get("status") != "passed" or not resident_result.get("claims", {}).get(
            "all_25_sessions_concurrently_resident"
        ):
            raise RuntimeError("all-resident prerequisite failed")
        if device90_result.get("status") != "diagnostic_completed" or not device90_result.get(
            "task_utility_diagnostic_passed"
        ):
            raise RuntimeError("device-resident test90 prerequisite failed")
        frozen = np.load(
            resident_root / "sequential_and_all_resident_device_logits.npz", allow_pickle=False
        )
        reference_logits = np.ascontiguousarray(
            frozen["all_sessions_resident_device_logits"], dtype=np.float32
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report["segments"]
        if len(rows) != 25 or [row["label"] for row in rows] != list(LABELS):
            raise RuntimeError("segment manifest/order drift")

        sessions, cache_rows = [], []
        load_started = perf_counter()
        for index, row in enumerate(rows):
            model = segment_root / "models" / Path(row["path"]).name
            lock(model, (int(row["size_bytes"]), row["sha256"]))
            base = (
                args.root / "segment25_static_cache54g_segment0"
                if index == 0
                else args.root / "segment25_static_remaining_caches" / f"segment_{index:02d}"
            )
            frozen_result = json.loads((base / "output/result.json").read_text(encoding="utf-8"))
            cache_meta = frozen_result["compiled_cache"]
            cache = base / f"segment_{index:02d}.mxr"
            lock(cache, (int(cache_meta["size_bytes"]), cache_meta["sha256"]))
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve())
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve())
            session = ort.InferenceSession(
                str(model), sess_options=options(ort), providers=[(MGX, {"device_id": 0})]
            )
            session.disable_fallback()
            if session.get_providers()[0] != MGX:
                raise RuntimeError(f"provider priority drift at segment {index}")
            sessions.append(session)
            cache_rows.append({"index": index, "cache": cache_meta})
        load_seconds = perf_counter() - load_started
        if len(sessions) != 25:
            raise RuntimeError("all sessions not resident")

        raw = load_raw(args.root / "sample_and_logits.pt")
        input_value = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        allocated_outputs, retained = [], {}
        current = input_value
        for index, session in enumerate(sessions):
            binding = session.io_binding()
            if index < 24:
                binding.bind_ortvalue_input(session.get_inputs()[0].name, current)
            else:
                for item in session.get_inputs():
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_name = session.get_outputs()[0].name
            binding.bind_output(output_name, "cuda", 0)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            outputs = binding.get_outputs()
            if len(outputs) != 1 or outputs[0].device_name() != "cuda":
                raise RuntimeError(f"initial output allocation failed at segment {index}")
            current = outputs[0]
            allocated_outputs.append(current)
            if index in RETAIN_AFTER:
                retained[output_name] = current

        fixed_bindings, retained = [], {}
        current = input_value
        for index, (session, output_value) in enumerate(zip(sessions, allocated_outputs, strict=True)):
            binding = session.io_binding()
            if index < 24:
                binding.bind_ortvalue_input(session.get_inputs()[0].name, current)
            else:
                for item in session.get_inputs():
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_name = session.get_outputs()[0].name
            binding.bind_ortvalue_output(output_name, output_value)
            binding.synchronize_inputs()
            fixed_bindings.append(binding)
            current = output_value
            if index in RETAIN_AFTER:
                retained[output_name] = current

        def operation() -> None:
            for session, binding in zip(sessions, fixed_bindings, strict=True):
                session.run_with_iobinding(binding)
            fixed_bindings[-1].synchronize_outputs()

        for _ in range(WARMUP):
            operation()
        warmup_logits = np.ascontiguousarray(allocated_outputs[-1].numpy(), dtype=np.float32)
        warmup_comparison = compare(reference_logits, warmup_logits)
        if warmup_comparison["mae"] > 1e-6 or warmup_comparison["max_abs"] > 1e-5:
            raise RuntimeError(f"warmup output drift: {warmup_comparison}")

        latencies_ms = []
        for measured_index in range(MEASURED):
            started_ns = perf_counter_ns()
            operation()
            latencies_ms.append((perf_counter_ns() - started_ns) / 1.0e6)
            if (measured_index + 1) % 20 == 0:
                print(f"trial {args.trial_index} measured {measured_index + 1}/{MEASURED}", flush=True)
        final_logits = np.ascontiguousarray(allocated_outputs[-1].numpy(), dtype=np.float32)
        final_comparison = compare(reference_logits, final_logits)
        gates = {
            "session_count_eq_25": len(sessions) == 25,
            "warmup_count_eq_30": WARMUP == 30,
            "measured_count_eq_100": len(latencies_ms) == 100,
            "all_latencies_positive_finite": bool(
                np.isfinite(np.asarray(latencies_ms)).all() and min(latencies_ms) > 0
            ),
            "final_mae_le_1e_6": final_comparison["mae"] <= 1e-6,
            "final_max_abs_le_1e_5": final_comparison["max_abs"] <= 1e-5,
            "final_predictions_exact": final_comparison["pixel_class_agreement"] == 1.0,
        }
        passed = all(gates.values())
        raw_path = args.output_dir / "latencies_ms.npy"
        np.save(raw_path, np.asarray(latencies_ms, dtype=np.float64), allow_pickle=False)
        result.update({
            "status": "passed" if passed else "failed",
            "identities": identities,
            "protocol": {
                "batch": 1,
                "warmup": WARMUP,
                "measured": MEASURED,
                "scope": "25 resident sessions; fixed device input/intermediate/output OrtValues; includes 25 Python ORT dispatches; excludes H2D/D2H and session load",
                "fresh_process_trial": True,
                "strict_cpu_ep_fallback_disabled": True,
            },
            "software": {
                "python": platform.python_version(),
                "onnxruntime": ort.__version__,
                "torch": torch.__version__,
                "hip": torch.version.hip,
            },
            "runtime": {
                "session_load_seconds_excluded_from_timing": load_seconds,
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "cache_lineage": cache_rows,
            "statistics": statistics(latencies_ms),
            "warmup_comparison": warmup_comparison,
            "final_comparison": final_comparison,
            "gates": gates,
            "artifacts": {"latencies_ms": lock(raw_path)},
            "evidence_boundary": {
                "absolute_compatibility_pipeline_latency_only": True,
                "no_same_protocol_fp32_baseline": True,
                "no_int8_speedup_claim": True,
                "strict_logits_numeric_gate_remains_failed": True,
                "provider_placement_locked_from_prerequisites_not_profiled_during_timing": True,
                "native_int8_kernel_not_proven": True,
            },
        })
        result["claims"]["absolute_pipeline_latency_trial_valid"] = passed
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
