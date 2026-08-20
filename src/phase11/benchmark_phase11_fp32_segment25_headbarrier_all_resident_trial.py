#!/usr/bin/env python3
"""One fresh-process performance trial for the admitted FP32 25-segment baseline.

The primary scope keeps the input, every intermediate, and the logits OrtValue on
the K100.  The secondary scope additionally performs one explicit H2D input copy
and one D2H logits copy per call.  Session construction is excluded from both.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import traceback
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import torch


SEGMENT_REPORT = (
    24_959,
    "c94692d130a0eb924290b3d56f790e166535fa3049fd9f3e72ef1fe16085c90f",
)
SAMPLE = (
    4_820_344,
    "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62",
)
CANDIDATE = (
    60_551_388,
    "593db691abcb2e07dd8df2f1fabc8315eeab08380161c93199778bc1963f299f",
)
BUILD_REPORT = (
    2_511,
    "1b288841074cda9134ff91a5e763b8116e3935a2e3124f6b69813f2def6b86d9",
)
HEAD_CACHE = (
    61_093_641,
    "e0c43b934aa235c1710468cfb76060e834e54672bf939e1c0a2a7a7096a6cb72",
)
SINGLE_RESULT = (
    33_109,
    "240479bbf29fef32165b908166c0550711384018d33e9a1a8ff75fdae3837655",
)
TEST90_RESULT = (
    39_291,
    "0904da34078801106b6b414dcc17d1b612b448a997460563fc7055c4e6988b3b",
)
RAW_SHA256 = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24))
RETAIN_AFTER = (5, 11, 17, 23)
MGX = "MIGraphXExecutionProvider"
WARMUP = 30
MEASURED = 100
SCOPES = ("model_only", "end_to_end_logits")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def load_raw(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found: list[torch.Tensor] = []

    def visit(value) -> None:
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(obj)
    if not found:
        raise RuntimeError("sample image missing")
    raw = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if raw.shape != (1, 6, 224, 224):
        raise RuntimeError(f"sample shape drift: {raw.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError("sample contains NaN/Inf")
    if hashlib.sha256(raw.tobytes(order="C")).hexdigest() != RAW_SHA256:
        raise RuntimeError("raw input identity drift")
    return raw


def session_options(ort):
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    value.intra_op_num_threads = 4
    value.inter_op_num_threads = 1
    value.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return value


def statistics(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (MEASURED,) or not np.isfinite(array).all() or np.min(array) <= 0:
        raise RuntimeError("latency array contract failure")
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
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    if reference.shape != (1, 2, 224, 224) or candidate.shape != reference.shape:
        raise RuntimeError(f"logits shape drift: reference={reference.shape}, candidate={candidate.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    left = np.argmax(reference, axis=1)
    right = np.argmax(candidate, axis=1)
    return {
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "pixel_class_agreement": float(np.mean(left == right)),
        "changed_pixels": int(np.count_nonzero(left != right)),
        "all_finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def cache_paths(root: Path, index: int) -> tuple[Path, Path]:
    base = (
        root / "cache_ln_segment00"
        if index == 0
        else root / "cache_ln_remaining" / f"segment_{index:02d}"
    )
    return base / f"segment_{index:02d}.mxr", base / "output" / "result.json"


def choose_primary(index: int, output_names: list[str], output_map: dict):
    if index < 24:
        if len(output_names) != 1:
            raise RuntimeError(f"encoder output contract drift at segment {index}: {output_names}")
        return output_map[output_names[0]]
    if "logits" in output_map:
        return output_map["logits"]
    if len(output_names) == 1:
        return output_map[output_names[0]]
    raise RuntimeError(f"head logits output is ambiguous: {output_names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--trial-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "phase11_fp32_segment25_headbarrier_performance_trial_v1",
        "status": "failed",
        "trial_index": args.trial_index,
        "claims": {
            "same_protocol_fp32_performance_trial_valid": False,
            "eligible_as_same_protocol_speedup_denominator": False,
            "provider_placement_profiled_during_timing": False,
            "deployment_ready": False,
        },
    }
    passed = False
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("runtime identity drift")

        report_path = args.root / "build_ln" / "segment25_report_deconv.json"
        candidate_path = args.candidate_root / "24_upernet_decoder_head_deconv_fpn4barrier.onnx"
        build_report_path = args.candidate_root / "build_report.json"
        head_cache_path = args.candidate_root / "single_v1" / "segment_24_fpn4barrier.mxr"
        single_result_path = args.candidate_root / "single_v1" / "output" / "result.json"
        test90_result_path = args.candidate_root / "test90_v1" / "output" / "result.json"
        identities = {
            "benchmark_script": identity(Path(__file__)),
            "segment_report": identity(report_path, SEGMENT_REPORT),
            "sample": identity(args.sample, SAMPLE),
            "candidate": identity(candidate_path, CANDIDATE),
            "candidate_build_report": identity(build_report_path, BUILD_REPORT),
            "head_cache": identity(head_cache_path, HEAD_CACHE),
            "single_result": identity(single_result_path, SINGLE_RESULT),
            "test90_result": identity(test90_result_path, TEST90_RESULT),
        }

        build_report = load_json(build_report_path)
        if build_report.get("schema") != "phase11_fp32_segment25_head_fpn4barrier_build_v1":
            raise RuntimeError("candidate build schema drift")
        if build_report.get("status") != "passed" or not all(build_report.get("gates", {}).values()):
            raise RuntimeError("candidate build prerequisite failed")
        build_candidate = build_report.get("candidate", {})
        if (int(build_candidate.get("size_bytes", -1)), str(build_candidate.get("sha256"))) != CANDIDATE:
            raise RuntimeError("candidate identity disagrees with build report")

        single = load_json(single_result_path)
        if single.get("schema") != "phase11_fp32_segment25_headbarrier_single_v1":
            raise RuntimeError("single result schema drift")
        required_single_claims = (
            "candidate_cpu_semantics_exact",
            "all_25_segments_strict_migraphx",
            "device_resident_intersegment_io",
            "task90_eligible",
        )
        if single.get("status") != "diagnostic_completed" or not all(
            single.get("claims", {}).get(key) for key in required_single_claims
        ):
            raise RuntimeError("single-sample admission prerequisite failed")
        single_cache = single.get("new_head_cache", {})
        if (int(single_cache.get("size_bytes", -1)), str(single_cache.get("sha256"))) != HEAD_CACHE:
            raise RuntimeError("head cache identity disagrees with single result")

        test90 = load_json(test90_result_path)
        if test90.get("schema") != "phase11_fp32_segment25_headbarrier_test90_v1":
            raise RuntimeError("test90 result schema drift")
        required_test90_claims = (
            "all_25_segments_strict_migraphx",
            "device_resident_intersegment_io_test90",
            "task_equivalence_vs_frozen_fp32",
            "performance_reference_eligible",
        )
        if test90.get("status") != "passed" or not all(
            test90.get("claims", {}).get(key) for key in required_test90_claims
        ):
            raise RuntimeError("test90 performance-reference admission failed")
        if not all(test90.get("gates", {}).values()):
            raise RuntimeError("one or more frozen test90 gates did not pass")

        report = load_json(report_path)
        rows = report.get("segments", [])
        if len(rows) != 25 or [row.get("label") for row in rows[:24]] != list(LABELS):
            raise RuntimeError("segment manifest/order drift")

        models: list[Path] = []
        caches: list[Path] = []
        cache_lineage: list[dict] = []
        for index, row in enumerate(rows[:24]):
            model = args.root / "build_ln" / "models" / Path(row["path"]).name
            model_identity = identity(model, (int(row["size_bytes"]), str(row["sha256"])))
            cache, cache_result_path = cache_paths(args.root, index)
            cache_result = load_json(cache_result_path)
            cache_meta = cache_result.get("compiled_cache", {})
            cache_identity = identity(
                cache, (int(cache_meta.get("size_bytes", -1)), str(cache_meta.get("sha256")))
            )
            provider_counts = cache_result.get("profile", {}).get("provider_event_counts", {})
            if int(provider_counts.get(MGX, 0)) <= 0 or int(
                provider_counts.get("CPUExecutionProvider", 0)
            ) != 0:
                raise RuntimeError(f"frozen cache placement prerequisite failed at segment {index}")
            models.append(model)
            caches.append(cache)
            cache_lineage.append(
                {
                    "index": index,
                    "label": LABELS[index],
                    "model": model_identity,
                    "cache": cache_identity,
                    "cache_result": identity(cache_result_path),
                    "provider_event_counts": provider_counts,
                }
            )
        models.append(candidate_path)
        caches.append(head_cache_path)
        cache_lineage.append(
            {
                "index": 24,
                "label": "upernet_decoder_head_fpn4barrier",
                "model": identities["candidate"],
                "cache": identities["head_cache"],
                "placement_inherited_from": identities["single_result"],
            }
        )

        sessions = []
        load_started = perf_counter()
        for index, (model, cache) in enumerate(zip(models, caches, strict=True)):
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve(strict=True))
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve(strict=True))
            session = ort.InferenceSession(
                str(model.resolve(strict=True)),
                sess_options=session_options(ort),
                providers=[(MGX, {"device_id": 0})],
            )
            session.disable_fallback()
            if session.get_providers()[0] != MGX:
                raise RuntimeError(f"provider priority drift at segment {index}")
            sessions.append(session)
        session_load_seconds = perf_counter() - load_started
        if len(sessions) != 25:
            raise RuntimeError("all 25 sessions are not concurrently resident")

        raw = load_raw(args.sample)
        device_input = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        if device_input.device_name() != "cuda":
            raise RuntimeError("initial input did not allocate on K100")

        # One untimed run allocates stable device outputs for every segment.
        allocated_outputs: list[dict] = []
        retained = {}
        current = device_input
        for index, session in enumerate(sessions):
            binding = session.io_binding()
            input_items = session.get_inputs()
            if index < 24:
                if len(input_items) != 1:
                    raise RuntimeError(f"encoder input contract drift at segment {index}")
                binding.bind_ortvalue_input(input_items[0].name, current)
            else:
                if {item.name for item in input_items} != set(retained):
                    raise RuntimeError("head retained-feature contract drift during allocation")
                for item in input_items:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                binding.bind_output(name, "cuda", 0)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            output_values = binding.get_outputs()
            if len(output_values) != len(output_names) or not all(
                value.device_name() == "cuda" for value in output_values
            ):
                raise RuntimeError(f"device output allocation failed at segment {index}")
            output_map = dict(zip(output_names, output_values, strict=True))
            allocated_outputs.append(output_map)
            current = choose_primary(index, output_names, output_map)
            if index in RETAIN_AFTER:
                retained[output_names[0]] = current
        reference_logits = np.ascontiguousarray(current.numpy(), dtype=np.float32)
        if reference_logits.shape != (1, 2, 224, 224) or not np.isfinite(reference_logits).all():
            raise RuntimeError("initial logits contract failure")

        # Bind the already allocated OrtValues once.  These bindings are reused in
        # the primary model-only scope, avoiding allocator and copy noise.
        fixed_bindings = []
        retained = {}
        current = device_input
        for index, (session, output_map) in enumerate(
            zip(sessions, allocated_outputs, strict=True)
        ):
            binding = session.io_binding()
            input_items = session.get_inputs()
            if index < 24:
                binding.bind_ortvalue_input(input_items[0].name, current)
            else:
                for item in input_items:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                binding.bind_ortvalue_output(name, output_map[name])
            binding.synchronize_inputs()
            fixed_bindings.append(binding)
            current = choose_primary(index, output_names, output_map)
            if index in RETAIN_AFTER:
                retained[output_names[0]] = current

        def run_model_only() -> None:
            for session, binding in zip(sessions, fixed_bindings, strict=True):
                session.run_with_iobinding(binding)
            fixed_bindings[-1].synchronize_outputs()

        def run_end_to_end() -> np.ndarray:
            # Explicitly include H2D allocation/copy and final D2H logits copy.
            fresh_input = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
            first_binding = sessions[0].io_binding()
            first_binding.bind_ortvalue_input(sessions[0].get_inputs()[0].name, fresh_input)
            for name, output_value in allocated_outputs[0].items():
                first_binding.bind_ortvalue_output(name, output_value)
            first_binding.synchronize_inputs()
            sessions[0].run_with_iobinding(first_binding)
            for index in range(1, 25):
                sessions[index].run_with_iobinding(fixed_bindings[index])
            fixed_bindings[-1].synchronize_outputs()
            logits_value = choose_primary(
                24,
                [item.name for item in sessions[-1].get_outputs()],
                allocated_outputs[-1],
            )
            return np.ascontiguousarray(logits_value.numpy(), dtype=np.float32)

        scope_order = (
            ("model_only", "end_to_end_logits")
            if args.trial_index in (1, 3)
            else ("end_to_end_logits", "model_only")
        )
        latencies: dict[str, list[float]] = {scope: [] for scope in SCOPES}
        comparisons = {}
        for scope in scope_order:
            last_logits = None
            for _ in range(WARMUP):
                if scope == "model_only":
                    run_model_only()
                else:
                    last_logits = run_end_to_end()
            if scope == "model_only":
                last_logits = np.ascontiguousarray(
                    choose_primary(
                        24,
                        [item.name for item in sessions[-1].get_outputs()],
                        allocated_outputs[-1],
                    ).numpy(),
                    dtype=np.float32,
                )
            warmup_comparison = compare(reference_logits, last_logits)
            if (
                not warmup_comparison["all_finite"]
                or warmup_comparison["mae"] > 1.0e-6
                or warmup_comparison["max_abs"] > 1.0e-5
            ):
                raise RuntimeError(f"{scope} warmup output drift: {warmup_comparison}")

            for measured_index in range(MEASURED):
                started_ns = perf_counter_ns()
                if scope == "model_only":
                    run_model_only()
                    measured_logits = None
                else:
                    measured_logits = run_end_to_end()
                latencies[scope].append((perf_counter_ns() - started_ns) / 1.0e6)
                if measured_logits is not None:
                    last_logits = measured_logits
                if (measured_index + 1) % 20 == 0:
                    print(
                        f"FP32 trial {args.trial_index} {scope} "
                        f"{measured_index + 1}/{MEASURED}",
                        flush=True,
                    )
            if scope == "model_only":
                last_logits = np.ascontiguousarray(
                    choose_primary(
                        24,
                        [item.name for item in sessions[-1].get_outputs()],
                        allocated_outputs[-1],
                    ).numpy(),
                    dtype=np.float32,
                )
            comparisons[scope] = {
                "warmup": warmup_comparison,
                "final": compare(reference_logits, last_logits),
            }

        artifacts = {}
        scope_statistics = {}
        for scope in SCOPES:
            latency_path = args.output_dir / f"{scope}_latencies_ms.npy"
            np.save(
                latency_path,
                np.asarray(latencies[scope], dtype=np.float64),
                allow_pickle=False,
            )
            artifacts[f"{scope}_latencies_ms"] = identity(latency_path)
            scope_statistics[scope] = statistics(latencies[scope])

        output_gates = {
            f"{scope}_final_output_stable": comparison["final"]["all_finite"]
            and comparison["final"]["mae"] <= 1.0e-6
            and comparison["final"]["max_abs"] <= 1.0e-5
            and comparison["final"]["pixel_class_agreement"] == 1.0
            for scope, comparison in comparisons.items()
        }
        gates = {
            "frozen_test90_reference_admission_passed": True,
            "session_count_eq_25": len(sessions) == 25,
            "warmup_per_scope_eq_30": WARMUP == 30,
            "measured_per_scope_eq_100": all(
                len(latencies[scope]) == MEASURED for scope in SCOPES
            ),
            "all_latencies_positive_finite": all(
                np.isfinite(np.asarray(latencies[scope])).all()
                and min(latencies[scope]) > 0
                for scope in SCOPES
            ),
            **output_gates,
        }
        passed = all(gates.values())
        try:
            import resource

            max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except ImportError:
            max_rss_kib = None

        result.update(
            {
                "status": "passed" if passed else "failed",
                "identities": identities,
                "protocol": {
                    "track": "same_protocol_fp32_reference",
                    "batch": 1,
                    "warmup_per_scope": WARMUP,
                    "measurements_per_scope": MEASURED,
                    "fresh_process_trial": True,
                    "scope_order": list(scope_order),
                    "model_only_scope": (
                        "25 resident sessions; fixed K100 input/intermediate/output OrtValues; "
                        "includes 25 Python ORT dispatches; excludes H2D, D2H, file I/O, "
                        "preprocessing, and session loading"
                    ),
                    "end_to_end_logits_scope": (
                        "host float32 input -> explicit H2D -> 25 resident sessions/device "
                        "OrtValues -> explicit D2H float32 logits; excludes file I/O, "
                        "preprocessing, and session loading"
                    ),
                },
                "runtime": {
                    "hostname": platform.node(),
                    "python": platform.python_version(),
                    "onnxruntime": ort.__version__,
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "session_load_seconds_excluded": session_load_seconds,
                    "process_max_rss_kib": max_rss_kib,
                },
                "cache_lineage": cache_lineage,
                "statistics": scope_statistics,
                "output_comparisons": comparisons,
                "gates": gates,
                "artifacts": artifacts,
                "evidence_boundary": {
                    "same_protocol_fp32_task90_admission_is_frozen": True,
                    "provider_placement_inherited_from_single_and_test90": True,
                    "timed_region_is_not_profiled": True,
                    "process_max_rss_is_not_K100_device_memory": True,
                    "speedup_is_not_computed_inside_a_single_trial": True,
                    "candidate_is_25_segments_not_monolithic": True,
                },
            }
        )
        result["claims"]["same_protocol_fp32_performance_trial_valid"] = passed
        result["claims"]["eligible_as_same_protocol_speedup_denominator"] = passed
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
