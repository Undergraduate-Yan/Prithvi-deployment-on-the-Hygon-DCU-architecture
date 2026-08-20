#!/usr/bin/env python3
"""One fresh-process 30+100 performance trial for one admitted candidate."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np

import phase11_mixed_precision_common as common


WARMUP = 30
MEASURED = 100
EXPECTED_TEST90_AGGREGATE_SCRIPT_SHA256 = "717aa6c5e10b889c082acb4626105ad708c9006acf70380793f0b93c71fbd159"
EXPECTED_COMMON_MODULE_SHA256 = "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8"
EXPECTED_RUNTIME_FINGERPRINT_GENERATOR_SHA256 = "f38e24ea72c02a164ec311ce4521f119da1e0f397fc796ea380dc62bb2167577"
RUNTIME_FINGERPRINT_SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test90-three-run-summary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--trial-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--candidate-position", type=int, required=True)
    parser.add_argument("--candidate-schedule", default="M0,M1,M2,M3,M4")
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    process_started_at_utc = utc_now()
    process_started_perf_counter_ns = perf_counter_ns()
    result = {
        "schema": "phase11_mixed_precision_task_equivalence_performance_trial_v1",
        "status": "failed",
        "trial_index": args.trial_index,
        "claims": {
            "task_equivalence_performance_trial_valid": False,
            "strict_numeric_equivalence_confirmed": False,
            "historical_m0_strict_failure_overridden": False,
            "native_int8_kernel_verified": False,
            "same_protocol_fp32_speedup_claim_allowed": False,
            "deployment_ready": False,
        },
        "process": {
            "pid": os.getpid(),
            "started_at_utc": process_started_at_utc,
        },
    }
    passed = False
    try:
        ort = common.require_runtime()
        if not IMAGE_ID_PATTERN.fullmatch(args.container_image_id):
            raise RuntimeError("container image ID must be a full sha256 digest")
        runtime_fingerprint_identity = common.identity(args.runtime_fingerprint)
        runtime_fingerprint = common.load_json(args.runtime_fingerprint)
        if runtime_fingerprint.get("schema") != RUNTIME_FINGERPRINT_SCHEMA:
            raise RuntimeError("runtime fingerprint schema drift")
        if runtime_fingerprint.get("onnxruntime") != ort.__version__:
            raise RuntimeError("runtime fingerprint ORT version drift")
        if "MIGraphXExecutionProvider" not in runtime_fingerprint.get(
            "available_providers", []
        ):
            raise RuntimeError("runtime fingerprint lacks MIGraphX EP")
        if runtime_fingerprint.get("generator", {}).get("sha256") != (
            EXPECTED_RUNTIME_FINGERPRINT_GENERATOR_SHA256
        ):
            raise RuntimeError("runtime fingerprint generator identity drift")
        manifest, manifest_identity, rows = common.load_candidate(args.bundle, args.manifest)
        common.validate_onnx_contracts(rows)
        schedule = common.parse_schedule(args.candidate_schedule)
        expected_order = common.rotated_schedule(schedule, args.trial_index)
        if not 1 <= args.candidate_position <= len(expected_order):
            raise RuntimeError("candidate position is outside the scheduled trial")
        expected_candidate = expected_order[args.candidate_position - 1]
        if manifest["candidate_id"] != expected_candidate:
            raise RuntimeError(
                f"candidate launch order drift: got={manifest['candidate_id']} expected={expected_candidate}"
            )
        admission_identity = common.identity(args.test90_three_run_summary)
        admission = common.load_json(args.test90_three_run_summary)
        if admission.get("schema") != "phase11_mixed_precision_test90_three_run_summary_v1":
            raise RuntimeError("test90 three-run prerequisite schema drift")
        if admission.get("status") != "passed" or not admission.get("claims", {}).get(
            "task_equivalence_performance_track_may_proceed"
        ):
            raise RuntimeError("test90 three-run task admission did not pass")
        if admission.get("identities", {}).get("aggregate_script", {}).get("sha256") != EXPECTED_TEST90_AGGREGATE_SCRIPT_SHA256:
            raise RuntimeError("test90 aggregate script identity drift")
        if admission.get("identities", {}).get("common_module", {}).get("sha256") != EXPECTED_COMMON_MODULE_SHA256:
            raise RuntimeError("test90 aggregate common module identity drift")
        common.validate_result_candidate(admission, manifest_identity, manifest["candidate_id"])
        if admission.get("claims", {}).get("strict_numeric_equivalence_confirmed"):
            raise RuntimeError("task admission improperly promoted strict equivalence")

        raw = common.load_raw_sample(args.sample)
        sessions = []
        load_started = perf_counter()
        for row in rows:
            sessions.append(common.create_migraphx_session(ort, row))
        session_load_seconds = perf_counter() - load_started
        if len(sessions) != 25:
            raise RuntimeError("all 25 sessions are not concurrently resident")

        device_input = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        allocated_outputs: list[dict] = []
        retained = {}
        current = device_input
        for index, session in enumerate(sessions):
            binding = session.io_binding()
            inputs = session.get_inputs()
            if index < 24:
                binding.bind_ortvalue_input(inputs[0].name, current)
            else:
                if {item.name for item in inputs} != set(retained):
                    raise RuntimeError("head retained-feature contract drift during allocation")
                for item in inputs:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                binding.bind_output(name, "cuda", 0)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            values = binding.get_outputs()
            if not all(value.device_name() == "cuda" for value in values):
                raise RuntimeError(f"output allocation left device at segment {index}")
            output_map = dict(zip(output_names, values, strict=True))
            allocated_outputs.append(output_map)
            current = common.choose_primary_output(index, output_map)
            if index in common.RETAIN_AFTER:
                retained[output_names[0]] = current
            del binding
        reference_logits = np.ascontiguousarray(current.numpy(), dtype=np.float32)
        if reference_logits.shape != (1, 2, 224, 224) or not np.isfinite(reference_logits).all():
            raise RuntimeError("initial logits contract failure")

        fixed_bindings = []
        retained = {}
        current = device_input
        for index, (session, output_map) in enumerate(
            zip(sessions, allocated_outputs, strict=True)
        ):
            binding = session.io_binding()
            inputs = session.get_inputs()
            if index < 24:
                binding.bind_ortvalue_input(inputs[0].name, current)
            else:
                for item in inputs:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            for name, output_value in output_map.items():
                binding.bind_ortvalue_output(name, output_value)
            binding.synchronize_inputs()
            fixed_bindings.append(binding)
            current = common.choose_primary_output(index, output_map)
            if index in common.RETAIN_AFTER:
                retained[next(iter(output_map))] = current

        def run_model_only() -> None:
            for session, binding in zip(sessions, fixed_bindings, strict=True):
                session.run_with_iobinding(binding)
            fixed_bindings[-1].synchronize_outputs()

        def run_end_to_end() -> np.ndarray:
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
            logits = common.choose_primary_output(24, allocated_outputs[-1]).numpy()
            return np.ascontiguousarray(logits, dtype=np.float32)

        scope_order = (
            ("model_only", "end_to_end_logits")
            if args.trial_index in (1, 3)
            else ("end_to_end_logits", "model_only")
        )
        latencies: dict[str, list[float]] = {name: [] for name in scope_order}
        output_comparisons = {}
        for scope in scope_order:
            operation = run_model_only if scope == "model_only" else run_end_to_end
            warmup_logits = None
            for warmup_index in range(WARMUP):
                returned_logits = operation()
                if warmup_index == WARMUP - 1:
                    warmup_logits = (
                        np.ascontiguousarray(
                            common.choose_primary_output(
                                24, allocated_outputs[-1]
                            ).numpy(),
                            dtype=np.float32,
                        )
                        if scope == "model_only"
                        else returned_logits
                    )
            if warmup_logits is None:
                raise RuntimeError(f"{scope} failed to retain warmup 30 output")
            warmup_comparison = common.compare_logits(reference_logits, warmup_logits)
            if warmup_comparison["mae"] > 1.0e-6 or warmup_comparison["max_abs"] > 1.0e-5:
                raise RuntimeError(f"{scope} warmup output drift: {warmup_comparison}")
            final_logits = None
            for measured_index in range(MEASURED):
                started_ns = perf_counter_ns()
                returned_logits = operation()
                latencies[scope].append((perf_counter_ns() - started_ns) / 1.0e6)
                if measured_index == MEASURED - 1:
                    final_logits = (
                        np.ascontiguousarray(
                            common.choose_primary_output(
                                24, allocated_outputs[-1]
                            ).numpy(),
                            dtype=np.float32,
                        )
                        if scope == "model_only"
                        else returned_logits
                    )
                if (measured_index + 1) % 20 == 0:
                    print(
                        f"{manifest['candidate_id']} trial {args.trial_index} "
                        f"{scope} {measured_index + 1}/{MEASURED}",
                        flush=True,
                    )
            if final_logits is None:
                raise RuntimeError(f"{scope} failed to retain measurement 100 output")
            output_comparisons[scope] = {
                "warmup": warmup_comparison,
                "final": common.compare_logits(reference_logits, final_logits),
            }

        artifacts = {}
        scope_statistics = {}
        for scope, values in latencies.items():
            path = args.output_dir / f"{scope}_latencies_ms.npy"
            np.save(path, np.asarray(values, dtype=np.float64), allow_pickle=False)
            artifacts[f"{scope}_latencies_ms"] = common.identity(path)
            scope_statistics[scope] = common.statistics(values)
        output_gates = {
            f"{scope}_final_output_stable": comparison["final"]["mae"] <= 1.0e-6
            and comparison["final"]["max_abs"] <= 1.0e-5
            and comparison["final"]["pixel_class_agreement"] == 1.0
            for scope, comparison in output_comparisons.items()
        }
        gates = {
            "task90_three_run_admission_passed": True,
            "session_count_eq_25": len(sessions) == 25,
            "warmup_per_scope_eq_30": WARMUP == 30,
            "measured_per_scope_eq_100": all(len(values) == 100 for values in latencies.values()),
            "all_latencies_positive_finite": all(
                np.isfinite(np.asarray(values)).all() and min(values) > 0
                for values in latencies.values()
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
                "candidate_lineage": {
                    "candidate_id": manifest["candidate_id"],
                    "manifest": manifest_identity,
                    "bundle": str(args.bundle.resolve(strict=True)),
                    "fp16_backbone_blocks": manifest["fp16_backbone_blocks"],
                    "int8_backbone_blocks": manifest["int8_backbone_blocks"],
                },
                "identities": {
                    "benchmark_script": common.identity(Path(__file__)),
                    "common_module": common.identity(Path(common.__file__)),
                    "sample": common.identity(args.sample),
                    "test90_three_run_summary": admission_identity,
                    "runtime_fingerprint": runtime_fingerprint_identity,
                },
                "schedule": {
                    "base_candidate_schedule": list(schedule),
                    "rotated_trial_order": list(expected_order),
                    "candidate_position_1_based": args.candidate_position,
                    "scope_order": list(scope_order),
                },
                "protocol": {
                    "track": "task_equivalence_performance",
                    "batch": 1,
                    "warmup_per_scope": WARMUP,
                    "measured_per_scope": MEASURED,
                    "fresh_process_trial": True,
                    "validation_outputs": (
                        "only inference 30 of warmup and inference 100 of measurement; "
                        "no extra inference between or after timed loops"
                    ),
                    "model_only_scope": (
                        "25 resident sessions; fixed device input/intermediate/output OrtValues; "
                        "includes 25 Python ORT dispatches; excludes H2D, D2H, and session load"
                    ),
                    "end_to_end_logits_scope": (
                        "host float32 input -> H2D -> 25 resident sessions/device OrtValues -> "
                        "D2H float32 logits; excludes file I/O, preprocessing, and session load"
                    ),
                },
                "runtime": {
                    "hostname": platform.node(),
                    "python": platform.python_version(),
                    "onnxruntime": ort.__version__,
                    "container_image_id": args.container_image_id,
                    "runtime_fingerprint_schema": runtime_fingerprint["schema"],
                    "session_load_seconds_excluded": session_load_seconds,
                    "process_max_rss_kib": max_rss_kib,
                },
                "statistics": scope_statistics,
                "output_comparisons": output_comparisons,
                "gates": gates,
                "artifacts": artifacts,
                "evidence_boundary": {
                    "task_equivalence_track_not_strict_numeric_track": True,
                    "historical_m0_strict_failure_is_immutable": True,
                    "provider_placement_inherited_from_three_passed_test90_runs": True,
                    "timed_region_is_not_profiled": True,
                    "vram_sampling_is_a_separate_process_and_not_present_in_timed_trials": True,
                    "native_int8_kernel_not_proven_by_latency": True,
                    "same_protocol_fp32_speedup_not_computed_here": True,
                },
            }
        )
        result["claims"]["task_equivalence_performance_trial_valid"] = passed
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result["process"].update(
        {
            "ended_at_utc": utc_now(),
            "elapsed_seconds": (
                perf_counter_ns() - process_started_perf_counter_ns
            )
            / 1.0e9,
        }
    )
    common.json_dump(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
