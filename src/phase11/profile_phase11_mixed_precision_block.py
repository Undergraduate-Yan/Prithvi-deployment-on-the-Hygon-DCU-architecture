#!/usr/bin/env python3
"""Run exactly one manifest-locked Phase 11 block under external hipprof.

The target process never constructs or executes an upstream segment. A separate,
unprofiled prepass materializes the selected block's realistic float32 boundary
input. The launcher starts this program through a direct outer hipprof invocation
(the Phase 10 mode); this process has no dynamic profiler-session controls.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import phase11_mixed_precision_common as common


SCHEMA = "phase11_mixed_precision_block_external_hipprof_target_v3"
PREPASS_SCHEMA = "phase11_mixed_precision_boundary_prepass_v3"
RUNTIME_SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"
IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_PRECISIONS = {"int8_qdq", "fp16"}


def pair(value: dict[str, Any]) -> tuple[int, str]:
    return int(value["size_bytes"]), str(value["sha256"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def validate_boundary_prepass(
    prepass: dict[str, Any],
    *,
    candidate_id: str,
    segment_index: int,
    precision: str,
    image_id: str,
    runtime_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    row: dict[str, Any],
    boundary_identity: dict[str, Any],
) -> None:
    if (
        prepass.get("schema") != PREPASS_SCHEMA
        or prepass.get("status") != "boundary_input_materialized_untracked_prepass_passed"
    ):
        raise RuntimeError("boundary prepass schema/status drift")
    selection = prepass.get("selection", {})
    if (
        selection.get("candidate_id") != candidate_id
        or int(selection.get("segment_index", -1)) != segment_index
        or selection.get("expected_precision") != precision
    ):
        raise RuntimeError("boundary prepass selection drift")
    if prepass.get("container_image_id") != image_id:
        raise RuntimeError("boundary prepass image identity drift")
    if pair(prepass.get("runtime_fingerprint", {})) != pair(runtime_identity):
        raise RuntimeError("boundary prepass runtime identity drift")
    if pair(prepass.get("manifest", {})) != pair(manifest_identity):
        raise RuntimeError("boundary prepass manifest identity drift")
    selected = prepass.get("selected_artifacts", {})
    if pair(selected.get("model", {})) != pair(row["verified_model_identity"]):
        raise RuntimeError("boundary prepass target-model identity drift")
    if pair(selected.get("cache", {})) != pair(row["verified_cache_identity"]):
        raise RuntimeError("boundary prepass target-cache identity drift")
    if pair(prepass.get("boundary_input", {})) != pair(boundary_identity):
        raise RuntimeError("boundary prepass file identity drift")
    runtime = prepass.get("runtime", {})
    if not (
        runtime.get("all_values_finite")
        and runtime.get("upstream_device_ortvalue_chain_preserved")
        and int(runtime.get("upstream_segments_executed", -1)) == segment_index
        and runtime.get("boundary_dtype") == "float32"
    ):
        raise RuntimeError("boundary prepass operational gates failed")
    claims = prepass.get("claims", {})
    if not (
        claims.get("untracked_prepass_completed")
        and claims.get("manifest_model_cache_lineage_locked")
        and claims.get("boundary_input_sha_locked")
    ):
        raise RuntimeError("boundary prepass claims are incomplete")
    if claims.get("kernel_precision_verified"):
        raise RuntimeError("boundary prepass improperly claimed kernel precision")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--expected-precision", choices=sorted(ALLOWED_PRECISIONS), required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--boundary-input", type=Path, required=True)
    parser.add_argument("--boundary-prepass-result", type=Path, required=True)
    parser.add_argument("--expected-boundary-input-size", type=int, required=True)
    parser.add_argument("--expected-boundary-input-sha256", required=True)
    parser.add_argument("--expected-prepass-result-size", type=int, required=True)
    parser.add_argument("--expected-prepass-result-sha256", required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.segment_index < 24:
        parser.error("--segment-index must select an encoder block in [0, 23]")
    if not 1 <= args.repetitions <= 1000:
        parser.error("--repetitions must be in [1, 1000]")
    if not IMAGE_RE.fullmatch(args.container_image_id):
        parser.error("--container-image-id must be a full sha256 image ID")
    for value, label in (
        (args.expected_model_sha256, "--expected-model-sha256"),
        (args.expected_cache_sha256, "--expected-cache-sha256"),
        (args.expected_boundary_input_sha256, "--expected-boundary-input-sha256"),
        (args.expected_prepass_result_sha256, "--expected-prepass-result-sha256"),
    ):
        if not SHA_RE.fullmatch(value):
            parser.error(f"{label} must be a lowercase SHA256")
    if args.expected_boundary_input_size <= 0 or args.expected_prepass_result_size <= 0:
        parser.error("expected boundary/prepass sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
        "generator": common.identity(Path(__file__)),
        "common_module": common.identity(Path(common.__file__)),
        "selection": {
            "candidate_id": args.candidate_id,
            "segment_index": args.segment_index,
            "expected_precision": args.expected_precision,
            "repetitions": args.repetitions,
        },
        "trace_contract": {
            "mode": "direct_outer_hipprof_no_dynamic_session_v3",
            "upstream_segments_executed_in_target_process": 0,
            "internal_trace_control_calls": 0,
            "target_segment_calls_requested": args.repetitions,
            "target_session_creation_and_boundary_upload_are_inside_outer_trace": True,
            "final_output_d2h_for_finite_sha_validation_is_inside_outer_trace": True,
        },
        "claims": {
            "manifest_model_cache_identity_locked": False,
            "boundary_prepass_lineage_locked": False,
            "selected_segment_migraphx_positive_cpu_zero": False,
            "profiled_input_and_output_device_resident": False,
            "selected_segment_only_no_upstream_execution": False,
            "internal_dynamic_hipprof_control_absent": True,
            "external_hipprof_trace_parsed": False,
            "kernel_precision_verified": False,
            "strict_numeric_equivalence_confirmed": False,
            "deployment_ready": False,
        },
        "claim_boundary": (
            "This target proves artifact/boundary identity, exactly one selected-segment process "
            "scope, device-resident execution, and ORT provider placement. I8II/HBH proof is "
            "decided only after the direct outer hipprof artifacts are parsed."
        ),
    }
    try:
        runtime_identity = common.identity(args.runtime_fingerprint)
        runtime = load_object(args.runtime_fingerprint)
        if runtime.get("schema") != RUNTIME_SCHEMA:
            raise RuntimeError("runtime fingerprint schema drift")
        if runtime.get("onnxruntime") != common.EXPECTED_ORT_VERSION:
            raise RuntimeError("runtime fingerprint ORT version drift")
        if common.MGX not in runtime.get("available_providers", []):
            raise RuntimeError("runtime fingerprint lacks MIGraphXExecutionProvider")

        boundary_identity = common.identity(
            args.boundary_input,
            (args.expected_boundary_input_size, args.expected_boundary_input_sha256),
        )
        prepass_identity = common.identity(
            args.boundary_prepass_result,
            (args.expected_prepass_result_size, args.expected_prepass_result_sha256),
        )
        prepass = load_object(args.boundary_prepass_result)

        ort = common.require_runtime()
        manifest, manifest_identity, rows = common.load_candidate(args.bundle, args.manifest)
        if manifest.get("candidate_id") != args.candidate_id:
            raise RuntimeError("candidate ID does not match the selected manifest")
        row = rows[args.segment_index]
        if row["precision"] != args.expected_precision:
            raise RuntimeError("selected block precision differs from the profile plan")
        if row["verified_model_identity"]["sha256"] != args.expected_model_sha256:
            raise RuntimeError("selected block model SHA differs from the profile plan")
        if row["verified_cache_identity"]["sha256"] != args.expected_cache_sha256:
            raise RuntimeError("selected block cache SHA differs from the profile plan")

        validate_boundary_prepass(
            prepass,
            candidate_id=args.candidate_id,
            segment_index=args.segment_index,
            precision=args.expected_precision,
            image_id=args.container_image_id,
            runtime_identity=runtime_identity,
            manifest_identity=manifest_identity,
            row=row,
            boundary_identity=boundary_identity,
        )

        boundary = np.load(args.boundary_input.resolve(strict=True), allow_pickle=False)
        if boundary.dtype != np.float32:
            raise RuntimeError(f"boundary input dtype must be float32, got {boundary.dtype}")
        boundary = np.ascontiguousarray(boundary)
        contract = row["input_contracts"]
        if len(contract) != 1 or contract[0]["shape"] is None:
            raise RuntimeError("selected encoder block must have one static input contract")
        if list(boundary.shape) != list(contract[0]["shape"]):
            raise RuntimeError(
                f"boundary input shape drift: {list(boundary.shape)} != {contract[0]['shape']}"
            )
        if not np.isfinite(boundary).all():
            raise RuntimeError("boundary input contains NaN or Inf")

        device_input = ort.OrtValue.ortvalue_from_numpy(boundary, "cuda", 0)
        if device_input.device_name() != "cuda":
            raise RuntimeError("boundary input did not enter K100 device memory")

        profile_prefix = args.output_dir / f"segment_{args.segment_index:02d}_ort_profile"
        target_session = common.create_migraphx_session(ort, row, profile_prefix)
        inputs = target_session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError("selected encoder segment must have exactly one input")
        output_names = [item.name for item in target_session.get_outputs()]
        binding = target_session.io_binding()
        binding.bind_ortvalue_input(inputs[0].name, device_input)
        for name in output_names:
            binding.bind_output(name, "cuda", 0)
        binding.synchronize_inputs()

        # First call allocates outputs; remaining calls reuse those exact device
        # OrtValues. Total selected-segment calls still equals repetitions.
        started_ns = time.perf_counter_ns()
        target_session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        outputs = binding.get_outputs()
        if len(outputs) != len(output_names):
            raise RuntimeError("selected segment output-count drift")
        if not all(value.device_name() == "cuda" for value in outputs):
            raise RuntimeError("selected segment output left the K100 device")

        fixed_binding = target_session.io_binding()
        fixed_binding.bind_ortvalue_input(inputs[0].name, device_input)
        for name, value in zip(output_names, outputs, strict=True):
            fixed_binding.bind_ortvalue_output(name, value)
        fixed_binding.synchronize_inputs()
        for _ in range(args.repetitions - 1):
            target_session.run_with_iobinding(fixed_binding)
            fixed_binding.synchronize_outputs()
        elapsed_ns = time.perf_counter_ns() - started_ns

        output_map = dict(zip(output_names, outputs, strict=True))
        primary = common.choose_primary_output(args.segment_index, output_map)
        output = np.ascontiguousarray(primary.numpy(), dtype=np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("profiled block produced non-finite output")
        ort_profile = Path(target_session.end_profiling()).resolve(strict=True)
        placement = common.parse_profile(ort_profile)
        if not placement["passed"]:
            raise RuntimeError(f"selected block provider placement failed: {placement}")

        result.update(
            {
                "status": "target_passed_pending_external_hipprof_parse",
                "ended_at_utc": utc_now(),
                "container_image_id": args.container_image_id,
                "runtime_fingerprint": runtime_identity,
                "boundary_prepass_result": prepass_identity,
                "boundary_input": boundary_identity,
                "source_sample": prepass.get("source_sample"),
                "manifest": manifest_identity,
                "selected_artifacts": {
                    "model": row["verified_model_identity"],
                    "cache": row["verified_cache_identity"],
                },
                "prepass_lineage": prepass.get("lineage"),
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "available_providers": ort.get_available_providers(),
                    "sessions_created": 1,
                    "upstream_segments_executed": 0,
                    "profiled_selected_segment_calls": args.repetitions,
                    "profiled_elapsed_ns_including_per_call_output_sync": elapsed_ns,
                    "input_shape": list(boundary.shape),
                    "input_dtype": str(boundary.dtype),
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    "output_tensor_sha256": common.array_sha256(output),
                    "all_output_finite": True,
                    "input_device": device_input.device_name(),
                    "output_devices": [value.device_name() for value in outputs],
                },
                "ort_provider_placement": {
                    **placement,
                    "profile": common.identity(ort_profile),
                },
                "claims": {
                    **result["claims"],
                    "manifest_model_cache_identity_locked": True,
                    "boundary_prepass_lineage_locked": True,
                    "selected_segment_migraphx_positive_cpu_zero": True,
                    "profiled_input_and_output_device_resident": True,
                    "selected_segment_only_no_upstream_execution": True,
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "ended_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    result_path = args.output_dir / "result.json"
    common.json_dump(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] != "target_passed_pending_external_hipprof_parse":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
