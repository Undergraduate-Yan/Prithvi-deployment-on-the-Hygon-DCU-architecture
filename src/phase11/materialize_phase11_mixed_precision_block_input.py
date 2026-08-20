#!/usr/bin/env python3
"""Materialize one realistic block-boundary input outside hipprof tracing.

This prepass executes only the canonical candidate's upstream encoder segments
with device OrtValue hand-off, copies the resulting boundary once to a float32
NPY file, and records complete sample/manifest/model/cache lineage. It never
makes a kernel-precision claim.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import phase11_mixed_precision_common as common


SCHEMA = "phase11_mixed_precision_boundary_prepass_v3"
RUNTIME_SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"
IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_PRECISIONS = {"int8_qdq", "fp16"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_encoder_once(ort: Any, session: Any, current: Any) -> tuple[Any, dict[str, Any]]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError("encoder segment must have exactly one input")
    output_names = [item.name for item in session.get_outputs()]
    binding = session.io_binding()
    binding.bind_ortvalue_input(inputs[0].name, current)
    for name in output_names:
        binding.bind_output(name, "cuda", 0)
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    outputs = binding.get_outputs()
    if len(outputs) != len(output_names):
        raise RuntimeError("encoder output-count drift")
    if not all(value.device_name() == "cuda" for value in outputs):
        raise RuntimeError("encoder output left the K100 device")
    output_map = dict(zip(output_names, outputs, strict=True))
    return common.choose_primary_output(0, output_map), output_map


def lineage_digest(lineage: dict[str, Any]) -> str:
    encoded = json.dumps(
        lineage, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--expected-precision", choices=sorted(ALLOWED_PRECISIONS), required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.segment_index < 24:
        parser.error("--segment-index must select an encoder block in [0, 23]")
    if not IMAGE_RE.fullmatch(args.container_image_id):
        parser.error("--container-image-id must be a full sha256 image ID")
    if not SHA_RE.fullmatch(args.expected_model_sha256):
        parser.error("--expected-model-sha256 must be a lowercase SHA256")
    if not SHA_RE.fullmatch(args.expected_cache_sha256):
        parser.error("--expected-cache-sha256 must be a lowercase SHA256")

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
        },
        "execution_scope": {
            "profiler_attached": False,
            "upstream_segments": list(range(args.segment_index)),
            "selected_segment_executed": False,
            "boundary_copy_to_host_count": 1,
        },
        "claims": {
            "untracked_prepass_completed": False,
            "manifest_model_cache_lineage_locked": False,
            "boundary_input_sha_locked": False,
            "upstream_device_ortvalue_chain_preserved": False,
            "kernel_precision_verified": False,
            "strict_numeric_equivalence_confirmed": False,
            "deployment_ready": False,
        },
        "claim_boundary": (
            "The prepass establishes only a realistic, identity-locked float32 boundary input. "
            "It is deliberately outside hipprof and cannot prove I8II, HBH, speedup, or strict "
            "numeric equivalence."
        ),
    }
    try:
        runtime_identity = common.identity(args.runtime_fingerprint)
        runtime = json.loads(args.runtime_fingerprint.read_text(encoding="utf-8"))
        if runtime.get("schema") != RUNTIME_SCHEMA:
            raise RuntimeError("runtime fingerprint schema drift")
        if runtime.get("onnxruntime") != common.EXPECTED_ORT_VERSION:
            raise RuntimeError("runtime fingerprint ORT version drift")
        if common.MGX not in runtime.get("available_providers", []):
            raise RuntimeError("runtime fingerprint lacks MIGraphXExecutionProvider")

        sample_identity = common.identity(args.sample)
        raw = common.load_raw_sample(args.sample)
        sample_tensor_sha = common.array_sha256(raw)
        ort = common.require_runtime()
        manifest, manifest_identity, rows = common.load_candidate(args.bundle, args.manifest)
        if manifest.get("candidate_id") != args.candidate_id:
            raise RuntimeError("candidate ID does not match the selected manifest")
        target_row = rows[args.segment_index]
        if target_row["precision"] != args.expected_precision:
            raise RuntimeError("target precision differs from the profile plan")
        if target_row["verified_model_identity"]["sha256"] != args.expected_model_sha256:
            raise RuntimeError("target model SHA differs from the profile plan")
        if target_row["verified_cache_identity"]["sha256"] != args.expected_cache_sha256:
            raise RuntimeError("target cache SHA differs from the profile plan")
        if len(target_row["input_contracts"]) != 1:
            raise RuntimeError("target encoder segment must have one input contract")

        current = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        if current.device_name() != "cuda":
            raise RuntimeError("sample did not enter K100 device memory")
        resident_sessions = []
        observed_upstream = []
        for index in range(args.segment_index):
            row = rows[index]
            session = common.create_migraphx_session(ort, row)
            resident_sessions.append(session)
            current, output_map = run_encoder_once(ort, session, current)
            observed_upstream.append(
                {
                    "index": index,
                    "precision": row["precision"],
                    "model": row["verified_model_identity"],
                    "cache": row["verified_cache_identity"],
                    "output_names": list(output_map),
                    "primary_output_device": current.device_name(),
                }
            )

        boundary = np.ascontiguousarray(current.numpy())
        if boundary.dtype != np.float32:
            raise RuntimeError(f"boundary dtype must be float32, got {boundary.dtype}")
        contract = target_row["input_contracts"][0]
        if contract["shape"] is None or list(boundary.shape) != list(contract["shape"]):
            raise RuntimeError(
                f"boundary shape drift: {list(boundary.shape)} != {contract.get('shape')}"
            )
        if not np.isfinite(boundary).all():
            raise RuntimeError("boundary contains NaN or Inf")

        boundary_path = args.output_dir / "boundary_input.npy"
        np.save(boundary_path, boundary, allow_pickle=False)
        boundary_identity = common.identity(boundary_path)
        reloaded = np.load(boundary_path, allow_pickle=False)
        if (
            reloaded.dtype != np.float32
            or list(reloaded.shape) != list(boundary.shape)
            or common.array_sha256(reloaded) != common.array_sha256(boundary)
        ):
            raise RuntimeError("saved boundary NPY did not round-trip exactly")

        lineage = {
            "canonical_candidate_id": args.candidate_id,
            "manifest": manifest_identity,
            "source_sample": sample_identity,
            "source_sample_tensor_sha256": sample_tensor_sha,
            "upstream_segments": [
                {
                    "index": row["index"],
                    "precision": row["precision"],
                    "model": row["model"],
                    "cache": row["cache"],
                }
                for row in observed_upstream
            ],
            "target_segment": {
                "index": args.segment_index,
                "precision": target_row["precision"],
                "model": target_row["verified_model_identity"],
                "cache": target_row["verified_cache_identity"],
            },
            "boundary_tensor_sha256": common.array_sha256(boundary),
        }
        result.update(
            {
                "status": "boundary_input_materialized_untracked_prepass_passed",
                "ended_at_utc": utc_now(),
                "container_image_id": args.container_image_id,
                "runtime_fingerprint": runtime_identity,
                "source_sample": sample_identity,
                "source_sample_tensor_sha256": sample_tensor_sha,
                "manifest": manifest_identity,
                "selected_artifacts": {
                    "model": target_row["verified_model_identity"],
                    "cache": target_row["verified_cache_identity"],
                },
                "boundary_input": boundary_identity,
                "lineage": lineage,
                "lineage_sha256": lineage_digest(lineage),
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "available_providers": ort.get_available_providers(),
                    "upstream_segments_executed": args.segment_index,
                    "resident_upstream_sessions": len(resident_sessions),
                    "upstream_device_ortvalue_chain_preserved": True,
                    "boundary_shape": list(boundary.shape),
                    "boundary_dtype": str(boundary.dtype),
                    "boundary_tensor_sha256": common.array_sha256(boundary),
                    "all_values_finite": True,
                },
                "claims": {
                    **result["claims"],
                    "untracked_prepass_completed": True,
                    "manifest_model_cache_lineage_locked": True,
                    "boundary_input_sha_locked": True,
                    "upstream_device_ortvalue_chain_preserved": True,
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
    if result["status"] != "boundary_input_materialized_untracked_prepass_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
