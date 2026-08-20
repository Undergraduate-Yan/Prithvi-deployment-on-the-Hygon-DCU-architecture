#!/usr/bin/env python3
"""Strict prerequisite and post-run checks for the machine2 segment25 test90 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


PORTABILITY_SCHEMA = "phase11_segment25_crossnode_cache_portability_validation_v1"
EXPECTED_IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
EXPECTED_BUNDLE_MANIFEST = (
    126_255,
    "0abd5a99c3fa9ebcd4236bf6f57090f9fce102777535402313e081777eb9c6cc",
)
EXPECTED_EVALUATOR = (
    18_982,
    "b0d4e876c6cd3845fad10775f803591c558621cb81af9977353152c73fdefe79",
)
EXPECTED_LSMOD = (
    819_664,
    "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12",
)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
EXPECTED_GATE_KEYS = (
    "sample_count_eq_90",
    "valid_pixels_exact",
    "all_outputs_finite",
    "all_25_profiles_migraphx_positive_cpu_zero",
    "all_intermediate_outputs_device_cuda_alias",
    "miou_drop_le_2_0pp",
    "water_iou_drop_le_3_0pp",
    "boundary_drop_le_3_0pp",
    "valid_prediction_agreement_vs_fp32_ge_98pct",
    "valid_prediction_agreement_vs_host_staged_ge_99_99pct",
    "metric_abs_delta_vs_host_staged_le_0_01pp",
)
EXPECTED_PORTABILITY_GATE_KEYS = (
    "manifest_identity_locked",
    "manifest_segment_contract_exact",
    "cache_path_set_exact_25",
    "cache_identities_unchanged_after_run",
    "no_cache_symlinks",
    "no_unexpected_mxr_files",
    "probe_exit_zero",
    "probe_result_status_passed",
    "all_probe_gates_true",
    "all_25_sessions_concurrently_resident",
    "device_ortvalue_pipeline",
    "all_25_profiles_migraphx_positive_cpu_zero",
    "node3_frozen_logits_mae_le_1e_6",
    "node3_frozen_logits_max_abs_le_1e_5",
    "node3_frozen_predictions_exact",
    "all_25_caches_opened_read_only",
    "all_25_caches_consumed_by_read_or_mmap",
    "no_mxr_write_create_truncate_access",
    "bundle_read_only_mount_declared",
    "critical_runtime_content_match",
    "exact_image_identity_is_false",
    "target_image_identity_locked",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock(
    path: Path,
    expected: tuple[int, str] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    item = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    if expected_sha256 is not None and item["sha256"] != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {resolved}: expected {expected_sha256}, found {item['sha256']}"
        )
    return item


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def require_bool(mapping: dict[str, Any], key: str, expected: bool) -> None:
    if mapping.get(key) is not expected:
        raise RuntimeError(f"{key} must be exactly {expected!r}, found {mapping.get(key)!r}")


def validate_portability(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("portability SHA-256 must be 64 lowercase hexadecimal digits")
    identity = lock(path, expected_sha256=expected_sha256)
    result = load_json(path)
    if result.get("schema") != PORTABILITY_SCHEMA:
        raise RuntimeError(f"cache-portability schema drift: {result.get('schema')!r}")
    if result.get("status") != "passed":
        raise RuntimeError(f"cache-portability validation did not pass: {result.get('status')!r}")
    if result.get("errors") != []:
        raise RuntimeError(f"cache-portability validation reports errors: {result.get('errors')!r}")
    gates = result.get("gates")
    claims = result.get("claims")
    if not isinstance(gates, dict) or not gates:
        raise RuntimeError("cache-portability validation has no gates")
    if not isinstance(claims, dict):
        raise RuntimeError("cache-portability validation has no claims object")
    for key in EXPECTED_PORTABILITY_GATE_KEYS:
        require_bool(gates, key, True)
    failed_gates = {key: value for key, value in gates.items() if value is not True}
    if failed_gates:
        raise RuntimeError(f"cache-portability gate failure: {failed_gates}")
    for key in (
        "cache_portability_verified",
        "origin_caches_portable_on_machine2",
        "origin_cache_portable_on_target_machine2",
        "all_25_origin_caches_loaded_on_target",
        "critical_runtime_content_match",
    ):
        require_bool(claims, key, True)
    for key in (
        "exact_image_identity_match",
        "exact_layer_manifest_match",
        "task_utility_test90_verified_on_target",
        "cross_node_performance_verified",
        "native_int8_kernel_verified",
        "deployment_ready",
    ):
        require_bool(claims, key, False)
    return result, identity


def validate_profile_counts(counts: Any, label: str) -> None:
    if not isinstance(counts, dict):
        raise RuntimeError(f"{label}: provider_event_counts is not an object")
    if not isinstance(counts.get(MGX), int) or counts[MGX] <= 0:
        raise RuntimeError(f"{label}: no positive MIGraphX provider count: {counts}")
    if counts.get(CPU, 0) != 0:
        raise RuntimeError(f"{label}: CPU provider events observed: {counts}")
    positive_other = {
        key: value
        for key, value in counts.items()
        if key != MGX and isinstance(value, int) and value > 0
    }
    if positive_other:
        raise RuntimeError(f"{label}: unexpected positive provider events: {positive_other}")


def resolve_output_artifact(output_dir: Path, recorded: dict[str, Any]) -> Path:
    recorded_path = PurePosixPath(str(recorded.get("path", "")))
    if recorded_path.parent != PurePosixPath("/work/run/output"):
        raise RuntimeError(f"recorded artifact path escapes /work/run/output: {recorded_path}")
    if not recorded_path.name or recorded_path.name in {".", ".."}:
        raise RuntimeError(f"invalid recorded artifact path: {recorded}")
    return output_dir / recorded_path.name


def validate_recorded_artifact(output_dir: Path, recorded: Any, label: str) -> dict[str, Any]:
    if not isinstance(recorded, dict):
        raise RuntimeError(f"{label}: recorded identity is not an object")
    expected = (int(recorded["size_bytes"]), str(recorded["sha256"]))
    return lock(resolve_output_artifact(output_dir, recorded), expected)


def profile_provider_counts(path: Path) -> dict[str, int]:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise RuntimeError(f"ORT profile is not an event list: {path}")
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if isinstance(event, dict)
        and event.get("cat") == "Node"
        and isinstance(event.get("args"), dict)
        and event["args"].get("provider")
    )
    return dict(counts)


def validate_test90(args: argparse.Namespace) -> dict[str, Any]:
    _, portability_identity = validate_portability(
        args.portability_result, args.portability_sha256
    )
    identities = {
        "cache_portability_validation": portability_identity,
        "bundle_manifest": lock(args.bundle_manifest, EXPECTED_BUNDLE_MANIFEST),
        "evaluator": lock(args.evaluator, EXPECTED_EVALUATOR),
        "static_lsmod_shim": lock(args.lsmod, EXPECTED_LSMOD),
        "test90_result": lock(args.test90_result),
        "image_inspect": lock(args.image_inspect),
    }
    image = json.loads(args.image_inspect.read_text(encoding="utf-8"))
    if not isinstance(image, list) or len(image) != 1 or image[0].get("Id") != EXPECTED_IMAGE:
        raise RuntimeError("machine2 image identity drift in image_inspect.json")

    result = load_json(args.test90_result)
    if result.get("status") != "diagnostic_completed":
        raise RuntimeError(f"test90 status must remain diagnostic_completed: {result.get('status')!r}")
    if result.get("variant") != "int8_backbone_compat_25_static_batch1_cached_device_resident_test90":
        raise RuntimeError(f"test90 variant drift: {result.get('variant')!r}")
    if result.get("evaluated_samples") != 90:
        raise RuntimeError(f"test90 evaluated_samples drift: {result.get('evaluated_samples')!r}")
    if result.get("task_utility_diagnostic_passed") is not True:
        raise RuntimeError("top-level task_utility_diagnostic_passed is not true")

    claims = result.get("claims")
    gates = result.get("gates")
    runtime = result.get("runtime")
    if not isinstance(claims, dict) or not isinstance(gates, dict) or not isinstance(runtime, dict):
        raise RuntimeError("test90 claims/gates/runtime contract is incomplete")
    for key in (
        "all_25_segments_strict_migraphx_placement",
        "device_resident_intersegment_io_test90",
        "task_utility_diagnostic_passed",
    ):
        require_bool(claims, key, True)
    for key in (
        "strict_end_to_end_numeric_admission",
        "performance",
        "native_int8_kernel_verified",
        "deployment_ready",
    ):
        require_bool(claims, key, False)
    for key in EXPECTED_GATE_KEYS:
        require_bool(gates, key, True)
    failed_gates = {key: value for key, value in gates.items() if value is not True}
    if failed_gates:
        raise RuntimeError(f"test90 contains a failed or non-boolean gate: {failed_gates}")
    if runtime.get("onnxruntime") != "1.19.2":
        raise RuntimeError(f"ONNX Runtime version drift: {runtime.get('onnxruntime')!r}")
    if runtime.get("device_type_string_for_rocm_build") != "cuda":
        raise RuntimeError("ROCm OrtValue device alias drift")
    if runtime.get("intersegment_transport") != "direct_OrtValue_device_binding":
        raise RuntimeError("inter-segment transport is not device-resident I/O Binding")
    if runtime.get("host_intermediate_copies") != 0:
        raise RuntimeError("host intermediate copies were reported")
    if result.get("metrics", {}).get("valid_pixels") != 3_927_398:
        raise RuntimeError("valid-pixel identity drift")

    profiles = result.get("profiles")
    segments = result.get("segments")
    if not isinstance(profiles, list) or len(profiles) != 25:
        raise RuntimeError("test90 must contain exactly 25 profiles")
    if not isinstance(segments, list) or len(segments) != 25:
        raise RuntimeError("test90 must contain exactly 25 segment rows")
    if [item.get("index") for item in profiles] != list(range(25)):
        raise RuntimeError("profile index/order drift")
    if [item.get("index") for item in segments] != list(range(25)):
        raise RuntimeError("segment index/order drift")
    for index, profile in enumerate(profiles):
        if profile.get("passed") is not True:
            raise RuntimeError(f"segment {index}: profile passed flag is not true")
        described_counts = profile.get("provider_event_counts")
        validate_profile_counts(described_counts, f"profile {index}")
        identities[f"profile_{index:02d}"] = validate_recorded_artifact(
            args.output_dir, profile, f"profile {index}"
        )
        raw_profile = resolve_output_artifact(args.output_dir, profile)
        observed_counts = profile_provider_counts(raw_profile)
        if observed_counts != described_counts:
            raise RuntimeError(
                f"profile {index}: raw/provider count disagreement: "
                f"recorded={described_counts}, observed={observed_counts}"
            )
        validate_profile_counts(observed_counts, f"raw profile {index}")
    for index, segment in enumerate(segments):
        if segment.get("all_90_outputs_device_cuda_alias") is not True:
            raise RuntimeError(f"segment {index}: one or more outputs were not device OrtValues")
        validate_profile_counts(segment.get("provider_event_counts"), f"segment {index}")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("test90 artifact map missing")
    identities["per_sample_csv"] = validate_recorded_artifact(
        args.output_dir, artifacts.get("per_sample_csv"), "per_sample_csv"
    )
    identities["predictions_and_targets"] = validate_recorded_artifact(
        args.output_dir,
        artifacts.get("predictions_and_targets"),
        "predictions_and_targets",
    )
    boundaries = result.get("evidence_boundary")
    if not isinstance(boundaries, dict):
        raise RuntimeError("test90 evidence_boundary missing")
    for key in (
        "strict_logits_numeric_gate_remains_failed",
        "task_gate_does_not_retroactively_pass_strict_numeric_admission",
        "device_resident_25_segment_task90_is_not_yet_a_monolithic_deployment_artifact",
        "timings_are_diagnostic_and_not_a_performance_benchmark",
        "profiles_do_not_prove_native_int8_kernels",
    ):
        require_bool(boundaries, key, True)

    return {
        "schema": "phase11_segment25_crossnode_device_test90_validation_v1",
        "status": "passed",
        "errors": [],
        "identities": identities,
        "gates": {
            "cache_portability_prerequisite_passed_and_sha_locked": True,
            "exact_machine2_image_id": True,
            "test90_status_is_diagnostic_completed": True,
            "all_test90_gates_true": True,
            "all_25_profiles_migraphx_positive_cpu_zero": True,
            "all_25_segment_outputs_device_ortvalues": True,
            "task_utility_diagnostic_passed": True,
            "output_artifacts_sha_locked": True,
        },
        "claims": {
            "cross_node_cache_portability_verified": True,
            "cross_node_task_utility_test90_verified": True,
            "device_resident_intersegment_io_test90": True,
            "strict_end_to_end_numeric_admission": False,
            "performance": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
        "evidence_boundary": {
            "status_diagnostic_completed_preserved": True,
            "test90_timing_is_not_a_performance_benchmark": True,
            "matched_runtime_rebuild_is_not_exact_image_identity": True,
            "task_utility_does_not_rewrite_the_failed_strict_logit_gate": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight-portability")
    preflight.add_argument("--portability-result", type=Path, required=True)
    preflight.add_argument("--portability-sha256", required=True)

    post = subparsers.add_parser("validate-result")
    post.add_argument("--portability-result", type=Path, required=True)
    post.add_argument("--portability-sha256", required=True)
    post.add_argument("--bundle-manifest", type=Path, required=True)
    post.add_argument("--evaluator", type=Path, required=True)
    post.add_argument("--lsmod", type=Path, required=True)
    post.add_argument("--image-inspect", type=Path, required=True)
    post.add_argument("--test90-result", type=Path, required=True)
    post.add_argument("--output-dir", type=Path, required=True)
    post.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "preflight-portability":
        result, identity = validate_portability(
            args.portability_result, args.portability_sha256
        )
        print(json.dumps({"status": "passed", "identity": identity, "claims": result["claims"]}, indent=2))
        return 0

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {args.output}")
    try:
        result = validate_test90(args)
        rc = 0
    except Exception as exc:
        result = {
            "schema": "phase11_segment25_crossnode_device_test90_validation_v1",
            "status": "failed",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc(),
            "claims": {
                "cross_node_cache_portability_verified": False,
                "cross_node_task_utility_test90_verified": False,
                "device_resident_intersegment_io_test90": False,
                "strict_end_to_end_numeric_admission": False,
                "performance": False,
                "native_int8_kernel_verified": False,
                "deployment_ready": False,
            },
        }
        rc = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
