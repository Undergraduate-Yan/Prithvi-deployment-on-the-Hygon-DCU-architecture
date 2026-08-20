#!/usr/bin/env python3
"""Freeze and validate the machine2 cross-node benchmark prerequisites."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from pathlib import Path
from typing import Any


TARGET_IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
RUNTIME_COMPARISON = (
    4_463,
    "0feb3e326db7b004667d0eb7321bd34e3d0b04782c8c25d8f0dda9b3c937a8a6",
)
PORTABILITY_SUMMARY = (
    37_717,
    "49b696167390637a31b9688bd978f5a9febde47e21b43d29fdf3dd2af18b076d",
)
TARGET_TEST90_RESULT = (
    40_583,
    "a00cb99910b4f5273cc8719669b6c6a5c513456043145aeb1c1eb9508c260e2e",
)
TARGET_TEST90_VALIDATION = (
    11_079,
    "c8fa2dad602d7a53d2ffac0d66958aa0fe4282bee5b0adc5438f6e73dbd5e387",
)
RUNTIME_CONTENT_SHA = "fe7324427d3d54499b5b4542cdbdf0af06e45f8290d7bab633f83799da9cf272"
PORTABILITY_REQUIRED_CLAIM_KEYS = (
    "origin_caches_portable_on_machine2",
    "cache_portability_verified",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    row = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }
    if expected_sha is not None:
        if not HEX64.fullmatch(expected_sha):
            raise ValueError(f"invalid expected SHA256: {expected_sha!r}")
        if row["sha256"] != expected_sha:
            raise RuntimeError(f"artifact SHA256 drift: {row}")
    return row


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def all_boolean_gates_pass(value: dict[str, Any], label: str) -> bool:
    gates = value.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise RuntimeError(f"{label} must contain non-empty gates")
    non_boolean = sorted(key for key, item in gates.items() if not isinstance(item, bool))
    if non_boolean:
        raise RuntimeError(f"{label} has non-boolean gates: {non_boolean}")
    return all(gates.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--portability-summary", type=Path, required=True)
    parser.add_argument("--portability-sha256", required=True)
    parser.add_argument("--portability-logical-path", required=True)
    parser.add_argument("--target-test90-result", type=Path, required=True)
    parser.add_argument("--target-test90-sha256", required=True)
    parser.add_argument("--target-test90-logical-path", required=True)
    parser.add_argument("--target-test90-validation", type=Path, required=True)
    parser.add_argument("--target-test90-validation-sha256", required=True)
    parser.add_argument("--target-test90-validation-logical-path", required=True)
    parser.add_argument("--runtime-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "failed",
        "schema": "phase11_segment25_machine2_benchmark_prerequisites_v1",
        "claims": {
            "benchmark_prerequisites_satisfied": False,
            "origin_cache_portability_proven_on_machine2": False,
            "target_device_resident_test90_passed": False,
            "critical_runtime_content_match": False,
            "exact_origin_image_identity": False,
        },
    }
    try:
        identities = {
            "protocol": artifact(args.protocol),
            "portability_summary": artifact(args.portability_summary, PORTABILITY_SUMMARY[1]),
            "target_test90_result": artifact(args.target_test90_result, TARGET_TEST90_RESULT[1]),
            "target_test90_post_validation": artifact(
                args.target_test90_validation, TARGET_TEST90_VALIDATION[1]
            ),
            "runtime_comparison": artifact(args.runtime_comparison, RUNTIME_COMPARISON[1]),
        }
        if identities["runtime_comparison"]["size_bytes"] != RUNTIME_COMPARISON[0]:
            raise RuntimeError("runtime comparison size drift")
        if args.portability_sha256 != PORTABILITY_SUMMARY[1]:
            raise RuntimeError("caller-supplied portability SHA256 differs from frozen protocol")
        if identities["portability_summary"]["size_bytes"] != PORTABILITY_SUMMARY[0]:
            raise RuntimeError("portability summary size drift")
        if args.target_test90_sha256 != TARGET_TEST90_RESULT[1]:
            raise RuntimeError("caller-supplied test90 result SHA256 differs from frozen protocol")
        if identities["target_test90_result"]["size_bytes"] != TARGET_TEST90_RESULT[0]:
            raise RuntimeError("target test90 result size drift")
        if args.target_test90_validation_sha256 != TARGET_TEST90_VALIDATION[1]:
            raise RuntimeError("caller-supplied test90 validation SHA256 differs from frozen protocol")
        if (
            identities["target_test90_post_validation"]["size_bytes"]
            != TARGET_TEST90_VALIDATION[0]
        ):
            raise RuntimeError("target test90 post-validation size drift")

        protocol = read_json(args.protocol)
        portability = read_json(args.portability_summary)
        test90 = read_json(args.target_test90_result)
        test90_validation = read_json(args.target_test90_validation)
        runtime = read_json(args.runtime_comparison)

        frozen = protocol.get("frozen_prerequisite_paths", {})
        path_gates = {
            "portability_path_matches_frozen_protocol": (
                args.portability_logical_path == frozen.get("cache_portability_summary")
            ),
            "test90_path_matches_frozen_protocol": (
                args.target_test90_logical_path
                == frozen.get("target_device_resident_test90_result")
            ),
            "test90_validation_path_matches_frozen_protocol": (
                args.target_test90_validation_logical_path
                == frozen.get("target_device_resident_test90_post_validation")
            ),
        }
        portability_claims = portability.get("claims", {})
        matched_claims = [
            key for key in PORTABILITY_REQUIRED_CLAIM_KEYS if portability_claims.get(key) is True
        ] if isinstance(portability_claims, dict) else []
        portability_gates_pass = all_boolean_gates_pass(portability, "cache portability summary")

        test90_claims = test90.get("claims", {})
        if not isinstance(test90_claims, dict):
            raise RuntimeError("target test90 claims must be an object")
        test90_gates_pass = all_boolean_gates_pass(test90, "target test90 result")
        test90_validation_gates_pass = all_boolean_gates_pass(
            test90_validation, "target test90 post-validation"
        )
        test90_validation_claims = test90_validation.get("claims", {})
        test90_validation_identities = test90_validation.get("identities", {})
        if not isinstance(test90_validation_claims, dict) or not isinstance(
            test90_validation_identities, dict
        ):
            raise RuntimeError("target test90 post-validation claims/identities must be objects")
        linked_raw_identity = test90_validation_identities.get("test90_result", {})
        linked_portability_identity = test90_validation_identities.get(
            "cache_portability_validation", {}
        )
        if not isinstance(linked_raw_identity, dict) or not isinstance(
            linked_portability_identity, dict
        ):
            raise RuntimeError("target test90 post-validation identity links are incomplete")
        runtime_claims = runtime.get("claims", {})
        runtime_images = runtime.get("images", {})
        if not isinstance(runtime_claims, dict) or not isinstance(runtime_images, dict):
            raise RuntimeError("runtime comparison claims/images must be objects")

        gates = {
            **path_gates,
            "protocol_target_image_exact": protocol.get("target", {}).get("image_id") == TARGET_IMAGE,
            "portability_status_passed": portability.get("status") == "passed",
            "portability_all_recorded_gates_passed": portability_gates_pass,
            "portability_required_claims_true": (
                len(matched_claims) == len(PORTABILITY_REQUIRED_CLAIM_KEYS)
            ),
            "target_test90_status_diagnostic_completed": (
                test90.get("status") == "diagnostic_completed"
            ),
            "target_test90_task_utility_passed": (
                test90.get("task_utility_diagnostic_passed") is True
            ),
            "target_test90_device_resident_intersegment_io": (
                test90_claims.get("device_resident_intersegment_io_test90") is True
            ),
            "target_test90_all_25_segments_strict_migraphx": (
                test90_claims.get("all_25_segments_strict_migraphx_placement") is True
            ),
            "target_test90_performance_claim_remains_false": (
                test90_claims.get("performance") is False
            ),
            "target_test90_native_int8_claim_remains_false": (
                test90_claims.get("native_int8_kernel_verified") is False
            ),
            "target_test90_deployment_ready_remains_false": (
                test90_claims.get("deployment_ready") is False
            ),
            "target_test90_all_recorded_gates_passed": test90_gates_pass,
            "target_test90_validation_schema_exact": (
                test90_validation.get("schema")
                == "phase11_segment25_crossnode_device_test90_validation_v1"
            ),
            "target_test90_validation_status_passed": test90_validation.get("status") == "passed",
            "target_test90_validation_errors_empty": test90_validation.get("errors") == [],
            "target_test90_validation_all_recorded_gates_passed": (
                test90_validation_gates_pass
            ),
            "target_test90_validation_cache_portability_claim": (
                test90_validation_claims.get("cross_node_cache_portability_verified") is True
            ),
            "target_test90_validation_task_utility_claim": (
                test90_validation_claims.get("cross_node_task_utility_test90_verified") is True
            ),
            "target_test90_validation_device_io_claim": (
                test90_validation_claims.get("device_resident_intersegment_io_test90") is True
            ),
            "target_test90_validation_performance_remains_false": (
                test90_validation_claims.get("performance") is False
            ),
            "target_test90_validation_native_int8_remains_false": (
                test90_validation_claims.get("native_int8_kernel_verified") is False
            ),
            "target_test90_validation_deployment_ready_remains_false": (
                test90_validation_claims.get("deployment_ready") is False
            ),
            "target_test90_validation_links_raw_result_sha": (
                linked_raw_identity.get("sha256") == args.target_test90_sha256
                and linked_raw_identity.get("size_bytes")
                == identities["target_test90_result"]["size_bytes"]
            ),
            "target_test90_validation_links_portability_sha": (
                linked_portability_identity.get("sha256") == PORTABILITY_SUMMARY[1]
                and linked_portability_identity.get("size_bytes") == PORTABILITY_SUMMARY[0]
            ),
            "runtime_status_content_match": (
                runtime.get("status") == "critical_runtime_content_match"
            ),
            "runtime_selected_content_match": runtime.get("selected_runtime_content_match") is True,
            "runtime_selected_content_sha_exact": (
                runtime.get("selected_runtime_content_sha256") == RUNTIME_CONTENT_SHA
            ),
            "runtime_critical_ort_migraphx_content_match": (
                runtime_claims.get("critical_ort_migraphx_runtime_content_match") is True
            ),
            "runtime_target_image_exact": runtime_images.get("target") == TARGET_IMAGE,
            "runtime_does_not_claim_exact_origin_image": (
                runtime_claims.get("exact_image_identity_match") is False
                and runtime_images.get("exact_image_identity_match") is False
            ),
        }
        passed = all(gates.values())
        result.update(
            {
                "status": "passed" if passed else "failed",
                "identities": identities,
                "logical_source_paths": {
                    "portability_summary": args.portability_logical_path,
                    "target_test90_result": args.target_test90_logical_path,
                    "target_test90_post_validation": args.target_test90_validation_logical_path,
                },
                "matched_portability_claim_keys": matched_claims,
                "gates": gates,
                "evidence_boundary": {
                    "matched_runtime_rebuild_not_exact_origin_image": True,
                    "prerequisite_pass_does_not_itself_create_a_performance_claim": True,
                    "prerequisite_artifacts_are_hard_locked_and_repeated_by_caller_sha256": True,
                    "test90_post_validation_is_the_authoritative_task_prerequisite": True,
                },
            }
        )
        result["claims"].update(
            {
                "benchmark_prerequisites_satisfied": passed,
                "origin_cache_portability_proven_on_machine2": (
                    gates["portability_status_passed"]
                    and gates["portability_all_recorded_gates_passed"]
                    and gates["portability_required_claims_true"]
                ),
                "target_device_resident_test90_passed": (
                    gates["target_test90_status_diagnostic_completed"]
                    and gates["target_test90_task_utility_passed"]
                    and gates["target_test90_device_resident_intersegment_io"]
                    and gates["target_test90_all_25_segments_strict_migraphx"]
                    and gates["target_test90_performance_claim_remains_false"]
                    and gates["target_test90_native_int8_claim_remains_false"]
                    and gates["target_test90_deployment_ready_remains_false"]
                    and gates["target_test90_all_recorded_gates_passed"]
                    and gates["target_test90_validation_schema_exact"]
                    and gates["target_test90_validation_status_passed"]
                    and gates["target_test90_validation_errors_empty"]
                    and gates["target_test90_validation_all_recorded_gates_passed"]
                    and gates["target_test90_validation_cache_portability_claim"]
                    and gates["target_test90_validation_task_utility_claim"]
                    and gates["target_test90_validation_device_io_claim"]
                    and gates["target_test90_validation_performance_remains_false"]
                    and gates["target_test90_validation_native_int8_remains_false"]
                    and gates["target_test90_validation_deployment_ready_remains_false"]
                    and gates["target_test90_validation_links_raw_result_sha"]
                    and gates["target_test90_validation_links_portability_sha"]
                ),
                "critical_runtime_content_match": (
                    gates["runtime_status_content_match"]
                    and gates["runtime_selected_content_match"]
                    and gates["runtime_selected_content_sha_exact"]
                ),
            }
        )
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
