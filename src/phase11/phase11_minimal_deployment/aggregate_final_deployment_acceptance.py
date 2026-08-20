#!/usr/bin/env python3
"""Read-only, fail-closed aggregation of final K100 deployment acceptance.

This tool never changes a bundle or an acceptance directory.  It re-hashes the
complete bundle payload and locks/validates the six required acceptance
receipts for both M5 and FP16 full before publishing a new aggregate directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "phase11_k100_final_deployment_acceptance_aggregate_v1"
BUNDLE_SCHEMA = "phase11_k100_minimal_deployment_bundle_v1"
EXCLUSION_SCHEMA = "phase11_k100_invalid_environmental_run_exclusions_v1"
OPERATIONAL_FAILURE_SCHEMA = "phase11_k100_operational_tool_failure_exclusion_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_LAYOUT = {
    "static_verify": "static_verification.json",
    "cold_start_5": "cold_start_5/result.json",
    "recovery_3": "recovery_3/result.json",
    "smoke_k100_2": "smoke_k100_2/result.json",
    "smoke_k100_3": "smoke_k100_3/result.json",
    "stability_60min": "stability_60min/result.json",
}
K1003_LAUNCH_ATTESTATION = "smoke_k100_3/container_launch_attestation.json"
CONTAINER_PATH = (
    "/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
)
PREENTRYPOINT_TOKEN = "docker_run_environment_before_entrypoint_v1"
EXPECTED_EXCLUSION_FAILURES = {
    "bdc7_with_node3_host_hyhal_lsmod_recursion_before_model": (
        "runtime_import_before_model_inference"
    ),
    "bdc7_without_host_hyhal_hip100_no_rocm_device": (
        "model_session_creation_before_successful_inference"
    ),
}
EXPECTED_STABILITY_GATES = {
    "duration_at_least_3600_seconds",
    "at_least_one_inference",
    "zero_inference_errors",
    "zero_nonfinite_outputs",
    "zero_fixed_prediction_drifts",
    "telemetry_interval_eq_5_seconds",
    "telemetry_sample_count_sufficient",
    "temperature_power_vram_present_every_sample",
    "telemetry_intervals_strictly_positive",
    "telemetry_minimum_interval_ge_4_seconds",
    "telemetry_maximum_interval_le_7_5_seconds",
    "image_identity_attested",
}
LSMOD_PAIR = (
    819_664,
    "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pair(row: dict) -> tuple[int, str]:
    return int(row.get("size_bytes", -1)), str(row.get("sha256", ""))


def contained(root: Path, child: Path) -> bool:
    return child == root or root in child.parents


def regular_file(path: Path, role: str) -> Path:
    unresolved = path.absolute()
    if unresolved.is_symlink():
        raise RuntimeError(f"{role} must not be a symlink: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"{role} must be a regular non-symlink file: {unresolved}")
    return resolved


def root_directory(path: Path, role: str) -> Path:
    unresolved = path.absolute()
    if unresolved.is_symlink():
        raise RuntimeError(f"{role} must not be a symlink: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"{role} must be a directory: {resolved}")
    return resolved


def resolve_below(root: Path, relative: str, role: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"invalid {role} relative path: {relative!r}")
    lexical = root
    for part in pure.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise RuntimeError(f"{role} path contains a symlink: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"missing or unreadable {role}: {relative}") from exc
    if not contained(root, resolved) or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"{role} escapes its root or is not a regular file: {relative}")
    return resolved


def resolve_node_below(root: Path, relative: str, role: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"invalid {role} relative path: {relative!r}")
    lexical = root
    for part in pure.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise RuntimeError(f"{role} path contains a symlink: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"missing or unreadable {role}: {relative}") from exc
    if not contained(root, resolved) or resolved.is_symlink() or not (
        resolved.is_file() or resolved.is_dir()
    ):
        raise RuntimeError(f"{role} escapes its root or has an unsupported type: {relative}")
    return resolved


def directory_tree_identity(directory: Path) -> dict:
    directory = root_directory(directory, "locked directory tree")
    rows = []
    total = 0
    for parent, names, files in os.walk(directory, followlinks=False):
        parent_path = Path(parent)
        for name in names:
            node = parent_path / name
            if node.is_symlink():
                raise RuntimeError(f"locked directory tree contains a symlink: {node}")
        for name in files:
            node = parent_path / name
            if node.is_symlink() or not node.is_file():
                raise RuntimeError(f"locked directory tree has a non-regular file: {node}")
            row = {
                "path": node.relative_to(directory).as_posix(),
                "size_bytes": node.stat().st_size,
                "sha256": sha256(node),
            }
            rows.append(row)
            total += row["size_bytes"]
    rows.sort(key=lambda row: row["path"])
    tree_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "file_count": len(rows),
        "total_bytes": total,
        "tree_sha256": tree_sha,
        "files": rows,
    }


def load_json(path: Path, role: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid {role} JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} JSON root must be an object: {path}")
    return value


def exact_bool(value: Any, expected: bool, context: str) -> None:
    if value is not expected:
        raise RuntimeError(f"{context} must be literal {expected}")


def finite_number(value: Any, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise RuntimeError(f"{context} is outside its finite range: {value!r}")
    return result


def validate_timestamps(payload: dict, context: str) -> None:
    try:
        started = datetime.fromisoformat(str(payload["started_at_utc"]))
        ended = datetime.fromisoformat(str(payload["ended_at_utc"]))
    except Exception as exc:
        raise RuntimeError(f"{context} timestamps are missing/invalid") from exc
    if started.tzinfo is None or ended.tzinfo is None or ended < started:
        raise RuntimeError(f"{context} timestamp order/timezone drift")


def inventory_bundle(bundle: Path, expected_manifest_sha256: str, label: str) -> dict:
    if not SHA256_PATTERN.fullmatch(expected_manifest_sha256):
        raise RuntimeError(f"{label} detached manifest SHA256 is invalid")
    bundle = root_directory(bundle, f"{label} bundle")
    manifest_path = resolve_below(bundle, "manifest.json", f"{label} manifest")
    manifest_row = identity(manifest_path)
    if manifest_row["sha256"] != expected_manifest_sha256:
        raise RuntimeError(f"{label} detached manifest SHA256 drift")
    manifest = load_json(manifest_path, f"{label} bundle manifest")
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("status") != (
        "static_bundle_identity_locked"
    ):
        raise RuntimeError(f"{label} bundle schema/status drift")
    claims = manifest.get("claims", {})
    exact_bool(claims.get("static_bundle_ready"), True, f"{label} static_bundle_ready")
    exact_bool(claims.get("runtime_validated"), False, f"{label} static runtime_validated")
    exact_bool(claims.get("deployment_ready"), False, f"{label} static deployment_ready")
    boundaries = manifest.get("claim_boundaries", {})
    if boundaries.get("historical_int8_strict_logits") != "failed_immutable":
        raise RuntimeError(f"{label} historical strict-failure boundary drift")
    if boundaries.get("provider_placement_does_not_prove_kernel_precision") is not True:
        raise RuntimeError(f"{label} provider/kernel evidence boundary drift")

    execution = manifest.get("execution", {})
    kind = execution.get("kind")
    if label == "M5":
        if kind != "segment25" or execution.get("source_candidate_id") != "M5":
            raise RuntimeError("M5 aggregate input is not the admitted M5 segment25 bundle")
        if boundaries.get("native_int8_kernel") != "unverified":
            raise RuntimeError("M5 static bundle must not infer native INT8 kernels")
    elif label == "FP16":
        if kind != "fp16_full":
            raise RuntimeError("FP16 aggregate input is not an FP16-full bundle")
        if execution.get("admission_state") != (
            "task_admitted_runtime_deployment_acceptance_pending"
        ):
            raise RuntimeError("FP16 static admission state drift")
        if boundaries.get("fp16_full_precision_semantics") != "verified_frozen_artifact":
            raise RuntimeError("FP16 frozen precision semantics are not verified")
        if boundaries.get("native_int8_kernel") != "not_applicable":
            raise RuntimeError("FP16 native-INT8 claim boundary drift")
    else:
        raise RuntimeError(f"unsupported candidate label: {label}")

    rows = manifest.get("files", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{label} bundle inventory is empty")
    expected: dict[str, dict] = {}
    for row in rows:
        relative = str(row.get("path", ""))
        if relative in expected:
            raise RuntimeError(f"{label} duplicate bundle inventory path: {relative}")
        expected[relative] = row
    actual: set[str] = set()
    for directory, names, files in os.walk(bundle, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"{label} bundle contains a directory symlink: {candidate}")
        for name in files:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"{label} bundle contains a file symlink: {candidate}")
            relative = candidate.relative_to(bundle).as_posix()
            if relative != "manifest.json":
                actual.add(relative)
    if actual != set(expected):
        raise RuntimeError(
            f"{label} bundle payload set drift: missing={sorted(set(expected)-actual)}, "
            f"unexpected={sorted(actual-set(expected))}"
        )
    total = 0
    for relative, wanted in expected.items():
        path = resolve_below(bundle, relative, f"{label} bundle payload")
        observed = identity(path)
        if pair(observed) != pair(wanted):
            raise RuntimeError(f"{label} bundle payload identity drift: {relative}")
        total += observed["size_bytes"]
    if len(actual) != int(manifest.get("payload_file_count", -1)) or total != int(
        manifest.get("payload_total_bytes", -1)
    ):
        raise RuntimeError(f"{label} bundle payload count/byte total drift")
    expected_prediction = str(
        manifest.get("fixed_sample", {}).get("expected_prediction_array_sha256", "")
    )
    if not SHA256_PATTERN.fullmatch(expected_prediction):
        raise RuntimeError(f"{label} expected fixed prediction SHA drift")
    runtime_contract = manifest.get("runtime_contract", {})
    if runtime_contract.get("onnxruntime") != "1.19.2" or runtime_contract.get(
        "provider"
    ) != "MIGraphXExecutionProvider":
        raise RuntimeError(f"{label} runtime ORT/provider contract drift")
    shim_contract = runtime_contract.get("static_lsmod", {})
    if shim_contract.get("path") != "tools/bin/lsmod" or pair(shim_contract) != LSMOD_PAIR:
        raise RuntimeError(f"{label} locked static lsmod contract drift")
    shim_inventory = expected.get("tools/bin/lsmod")
    if shim_inventory is None or pair(shim_inventory) != LSMOD_PAIR or pair(
        shim_contract
    ) != pair(shim_inventory):
        raise RuntimeError(f"{label} static lsmod payload identity drift")
    origin_row = runtime_contract.get("origin_runtime_fingerprint", {})
    origin_path = resolve_below(
        bundle, str(origin_row.get("path", "")), f"{label} origin runtime fingerprint"
    )
    if pair(identity(origin_path)) != pair(origin_row):
        raise RuntimeError(f"{label} origin runtime fingerprint identity drift")
    origin = load_json(origin_path, f"{label} origin runtime fingerprint")
    if origin.get("schema") != "phase11_k100_deployment_runtime_fingerprint_v1" or origin.get(
        "status"
    ) != "captured":
        raise RuntimeError(f"{label} origin runtime fingerprint schema/status drift")
    if origin.get("portable_runtime_signature_sha256") != runtime_contract.get(
        "portable_runtime_signature_sha256"
    ) or origin.get("portable_fields", {}).get("official_image_id") != runtime_contract.get(
        "official_image_id"
    ):
        raise RuntimeError(f"{label} origin runtime signature/image drift")
    exact_bool(
        origin.get("observation", {}).get("image_identity_attested"),
        True,
        f"{label} origin image attestation",
    )
    if origin.get("observation", {}).get("attested_image_id") != runtime_contract.get(
        "official_image_id"
    ):
        raise RuntimeError(f"{label} origin attested image identity drift")
    portable = origin.get("portable_fields", {})
    if portable.get("onnxruntime") != "1.19.2" or pair(
        portable.get("static_lsmod", {})
    ) != LSMOD_PAIR or pair(origin.get("static_lsmod", {})) != LSMOD_PAIR:
        raise RuntimeError(f"{label} origin runtime portable contract drift")
    recomputed_signature = hashlib.sha256(
        json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed_signature != origin.get("portable_runtime_signature_sha256"):
        raise RuntimeError(f"{label} origin portable runtime signature is not reproducible")
    capture_row = expected.get("tools/capture_k100_runtime_fingerprint.py")
    if capture_row is None or pair(origin.get("generator", {})) != pair(capture_row):
        raise RuntimeError(f"{label} origin runtime fingerprint generator drift")
    return {
        "root": bundle,
        "manifest": manifest,
        "manifest_identity": manifest_row,
        "bundle_id": str(manifest.get("bundle_id", "")),
        "kind": str(kind),
        "image_id": str(manifest.get("runtime_contract", {}).get("official_image_id", "")),
        "expected_prediction_sha256": expected_prediction,
        "origin_runtime_fingerprint": origin,
        "payload_file_count": len(actual),
        "payload_total_bytes": total,
    }


def validate_inference_receipt(receipt: dict, context: dict, role: str) -> None:
    if receipt.get("schema") != "phase11_k100_inference_receipt_v1" or receipt.get(
        "status"
    ) != "passed":
        raise RuntimeError(f"{role} inference receipt schema/status drift")
    if receipt.get("bundle_id") != context["bundle_id"] or receipt.get(
        "execution_kind"
    ) != context["kind"]:
        raise RuntimeError(f"{role} inference bundle lineage drift")
    if pair(receipt.get("manifest", {})) != pair(context["manifest_identity"]):
        raise RuntimeError(f"{role} inference manifest identity drift")
    runtime = receipt.get("runtime", {})
    if runtime.get("selected_provider") != "MIGraphXExecutionProvider":
        raise RuntimeError(f"{role} did not select MIGraphX")
    exact_bool(runtime.get("cpu_fallback_disabled"), True, f"{role} CPU fallback")
    exact_bool(runtime.get("image_identity_attested"), True, f"{role} image attestation")
    if runtime.get("expected_official_image_id") != context["image_id"] or runtime.get(
        "attested_image_id"
    ) != context["image_id"]:
        raise RuntimeError(f"{role} official image identity drift")
    if pair(runtime.get("static_lsmod", {})) != LSMOD_PAIR or runtime.get(
        "static_lsmod", {}
    ).get("path_prepend_active") is not True:
        raise RuntimeError(f"{role} static lsmod runtime contract drift")
    fixed = receipt.get("fixed_sample_gate", {})
    exact_bool(fixed.get("input_matches_bundled_fixed_sample"), True, f"{role} fixed input")
    exact_bool(fixed.get("prediction_matches_expected"), True, f"{role} fixed prediction")
    if fixed.get("expected_prediction_array_sha256") != context[
        "expected_prediction_sha256"
    ] or fixed.get("observed_prediction_array_sha256") != context[
        "expected_prediction_sha256"
    ]:
        raise RuntimeError(f"{role} fixed prediction SHA drift")
    input_row = receipt.get("input", {})
    if input_row.get("shape") != [1, 6, 224, 224] or input_row.get("dtype") != "float32":
        raise RuntimeError(f"{role} input shape/dtype drift")
    exact_bool(input_row.get("finite"), True, f"{role} finite input")
    exact_bool(
        input_row.get("preprocessing_applied_by_cli"), False, f"{role} preprocessing boundary"
    )
    output = receipt.get("output", {})
    if (
        output.get("logits_shape") != [1, 2, 224, 224]
        or output.get("prediction_shape") != [1, 224, 224]
        or output.get("logits_dtype") != "float32"
        or output.get("prediction_dtype") != "uint8"
    ):
        raise RuntimeError(f"{role} output contract drift")
    exact_bool(output.get("finite_logits"), True, f"{role} finite logits")
    boundaries = receipt.get("evidence_boundary", {})
    exact_bool(
        boundaries.get("strict_logit_equivalence_inferred_from_this_run"),
        False,
        f"{role} strict-logit boundary",
    )
    exact_bool(
        boundaries.get("native_kernel_precision_inferred_from_provider_placement"),
        False,
        f"{role} kernel boundary",
    )


def common_acceptance_receipt(
    payload: dict, schema: str, context: dict, role: str, claim: str
) -> None:
    if payload.get("schema") != schema or payload.get("status") != "passed":
        raise RuntimeError(f"{role} schema/status drift")
    validate_timestamps(payload, role)
    if payload.get("bundle_id") != context["bundle_id"] or payload.get(
        "manifest_sha256"
    ) != context["manifest_identity"]["sha256"]:
        raise RuntimeError(f"{role} bundle/manifest lineage drift")
    exact_bool(payload.get("claims", {}).get(claim), True, f"{role} admission claim")
    exact_bool(
        payload.get("claims", {}).get("deployment_ready"),
        False,
        f"{role} local deployment-ready boundary",
    )


def validate_static(payload: dict, context: dict) -> dict:
    if payload.get("schema") != "phase11_k100_bundle_static_verification_v1" or payload.get(
        "status"
    ) != "passed":
        raise RuntimeError("static verification schema/status drift")
    if payload.get("bundle_id") != context["bundle_id"] or payload.get(
        "execution_kind"
    ) != context["kind"]:
        raise RuntimeError("static verification bundle lineage drift")
    if pair(payload.get("manifest", {})) != pair(context["manifest_identity"]):
        raise RuntimeError("static verification manifest identity drift")
    if int(payload.get("payload_file_count", -1)) != context[
        "payload_file_count"
    ] or int(payload.get("payload_total_bytes", -1)) != context["payload_total_bytes"]:
        raise RuntimeError("static verification payload totals drift")
    claims = payload.get("claims", {})
    exact_bool(claims.get("static_bundle_identity_verified"), True, "static identity claim")
    for key in ("runtime_validated", "cache_portability_verified", "deployment_ready"):
        exact_bool(claims.get(key), False, f"static {key} boundary")
    return {"passed": True}


def validate_cold(payload: dict, context: dict) -> dict:
    common_acceptance_receipt(
        payload,
        "phase11_k100_cold_start_5_v1",
        context,
        "cold-start-5",
        "five_fresh_processes_passed",
    )
    protocol = payload.get("protocol", {})
    if protocol.get("fresh_python_processes") != 5:
        raise RuntimeError("cold-start process count drift")
    exact_bool(protocol.get("full_payload_hash_each_process"), True, "cold full-payload hash")
    exact_bool(
        protocol.get("fixed_prediction_sha_gate_each_process"), True, "cold prediction gate"
    )
    trials = payload.get("trials", [])
    if len(trials) != 5 or [row.get("trial") for row in trials] != list(range(1, 6)):
        raise RuntimeError("cold-start trials must be exactly 1..5")
    load_times = []
    for index, row in enumerate(trials, start=1):
        if row.get("returncode") != 0:
            raise RuntimeError(f"cold-start trial {index} return code drift")
        validate_inference_receipt(row.get("receipt", {}), context, f"cold trial {index}")
        load_times.append(
            finite_number(
                row["receipt"].get("timing", {}).get("session_load_seconds"),
                f"cold trial {index} load time",
                0.0,
            )
        )
    return {
        "passed": True,
        "session_load_seconds_min": min(load_times),
        "session_load_seconds_max": max(load_times),
    }


def validate_recovery(payload: dict, context: dict) -> dict:
    common_acceptance_receipt(
        payload,
        "phase11_k100_recovery_3_v1",
        context,
        "recovery-3",
        "three_active_termination_reloads_passed",
    )
    protocol = payload.get("protocol", {})
    if protocol.get("active_termination_cycles") != 3:
        raise RuntimeError("recovery active-termination cycle count drift")
    exact_bool(protocol.get("fresh_reload_process_each_cycle"), True, "recovery fresh reload")
    cycles = payload.get("cycles", [])
    if len(cycles) != 3 or [row.get("cycle") for row in cycles] != [1, 2, 3]:
        raise RuntimeError("recovery cycles must be exactly 1..3")
    for index, row in enumerate(cycles, start=1):
        exact_bool(row.get("worker_ready"), True, f"recovery cycle {index} worker-ready")
        if row.get("active_termination_returncode") not in {-15, -9}:
            raise RuntimeError(f"recovery cycle {index} was not actively signal-terminated")
        if row.get("reload_returncode") != 0:
            raise RuntimeError(f"recovery cycle {index} reload return code drift")
        exact_bool(
            row.get("reload_fixed_prediction_passed"),
            True,
            f"recovery cycle {index} fixed prediction",
        )
        validate_inference_receipt(
            row.get("reload_receipt", {}), context, f"recovery cycle {index} reload"
        )
    return {"passed": True, "active_termination_cycles": 3}


def validate_runtime_fingerprint(payload: dict, context: dict, role: str) -> None:
    if payload.get("schema") != "phase11_k100_deployment_runtime_fingerprint_v1" or payload.get(
        "status"
    ) != "captured":
        raise RuntimeError(f"{role} target fingerprint schema/status drift")
    portable = payload.get("portable_fields", {})
    if portable.get("official_image_id") != context["image_id"] or portable.get(
        "onnxruntime"
    ) != "1.19.2":
        raise RuntimeError(f"{role} target runtime image/ORT drift")
    if pair(portable.get("static_lsmod", {})) != LSMOD_PAIR or pair(
        payload.get("static_lsmod", {})
    ) != LSMOD_PAIR:
        raise RuntimeError(f"{role} target static lsmod drift")
    observation = payload.get("observation", {})
    exact_bool(observation.get("image_identity_attested"), True, f"{role} image attestation")
    if observation.get("attested_image_id") != context["image_id"]:
        raise RuntimeError(f"{role} target attested image drift")
    expected_signature = context["manifest"].get("runtime_contract", {}).get(
        "portable_runtime_signature_sha256"
    )
    if payload.get("portable_runtime_signature_sha256") != expected_signature:
        raise RuntimeError(f"{role} target portable runtime signature drift")
    if portable != context["origin_runtime_fingerprint"].get("portable_fields"):
        raise RuntimeError(f"{role} portable runtime fields differ from bundle origin")
    recomputed_signature = hashlib.sha256(
        json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed_signature != expected_signature:
        raise RuntimeError(f"{role} target runtime signature is not reproducible")
    capture_row = next(
        (
            row
            for row in context["manifest"].get("files", [])
            if row.get("path") == "tools/capture_k100_runtime_fingerprint.py"
        ),
        None,
    )
    if capture_row is None or pair(payload.get("generator", {})) != pair(capture_row):
        raise RuntimeError(f"{role} target fingerprint generator identity drift")


def validate_smoke(
    payload: dict, context: dict, node_label: str, fingerprint_path: Path
) -> dict:
    role = f"{node_label} smoke"
    common_acceptance_receipt(
        payload,
        "phase11_k100_cross_node_smoke_v1",
        context,
        role,
        "target_cache_smoke_passed",
    )
    if payload.get("node_label") != node_label or not str(payload.get("hostname", "")).strip():
        raise RuntimeError(f"{role} node/hostname identity drift")
    exact_bool(payload.get("runtime_signature_match"), True, f"{role} runtime signature")
    if payload.get("inference_returncode") != 0:
        raise RuntimeError(f"{role} inference return code drift")
    exact_bool(payload.get("fixed_prediction_passed"), True, f"{role} fixed prediction")
    exact_bool(
        payload.get("claims", {}).get("cache_portability_verified"),
        True,
        f"{role} portability claim",
    )
    validate_inference_receipt(payload.get("inference_receipt", {}), context, role)
    fingerprint = load_json(fingerprint_path, f"{role} target fingerprint")
    validate_runtime_fingerprint(fingerprint, context, role)
    return {
        "passed": True,
        "node_label": node_label,
        "hostname": payload["hostname"],
        "target_runtime_fingerprint": identity(fingerprint_path),
    }


def require_argv_pair(argv: list, option: str, value: str, role: str) -> int:
    hits = [index for index in range(len(argv) - 1) if argv[index : index + 2] == [option, value]]
    if len(hits) != 1:
        raise RuntimeError(f"{role} must contain exactly one {option} {value!r} pair")
    return hits[0]


def validate_k1003_launch_attestation(
    payload: dict, path: Path, smoke_path: Path, context: dict, smoke: dict
) -> dict:
    role = "K100-3 pre-entrypoint container launch"
    if payload.get("schema") != "phase11_k100_container_launch_attestation_v1" or payload.get(
        "status"
    ) != "passed":
        raise RuntimeError(f"{role} schema/status drift")
    validate_timestamps(payload, role)
    launch_started = datetime.fromisoformat(str(payload["started_at_utc"]))
    launch_ended = datetime.fromisoformat(str(payload["ended_at_utc"]))
    smoke_started = datetime.fromisoformat(str(smoke["started_at_utc"]))
    smoke_ended = datetime.fromisoformat(str(smoke["ended_at_utc"]))
    if not (launch_started <= smoke_started <= smoke_ended <= launch_ended):
        raise RuntimeError(f"{role} does not temporally enclose the smoke receipt")
    launcher = Path(__file__).resolve().parent / "launch_cross_node_smoke_container.py"
    if pair(payload.get("generator", {})) != pair(identity(launcher)):
        raise RuntimeError(f"{role} generator identity drift")
    launcher_hostname = str(payload.get("hostname", "")).strip()
    smoke_hostname = str(smoke.get("hostname", "")).strip()
    if (
        payload.get("node_label") != "K100-3"
        or not launcher_hostname
        or not smoke_hostname
    ):
        raise RuntimeError(f"{role} node/hostname lineage drift")
    # The outer attestation is written by the physical host, while the smoke
    # receipt is written inside Docker and therefore records the container
    # hostname. Requiring equality conflates two distinct identities.
    if launcher_hostname == smoke_hostname:
        raise RuntimeError(f"{role} host/container hostname identities collapsed")
    device = payload.get("device")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise RuntimeError(f"{role} device index drift")
    if payload.get("container_bundle_dir") != "/bundle" or payload.get(
        "container_output_dir"
    ) != "/acceptance/container_smoke":
        raise RuntimeError(f"{role} container bundle/output path drift")
    host_output = str(payload.get("host_output_dir", ""))
    ownership = payload.get("host_output_ownership", {})
    if not host_output.startswith("/") or ownership.get(
        "created_exclusively_before_docker"
    ) is not True:
        raise RuntimeError(f"{role} host output was not created before Docker")
    for field in ("uid", "gid"):
        value = ownership.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{role} host output {field} drift")
    if not re.fullmatch(r"0o[0-7]{3}", str(ownership.get("mode", ""))):
        raise RuntimeError(f"{role} host output mode drift")
    if payload.get("bundle_id") != context["bundle_id"] or pair(
        payload.get("manifest", {})
    ) != pair(context["manifest_identity"]):
        raise RuntimeError(f"{role} bundle/manifest lineage drift")
    if payload.get("official_image_id") != context["image_id"]:
        raise RuntimeError(f"{role} official image drift")
    inspect = payload.get("docker_image_inspect", {})
    if inspect.get("returncode") != 0 or inspect.get("observed_image_id") != context[
        "image_id"
    ]:
        raise RuntimeError(f"{role} Docker image attestation failed")
    if payload.get("docker_returncode") != 0:
        raise RuntimeError(f"{role} Docker run did not pass")
    launcher_logs = payload.get("launcher_logs", {})
    for key, filename in (
        ("stdout", "container_launcher_stdout.log"),
        ("stderr", "container_launcher_stderr.log"),
    ):
        log_path = regular_file(path.parent / filename, f"{role} {key} log")
        if pair(launcher_logs.get(key, {})) != pair(identity(log_path)):
            raise RuntimeError(f"{role} {key} log identity drift")
    environment = payload.get("pre_entrypoint_environment", {})
    expected_environment = {
        "PATH": CONTAINER_PATH,
        "PHASE11_K100_IMAGE_ID": context["image_id"],
        "PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION": PREENTRYPOINT_TOKEN,
    }
    if environment != expected_environment:
        raise RuntimeError(f"{role} locked environment drift")
    static_lsmod = payload.get("static_lsmod", {})
    if pair(static_lsmod) != LSMOD_PAIR or static_lsmod.get(
        "container_path"
    ) != "/bundle/tools/bin/lsmod":
        raise RuntimeError(f"{role} static lsmod drift")
    hyhal = payload.get("host_hyhal_mount", {})
    if (
        hyhal.get("target") != "/opt/hyhal"
        or hyhal.get("read_only") is not True
        or not str(hyhal.get("source", "")).startswith("/")
        or not str(hyhal.get("requested_source", "")).startswith("/")
        or hyhal.get("symlink_resolved_before_docker") is not True
        or not isinstance(hyhal.get("requested_source_was_symlink"), bool)
    ):
        raise RuntimeError(f"{role} host hyhal realpath/read-only mount drift")
    claims = payload.get("claims", {})
    for name in (
        "path_injected_by_docker_run_before_entrypoint",
        "locked_static_lsmod_first_on_path",
        "host_hyhal_mounted_read_only",
        "host_output_created_exclusively_before_docker",
        "container_writes_isolated_subdirectory",
        "formal_receipts_copied_without_byte_drift",
    ):
        exact_bool(claims.get(name), True, f"{role} {name}")
    for name in (
        "invalid_prior_launcher_runs_count_toward_acceptance",
        "no_hyhal_diagnostics_count_toward_acceptance",
    ):
        exact_bool(claims.get(name), False, f"{role} {name}")
    if pair(payload.get("smoke_result", {})) != pair(identity(smoke_path)):
        raise RuntimeError(f"{role} smoke receipt identity drift")
    formal_fingerprint_path = regular_file(
        path.parent / "target_runtime_fingerprint.json", f"{role} formal fingerprint"
    )
    formal_fingerprint = load_json(formal_fingerprint_path, f"{role} formal fingerprint")
    if (
        str(formal_fingerprint.get("observation", {}).get("hostname", "")).strip()
        != smoke_hostname
    ):
        raise RuntimeError(f"{role} container hostname/fingerprint lineage drift")
    child_result_path = regular_file(
        path.parent / "container_smoke/result.json", f"{role} container result"
    )
    child_fingerprint_path = regular_file(
        path.parent / "container_smoke/target_runtime_fingerprint.json",
        f"{role} container fingerprint",
    )
    if pair(payload.get("formal_runtime_fingerprint", {})) != pair(
        identity(formal_fingerprint_path)
    ):
        raise RuntimeError(f"{role} formal fingerprint identity drift")
    if pair(payload.get("container_smoke_result", {})) != pair(
        identity(child_result_path)
    ) or pair(identity(child_result_path)) != pair(identity(smoke_path)):
        raise RuntimeError(f"{role} container/formal smoke receipt copy drift")
    if pair(payload.get("container_runtime_fingerprint", {})) != pair(
        identity(child_fingerprint_path)
    ) or pair(identity(child_fingerprint_path)) != pair(identity(formal_fingerprint_path)):
        raise RuntimeError(f"{role} container/formal runtime fingerprint copy drift")
    if payload.get("failure") is not None:
        raise RuntimeError(f"{role} passed receipt retains a failure")

    argv = payload.get("docker_argv")
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise RuntimeError(f"{role} Docker argv must be a string array")
    if len(argv) < 4 or Path(argv[0]).name != "docker" or argv[1:3] != ["run", "--rm"]:
        raise RuntimeError(f"{role} Docker argv prefix drift")
    path_index = require_argv_pair(argv, "-e", f"PATH={CONTAINER_PATH}", role)
    require_argv_pair(
        argv, "-e", f"PHASE11_K100_IMAGE_ID={context['image_id']}", role
    )
    require_argv_pair(
        argv,
        "-e",
        f"PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION={PREENTRYPOINT_TOKEN}",
        role,
    )
    entrypoint_index = require_argv_pair(
        argv, "--entrypoint", "/bundle/tools/run_cross_node_smoke.sh", role
    )
    if path_index > entrypoint_index:
        raise RuntimeError(f"{role} PATH option appears after entrypoint option")
    image_positions = [index for index, value in enumerate(argv) if value == context["image_id"]]
    if len(image_positions) != 1 or image_positions[0] <= entrypoint_index:
        raise RuntimeError(f"{role} immutable image argument placement drift")
    if not any(value.endswith(":/bundle:ro") for value in argv):
        raise RuntimeError(f"{role} immutable bundle read-only mount missing")
    if f"{hyhal['source']}:/opt/hyhal:ro" not in argv:
        raise RuntimeError(f"{role} host hyhal read-only mount missing")
    if f"{host_output}:/acceptance" not in argv:
        raise RuntimeError(f"{role} pre-created host output mount missing")
    require_argv_pair(argv, "--bundle", "/bundle", role)
    require_argv_pair(argv, "--output-dir", payload["container_output_dir"], role)
    require_argv_pair(argv, "--device", str(device), role)
    require_argv_pair(argv, "--node-label", "K100-3", role)
    require_argv_pair(
        argv,
        "--expected-manifest-sha256",
        context["manifest_identity"]["sha256"],
        role,
    )
    return {
        "passed": True,
        "attestation": identity(path),
        "pre_entrypoint_path": CONTAINER_PATH,
        "host_hyhal_mounted_read_only": True,
    }


def validate_environment_exclusion_ledger(
    root: Path,
    ledger_path: Path,
    detached_sha256: str,
    allowed_manifest_sha256: set[str],
    expected_image_ids: set[str],
) -> dict:
    """Lock failed environmental attempts without letting them satisfy a gate."""
    root = root_directory(root, "environment exclusion root")
    ledger_path = regular_file(ledger_path, "environment exclusion ledger")
    if not contained(root, ledger_path):
        raise RuntimeError("environment exclusion ledger escapes its authorized root")
    if not SHA256_PATTERN.fullmatch(detached_sha256) or sha256(ledger_path) != detached_sha256:
        raise RuntimeError("environment exclusion ledger detached SHA256 drift")
    payload = load_json(ledger_path, "environment exclusion ledger")
    if payload.get("schema") != EXCLUSION_SCHEMA or payload.get("status") != (
        "locked_invalid_environmental_runs_excluded"
    ):
        raise RuntimeError("environment exclusion ledger schema/status drift")
    ledger_builder = Path(__file__).resolve().parent / "build_environment_exclusion_ledger.py"
    if pair(payload.get("generator", {})) != pair(identity(ledger_builder)):
        raise RuntimeError("environment exclusion ledger generator identity drift")
    claims = payload.get("claims", {})
    for name in (
        "listed_runs_are_not_acceptance_evidence",
        "formal_receipts_are_not_overwritten",
        "fresh_formal_rerun_required",
    ):
        exact_bool(claims.get(name), True, f"environment exclusion {name}")
    exact_bool(
        claims.get("listed_runs_satisfy_any_deployment_gate"),
        False,
        "environment exclusion gate boundary",
    )
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_EXCLUSION_FAILURES):
        raise RuntimeError("environment exclusion ledger must contain exactly the two known runs")
    modes = [str(entry.get("failure_mode", "")) for entry in entries]
    if set(modes) != set(EXPECTED_EXCLUSION_FAILURES) or len(set(modes)) != len(modes):
        raise RuntimeError("environment exclusion failure-mode set drift")
    artifact_identities: dict[str, list[dict]] = {}
    normalized_entries = []
    used_paths: set[Path] = set()
    for entry in entries:
        mode = str(entry.get("failure_mode", ""))
        stage = EXPECTED_EXCLUSION_FAILURES[mode]
        entry_id = str(entry.get("entry_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", entry_id):
            raise RuntimeError(f"invalid environment exclusion entry_id: {entry_id!r}")
        if entry.get("failure_stage") != stage or entry.get("node_label") != "K100-3":
            raise RuntimeError(f"environment exclusion stage/node drift for {mode}")
        if entry.get("manifest_sha256") not in allowed_manifest_sha256:
            raise RuntimeError(f"environment exclusion manifest lineage drift for {mode}")
        if entry.get("official_image_id") not in expected_image_ids:
            raise RuntimeError(f"environment exclusion image lineage drift for {mode}")
        exact_bool(entry.get("accepted_environment"), False, f"{mode} environment")
        exact_bool(entry.get("counts_toward_acceptance"), False, f"{mode} acceptance count")
        exact_bool(entry.get("model_execution_started"), False, f"{mode} model execution")
        if entry.get("successful_model_inferences") != 0:
            raise RuntimeError(f"environment exclusion {mode} has a successful inference")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError(f"environment exclusion {mode} has no locked artifact")
        locked = []
        for row in artifacts:
            relative = str(row.get("path", ""))
            artifact = resolve_node_below(
                root, relative, f"environment exclusion {mode} artifact"
            )
            if artifact == ledger_path or (
                artifact.is_dir() and contained(artifact, ledger_path)
            ) or any(
                artifact == used
                or (artifact.is_dir() and contained(artifact, used))
                or (used.is_dir() and contained(used, artifact))
                for used in used_paths
            ):
                raise RuntimeError(
                    "environment exclusion artifacts must be distinct non-overlapping trees"
                )
            kind = row.get("kind")
            if kind == "file":
                if not artifact.is_file():
                    raise RuntimeError(f"environment exclusion artifact is not a file: {relative}")
                observed = identity(artifact)
                if pair(observed) != pair(row) or observed["size_bytes"] <= 0:
                    raise RuntimeError(f"environment exclusion artifact identity drift: {relative}")
                normalized = {
                    "kind": "file",
                    "path": relative,
                    "size_bytes": observed["size_bytes"],
                    "sha256": observed["sha256"],
                }
            elif kind == "directory_tree":
                if not artifact.is_dir():
                    raise RuntimeError(f"environment exclusion artifact is not a directory: {relative}")
                observed = directory_tree_identity(artifact)
                for field in ("file_count", "total_bytes", "tree_sha256"):
                    if row.get(field) != observed[field]:
                        raise RuntimeError(
                            f"environment exclusion directory-tree identity drift: {relative}"
                        )
                normalized = {
                    "kind": "directory_tree",
                    "path": relative,
                    "file_count": observed["file_count"],
                    "total_bytes": observed["total_bytes"],
                    "tree_sha256": observed["tree_sha256"],
                }
            else:
                raise RuntimeError(
                    f"environment exclusion artifact kind must be file or directory_tree: {relative}"
                )
            used_paths.add(artifact)
            locked.append(normalized)
        artifact_identities[entry_id] = locked
        normalized_entries.append(
            {
                "entry_id": entry_id,
                "failure_mode": mode,
                "failure_stage": stage,
                "node_label": "K100-3",
                "counts_toward_acceptance": False,
                "successful_model_inferences": 0,
            }
        )
    return {
        "status": "passed",
        "ledger": identity(ledger_path),
        "root": str(root),
        "entries": sorted(normalized_entries, key=lambda row: row["failure_mode"]),
        "artifacts": artifact_identities,
        "known_invalid_runs_count_toward_acceptance": False,
    }


def validate_operational_tool_failure_ledger(
    root: Path,
    ledger_path: Path,
    detached_sha256: str,
    candidate_manifests: dict[str, str],
    expected_image_ids: set[str],
) -> dict:
    root = root_directory(root, "operational tool failure root")
    ledger_path = regular_file(ledger_path, "operational tool failure ledger")
    if not contained(root, ledger_path):
        raise RuntimeError("operational tool failure ledger escapes its root")
    if not SHA256_PATTERN.fullmatch(detached_sha256) or sha256(ledger_path) != detached_sha256:
        raise RuntimeError("operational tool failure ledger detached SHA256 drift")
    payload = load_json(ledger_path, "operational tool failure ledger")
    if payload.get("schema") != OPERATIONAL_FAILURE_SCHEMA or payload.get("status") != (
        "locked_operational_tool_failure_excluded"
    ):
        raise RuntimeError("operational tool failure ledger schema/status drift")
    builder = Path(__file__).resolve().parent / "build_operational_tool_failure_ledger.py"
    if pair(payload.get("generator", {})) != pair(identity(builder)):
        raise RuntimeError("operational tool failure ledger generator identity drift")
    claims = payload.get("claims", {})
    for name in (
        "listed_run_is_not_formal_acceptance_evidence",
        "successful_inner_model_smoke_does_not_replace_missing_outer_attestation",
        "formal_receipts_are_not_overwritten",
        "fresh_formal_launcher_rerun_required",
    ):
        exact_bool(claims.get(name), True, f"operational failure {name}")
    exact_bool(
        claims.get("listed_run_satisfies_any_deployment_gate"),
        False,
        "operational failure gate boundary",
    )
    entry = payload.get("entry", {})
    if (
        entry.get("entry_id") != "k100_3_launcher_host_output_permission_failure_v1"
        or entry.get("category") != "operational_tool_failure"
        or entry.get("failure_mode")
        != "host_output_permission_denied_after_successful_container_smoke"
        or entry.get("failure_stage") != "post_container_smoke_host_attestation_write"
        or entry.get("node_label") != "K100-3"
        or entry.get("inner_model_smoke_status") != "passed_observation_only"
    ):
        raise RuntimeError("operational tool failure entry semantics drift")
    label = str(entry.get("candidate_label", ""))
    if label not in candidate_manifests or entry.get("manifest_sha256") != candidate_manifests[
        label
    ]:
        raise RuntimeError("operational tool failure candidate/manifest lineage drift")
    if entry.get("official_image_id") not in expected_image_ids:
        raise RuntimeError("operational tool failure image lineage drift")
    exact_bool(
        entry.get("outer_launch_attestation_written"),
        False,
        "operational failure outer attestation",
    )
    exact_bool(
        entry.get("counts_toward_acceptance"),
        False,
        "operational failure acceptance count",
    )
    artifact_row = entry.get("artifact", {})
    if artifact_row.get("kind") != "directory_tree":
        raise RuntimeError("operational tool failure artifact must be a directory tree")
    artifact = resolve_node_below(
        root, str(artifact_row.get("path", "")), "operational tool failure artifact"
    )
    if not artifact.is_dir() or contained(artifact, ledger_path):
        raise RuntimeError("operational tool failure artifact/ledger overlap")
    observed = directory_tree_identity(artifact)
    for field in ("file_count", "total_bytes", "tree_sha256"):
        if artifact_row.get(field) != observed[field]:
            raise RuntimeError("operational tool failure directory identity drift")
    return {
        "status": "passed",
        "ledger": identity(ledger_path),
        "root": str(root),
        "candidate_label": label,
        "failed_launcher_tree": {
            "path": str(artifact),
            "file_count": observed["file_count"],
            "total_bytes": observed["total_bytes"],
            "tree_sha256": observed["tree_sha256"],
        },
        "counts_toward_acceptance": False,
    }


def telemetry_summary(rows: list[dict]) -> dict:
    required = ("temperature_c", "power_w", "vram_used_bytes")
    if not rows:
        raise RuntimeError("stability telemetry is empty")
    timestamps = []
    coverage_counts = {name: 0 for name in required}
    flattened = {name: [] for name in required}
    last_counters = {"inferences": -1, "errors": 0, "nonfinite_outputs": 0, "fixed_prediction_drifts": 0}
    for index, row in enumerate(rows, start=1):
        timestamp = row.get("monotonic_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise RuntimeError(f"telemetry sample {index} monotonic timestamp drift")
        timestamps.append(timestamp)
        for name in required:
            values = row.get("metrics", {}).get(name, {}).get("values")
            if not isinstance(values, list) or not values:
                raise RuntimeError(f"telemetry sample {index} lacks {name}")
            checked = [finite_number(value, f"telemetry {name}", 0.0) for value in values]
            coverage_counts[name] += 1
            flattened[name].extend(checked)
        counters = row.get("runtime_counters", {})
        current = {}
        for key in last_counters:
            value = counters.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"telemetry sample {index} counter {key} drift")
            current[key] = value
        if current["inferences"] < last_counters["inferences"]:
            raise RuntimeError("telemetry inference counter decreased")
        if any(current[key] != 0 for key in ("errors", "nonfinite_outputs", "fixed_prediction_drifts")):
            raise RuntimeError("telemetry contains a nonzero error/nonfinite/drift counter")
        last_counters = current
    intervals = [
        (right - left) / 1.0e9 for left, right in zip(timestamps, timestamps[1:])
    ]
    return {
        "samples": len(rows),
        "required_metrics": list(required),
        "coverage_sample_counts": coverage_counts,
        "all_three_metrics_present_in_every_sample": True,
        "interval_count": len(intervals),
        "minimum_interval_seconds": min(intervals) if intervals else None,
        "maximum_interval_seconds": max(intervals) if intervals else None,
        "intervals_strictly_positive": bool(intervals) and all(value > 0 for value in intervals),
        "minimum_interval_ge_4_seconds": bool(intervals) and min(intervals) >= 4.0,
        "maximum_interval_le_7_5_seconds": bool(intervals) and max(intervals) <= 7.5,
        "values": {
            name: {
                "value_count": len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in flattened.items()
        },
    }


def validate_stability(payload: dict, context: dict, telemetry_path: Path) -> dict:
    common_acceptance_receipt(
        payload,
        "phase11_k100_stability_60min_v1",
        context,
        "stability-60min",
        "continuous_60min_passed",
    )
    protocol = payload.get("protocol", {})
    requested = finite_number(
        protocol.get("requested_duration_seconds"), "stability requested duration", 3600.0
    )
    observed = finite_number(
        protocol.get("observed_duration_seconds"), "stability observed duration", 3600.0
    )
    if protocol.get("telemetry_interval_seconds") != 5 or protocol.get(
        "telemetry_allowed_interval_seconds"
    ) != [4.0, 7.5] or protocol.get("max_inferences") is not None:
        raise RuntimeError("stability formal duration/telemetry protocol drift")
    gates = payload.get("gates", {})
    if set(gates) != EXPECTED_STABILITY_GATES or not all(
        value is True for value in gates.values()
    ):
        raise RuntimeError("stability gates are incomplete or failed")
    measurements = payload.get("measurements", {})
    inferences = measurements.get("inferences")
    if isinstance(inferences, bool) or not isinstance(inferences, int) or inferences <= 0:
        raise RuntimeError("stability must contain at least one inference")
    for key in ("errors", "nonfinite_outputs", "fixed_prediction_drifts"):
        if measurements.get(key) != 0:
            raise RuntimeError(f"stability {key} is nonzero")
    if measurements.get("error_rate") != 0.0:
        raise RuntimeError("stability error rate is nonzero")
    finite_number(
        measurements.get("inference_seconds_median"), "stability median inference", 0.0
    )
    finite_number(measurements.get("inference_seconds_p95"), "stability p95 inference", 0.0)
    lines = telemetry_path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise RuntimeError("stability telemetry JSONL has empty records")
    try:
        rows = [json.loads(line) for line in lines]
    except Exception as exc:
        raise RuntimeError("stability telemetry JSONL parse failure") from exc
    recomputed = telemetry_summary(rows)
    if recomputed != measurements.get("telemetry"):
        raise RuntimeError("stability telemetry summary differs from locked raw JSONL")
    if recomputed["samples"] < int(requested // 5.0):
        raise RuntimeError("stability telemetry sample count is insufficient")
    return {
        "passed": True,
        "requested_duration_seconds": requested,
        "observed_duration_seconds": observed,
        "inferences": inferences,
        "telemetry_samples": recomputed["samples"],
        "telemetry": identity(telemetry_path),
    }


def validate_candidate(label: str, bundle: Path, manifest_sha: str, acceptance_root: Path) -> dict:
    context = inventory_bundle(bundle, manifest_sha, label)
    root = root_directory(acceptance_root, f"{label} acceptance root")
    receipts = {
        name: resolve_below(root, relative, f"{label} {name} receipt")
        for name, relative in RECEIPT_LAYOUT.items()
    }
    if len({path for path in receipts.values()}) != len(RECEIPT_LAYOUT):
        raise RuntimeError(f"{label} acceptance receipt paths are not unique")
    payloads = {
        name: load_json(path, f"{label} {name} receipt") for name, path in receipts.items()
    }
    gates = {
        "static_verify": validate_static(payloads["static_verify"], context),
        "cold_start_5": validate_cold(payloads["cold_start_5"], context),
        "recovery_3": validate_recovery(payloads["recovery_3"], context),
    }
    smoke2_fingerprint = resolve_below(
        root,
        "smoke_k100_2/target_runtime_fingerprint.json",
        f"{label} K100-2 target fingerprint",
    )
    smoke3_fingerprint = resolve_below(
        root,
        "smoke_k100_3/target_runtime_fingerprint.json",
        f"{label} K100-3 target fingerprint",
    )
    gates["smoke_k100_2"] = validate_smoke(
        payloads["smoke_k100_2"], context, "K100-2", smoke2_fingerprint
    )
    gates["smoke_k100_3"] = validate_smoke(
        payloads["smoke_k100_3"], context, "K100-3", smoke3_fingerprint
    )
    launch_path = resolve_below(
        root, K1003_LAUNCH_ATTESTATION, f"{label} K100-3 launch attestation"
    )
    launch_payload = load_json(launch_path, f"{label} K100-3 launch attestation")
    gates["smoke_k100_3"]["pre_entrypoint_launch"] = validate_k1003_launch_attestation(
        launch_payload,
        launch_path,
        receipts["smoke_k100_3"],
        context,
        payloads["smoke_k100_3"],
    )
    if gates["smoke_k100_2"]["hostname"] == gates["smoke_k100_3"]["hostname"]:
        raise RuntimeError(f"{label} K100-2 and K100-3 smoke use the same hostname")
    telemetry = resolve_below(
        root, "stability_60min/telemetry_5s.jsonl", f"{label} stability telemetry"
    )
    gates["stability_60min"] = validate_stability(
        payloads["stability_60min"], context, telemetry
    )
    receipt_identities = {name: identity(path) for name, path in receipts.items()}
    if len({row["sha256"] for row in receipt_identities.values()}) != len(
        receipt_identities
    ):
        raise RuntimeError(f"{label} reuses a receipt byte-for-byte across distinct gates")
    return {
        "label": label,
        "status": "passed",
        "bundle_id": context["bundle_id"],
        "execution_kind": context["kind"],
        "official_image_id": context["image_id"],
        "portable_runtime_signature_sha256": context["manifest"]["runtime_contract"][
            "portable_runtime_signature_sha256"
        ],
        "bundle_root": str(context["root"]),
        "acceptance_root": str(root),
        "manifest": context["manifest_identity"],
        "bundle_payload_file_count": context["payload_file_count"],
        "bundle_payload_total_bytes": context["payload_total_bytes"],
        "receipts": receipt_identities,
        "support_artifacts": {
            "k100_3_pre_entrypoint_launch_attestation": identity(launch_path),
        },
        "gates": gates,
        "all_six_required_gates_passed": True,
        "claim_boundaries": {
            "strict_logits_equivalence_inferred": False,
            "native_int8_kernel_precision_inferred": False,
            "operational_acceptance_does_not_select_the_pareto_winner": True,
        },
    }


def write_outputs(output: Path, result: dict, markdown: bool) -> dict:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing aggregate output: {output}")
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    if partial.exists():
        raise RuntimeError(f"stale partial aggregate must be inspected explicitly: {partial}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        json_path = partial / "final_deployment_acceptance.json"
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        columns = [
            "candidate",
            "status",
            "bundle_id",
            "execution_kind",
            "manifest_sha256",
            "official_image_id",
            "portable_runtime_signature_sha256",
            "k100_3_pre_entrypoint_launch_attestation_sha256",
            *[f"{name}_passed" for name in RECEIPT_LAYOUT],
            *[f"{name}_receipt_sha256" for name in RECEIPT_LAYOUT],
            "operational_deployment_acceptance_passed",
            "known_invalid_environmental_runs_excluded",
            "known_operational_tool_failure_excluded",
        ]
        csv_path = partial / "final_deployment_acceptance.csv"
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for candidate in result.get("candidates", []):
                gates = candidate.get("gates", {})
                receipts = candidate.get("receipts", {})
                writer.writerow(
                    {
                        "candidate": candidate.get("label"),
                        "status": candidate.get("status"),
                        "bundle_id": candidate.get("bundle_id", ""),
                        "execution_kind": candidate.get("execution_kind", ""),
                        "manifest_sha256": candidate.get("manifest", {}).get("sha256", ""),
                        "official_image_id": candidate.get("official_image_id", ""),
                        "portable_runtime_signature_sha256": candidate.get(
                            "portable_runtime_signature_sha256", ""
                        ),
                        "k100_3_pre_entrypoint_launch_attestation_sha256": candidate.get(
                            "support_artifacts", {}
                        ).get("k100_3_pre_entrypoint_launch_attestation", {}).get(
                            "sha256", ""
                        ),
                        **{
                            f"{name}_passed": gates.get(name, {}).get("passed", False)
                            for name in RECEIPT_LAYOUT
                        },
                        **{
                            f"{name}_receipt_sha256": receipts.get(name, {}).get("sha256", "")
                            for name in RECEIPT_LAYOUT
                        },
                        "operational_deployment_acceptance_passed": candidate.get(
                            "all_six_required_gates_passed", False
                        ),
                        "known_invalid_environmental_runs_excluded": result.get(
                            "environmental_run_exclusions", {}
                        ).get("status") == "passed",
                        "known_operational_tool_failure_excluded": result.get(
                            "operational_tool_failure_exclusion", {}
                        ).get("status") == "passed",
                    }
                )
        if markdown:
            md_path = partial / "final_deployment_acceptance.md"
            lines = [
                "# K100 final deployment acceptance",
                "",
                f"Overall status: **{result['status']}**",
                "",
                "| Candidate | Static | Cold 5 | Recovery 3 | K100-2 | K100-3 | 60 min | Overall |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for candidate in result.get("candidates", []):
                gates = candidate.get("gates", {})
                marker = lambda name: "PASS" if gates.get(name, {}).get("passed") else "FAIL"
                lines.append(
                    "| {label} | {static} | {cold} | {recovery} | {smoke2} | {smoke3} | "
                    "{stability} | {overall} |".format(
                        label=candidate.get("label"),
                        static=marker("static_verify"),
                        cold=marker("cold_start_5"),
                        recovery=marker("recovery_3"),
                        smoke2=marker("smoke_k100_2"),
                        smoke3=marker("smoke_k100_3"),
                        stability=marker("stability_60min"),
                        overall="PASS"
                        if candidate.get("all_six_required_gates_passed")
                        else "FAIL",
                    )
                )
            lines.extend(
                [
                    "",
                    "This aggregate proves operational deployment acceptance only. It does not "
                    "rewrite strict-logits failures, prove native INT8 kernels, or select the "
                    "final Pareto recommendation.",
                    "",
                    "The two known K100-3 environmental attempts (PATH injected too late; "
                    "no-hyhal HIP-100 diagnostic) are identity-locked in the exclusion ledger "
                    "and satisfy no acceptance gate. Formal K100-3 smoke must inject the locked "
                    "PATH through `docker run -e` before the entrypoint.",
                    "The separate host-output permission failure is locked as an operational "
                    "tool failure: its inner smoke observation cannot replace the missing outer "
                    "attestation and satisfies no gate.",
                ]
            )
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        partial.rename(output)
    except Exception:
        # Keep a failed partial directory for explicit forensic inspection; no
        # old evidence or previously published aggregate is removed.
        raise
    files = {
        path.name: identity(path)
        for path in sorted(item for item in output.iterdir() if item.is_file())
    }
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5-bundle", type=Path, required=True)
    parser.add_argument("--m5-manifest-sha256", required=True)
    parser.add_argument("--m5-acceptance-root", type=Path, required=True)
    parser.add_argument("--fp16-bundle", type=Path, required=True)
    parser.add_argument("--fp16-manifest-sha256", required=True)
    parser.add_argument("--fp16-acceptance-root", type=Path, required=True)
    parser.add_argument("--environment-exclusion-root", type=Path, required=True)
    parser.add_argument("--environment-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--environment-exclusion-ledger-sha256", required=True)
    parser.add_argument("--operational-failure-root", type=Path, required=True)
    parser.add_argument("--operational-failure-ledger", type=Path, required=True)
    parser.add_argument("--operational-failure-ledger-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-markdown", action="store_true")
    args = parser.parse_args()

    roots = [
        root_directory(args.m5_bundle, "M5 bundle"),
        root_directory(args.m5_acceptance_root, "M5 acceptance root"),
        root_directory(args.fp16_bundle, "FP16 bundle"),
        root_directory(args.fp16_acceptance_root, "FP16 acceptance root"),
        root_directory(args.environment_exclusion_root, "environment exclusion root"),
        root_directory(args.operational_failure_root, "operational tool failure root"),
    ]
    if len(set(roots)) != 6:
        raise RuntimeError(
            "M5/FP16 bundle, acceptance and both exclusion roots must be six distinct directories"
        )
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if contained(left, right) or contained(right, left):
                raise RuntimeError("immutable bundle/acceptance/exclusion roots must not overlap")
    raw_output = args.output_dir.absolute()
    if raw_output.name in {"", ".", ".."}:
        raise RuntimeError("final aggregate output must have a normal leaf directory name")
    output_parent = root_directory(raw_output.parent, "final aggregate output parent")
    output = output_parent / raw_output.name
    if output.exists():
        raise RuntimeError(f"refusing to overwrite final aggregate directory: {output}")
    if any(contained(root, output) for root in roots):
        raise RuntimeError("final aggregate output must be outside every immutable input root")

    candidates = []
    failures = []
    for label, bundle, manifest_sha, acceptance_root in (
        ("M5", args.m5_bundle, args.m5_manifest_sha256, args.m5_acceptance_root),
        ("FP16", args.fp16_bundle, args.fp16_manifest_sha256, args.fp16_acceptance_root),
    ):
        try:
            candidates.append(validate_candidate(label, bundle, manifest_sha, acceptance_root))
        except Exception as exc:
            failure = {"label": label, "type": type(exc).__name__, "message": str(exc)}
            failures.append(failure)
            candidates.append(
                {
                    "label": label,
                    "status": "failed",
                    "all_six_required_gates_passed": False,
                    "failure": failure,
                }
            )
    passed_candidates = [row for row in candidates if row.get("status") == "passed"]
    if len(passed_candidates) == 2 and (
        len({row["official_image_id"] for row in passed_candidates}) != 1
        or len(
            {row["portable_runtime_signature_sha256"] for row in passed_candidates}
        )
        != 1
    ):
        failure = {
            "label": "cross_candidate_runtime",
            "type": "RuntimeError",
            "message": "M5 and FP16 do not share the same locked image/runtime signature",
        }
        failures.append(failure)
    exclusions = None
    try:
        exclusions = validate_environment_exclusion_ledger(
            args.environment_exclusion_root,
            args.environment_exclusion_ledger,
            args.environment_exclusion_ledger_sha256,
            {args.m5_manifest_sha256, args.fp16_manifest_sha256},
            {
                candidate["official_image_id"]
                for candidate in candidates
                if candidate.get("status") == "passed"
            },
        )
    except Exception as exc:
        failure = {
            "label": "environment_exclusions",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        failures.append(failure)
        exclusions = {"status": "failed", "failure": failure}
    operational_failure = None
    try:
        operational_failure = validate_operational_tool_failure_ledger(
            args.operational_failure_root,
            args.operational_failure_ledger,
            args.operational_failure_ledger_sha256,
            {"M5": args.m5_manifest_sha256, "FP16": args.fp16_manifest_sha256},
            {
                candidate["official_image_id"]
                for candidate in candidates
                if candidate.get("status") == "passed"
            },
        )
    except Exception as exc:
        failure = {
            "label": "operational_tool_failure_exclusion",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        failures.append(failure)
        operational_failure = {"status": "failed", "failure": failure}
    passed = len(failures) == 0 and len(candidates) == 2 and all(
        candidate.get("all_six_required_gates_passed") is True for candidate in candidates
    ) and exclusions.get("status") == "passed" and operational_failure.get(
        "status"
    ) == "passed"
    result = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "aggregated_at_utc": utc_now(),
        "generator": identity(Path(__file__)),
        "required_candidates": ["M5", "FP16"],
        "required_gate_layout": RECEIPT_LAYOUT,
        "candidates": candidates,
        "environmental_run_exclusions": exclusions,
        "operational_tool_failure_exclusion": operational_failure,
        "failures": failures,
        "claims": {
            "both_candidates_operational_deployment_acceptance_passed": passed,
            "all_bundle_payloads_rehashed": passed,
            "all_six_receipts_per_candidate_identity_locked_and_semantically_passed": passed,
            "strict_logits_equivalence_inferred": False,
            "native_int8_kernel_precision_inferred": False,
            "pareto_recommendation_selected": False,
            "known_invalid_environmental_runs_excluded_from_all_gates": passed,
            "docker_path_injected_before_entrypoint_for_k100_3": passed,
            "known_launcher_permission_failure_excluded_from_all_gates": passed,
        },
    }
    output_files = write_outputs(output, result, args.write_markdown)
    receipt = {
        "status": result["status"],
        "output_dir": str(output.resolve(strict=True)),
        "files": output_files,
        "exit_code": 0 if passed else 2,
    }
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
