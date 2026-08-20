#!/usr/bin/env python3
"""Fail-closed validation of the cross-node segment-25 cache smoke.

This validator deliberately does not infer portability from a successful ORT run
alone.  It locks the immutable bundle manifest, re-hashes all 25 origin caches,
checks the target runtime boundary, parses the raw strace files, and independently
reads every ORT profile before granting the cache-portability claim.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "phase11_segment25_crossnode_cache_portability_validation_v1"
MANIFEST_IDENTITY = (
    126_255,
    "0abd5a99c3fa9ebcd4236bf6f57090f9fce102777535402313e081777eb9c6cc",
)
RUNTIME_COMPARISON_IDENTITY = (
    4_463,
    "0feb3e326db7b004667d0eb7321bd34e3d0b04782c8c25d8f0dda9b3c937a8a6",
)
ORIGIN_IMAGE = "sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291"
TARGET_IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
SELECTED_RUNTIME_CONTENT_SHA256 = (
    "fe7324427d3d54499b5b4542cdbdf0af06e45f8290d7bab633f83799da9cf272"
)
CONTAINER_BUNDLE_ROOT = PurePosixPath("/work/root")
CONTAINER_RUN_ROOT = PurePosixPath("/work/run")
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
EXPECTED_LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24)) + (
    "upernet_decoder_head",
)
FROZEN_REFERENCE_FILES = (
    "segment25_iobinding_end_to_end_single/output/result.json",
    "segment25_iobinding_end_to_end_single/output/cpu_host_staged_and_device_logits.npz",
)
REQUIRED_PROBE_GATES = (
    "all_25_sessions_concurrently_loaded",
    "all_25_profiles_migraphx_positive_cpu_zero",
    "final_output_device_cuda_alias",
    "resident_vs_sequential_device_mae_le_1e_6",
    "resident_vs_sequential_device_max_abs_le_1e_5",
    "resident_vs_sequential_device_predictions_exact",
)
GATE_KEYS = (
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
FORBIDDEN_OPEN_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND")
OPEN_RE = re.compile(
    r"openat\([^,]+,\s*(?P<quoted>\"(?:\\.|[^\"\\])*\"),\s*"
    r"(?P<flags>[^)]*)\)\s*=\s*(?P<fd>-?\d+)(?:<(?P<resolved>[^>]*)>)?"
)
READ_RE = re.compile(
    r"read\(\s*(?P<fd>\d+)(?:<(?P<annotated>[^>]*)>)?,.*\)\s*=\s*(?P<ret>-?\d+)"
)
MMAP_RE = re.compile(
    r"mmap\((?P<a0>[^,]*),(?P<a1>[^,]*),(?P<prot>[^,]*),(?P<mapflags>[^,]*),"
    r"\s*(?P<fd>-?\d+)(?:<(?P<annotated>[^>]*)>)?,(?P<offset>[^)]*)\)\s*=\s*(?P<ret>\S+)"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(RuntimeError):
    """A fail-closed contract violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValidationError(f"regular non-symlink file required: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValidationError(f"regular non-symlink file required: {path}")
    return {"size_bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def safe_bundle_relative(value: Any) -> str:
    require(isinstance(value, str) and bool(value), f"invalid relative path: {value!r}")
    require("\\" not in value, f"bundle path must use POSIX separators: {value!r}")
    parsed = PurePosixPath(value)
    require(not parsed.is_absolute(), f"absolute bundle path forbidden: {value!r}")
    require(".." not in parsed.parts and "." not in parsed.parts, f"non-canonical bundle path: {value!r}")
    require(str(parsed) == value, f"non-canonical bundle path: {value!r}")
    return value


def container_path(relative: str) -> str:
    return str(CONTAINER_BUNDLE_ROOT / PurePosixPath(relative))


def decode_strace_string(quoted: str) -> str:
    try:
        value = ast.literal_eval(quoted)
    except (SyntaxError, ValueError) as exc:
        raise ValidationError(f"invalid quoted path in strace: {quoted!r}") from exc
    require(isinstance(value, str), f"strace path is not text: {quoted!r}")
    return value


def normalized_trace_path(value: str | None) -> str | None:
    if not value or not value.startswith("/"):
        return value
    suffix = " (deleted)"
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return str(PurePosixPath(value))


def validate_manifest(bundle_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = bundle_root / "BUNDLE_MANIFEST.json"
    manifest_identity = identity(manifest_path)
    require(
        (manifest_identity["size_bytes"], manifest_identity["sha256"]) == MANIFEST_IDENTITY,
        f"bundle manifest identity drift: {manifest_identity}",
    )
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "assembled_and_identity_locked", "manifest status drift")
    require(manifest.get("bundle_format") == "phase11_int8_backbone_segment25_v1", "bundle format drift")
    require(manifest.get("official_image_id") == ORIGIN_IMAGE, "origin image identity drift")
    segments = manifest.get("segments")
    require(isinstance(segments, list) and len(segments) == 25, "manifest must contain 25 segments")
    file_rows = manifest.get("files")
    require(isinstance(file_rows, list), "manifest files table missing")
    files_by_path: dict[str, dict[str, Any]] = {}
    for row in file_rows:
        relative = safe_bundle_relative(row.get("path"))
        require(relative not in files_by_path, f"duplicate manifest file path: {relative}")
        files_by_path[relative] = row

    caches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(segments):
        require(row.get("index") == index, f"segment index/order drift at {index}")
        require(row.get("label") == EXPECTED_LABELS[index], f"segment label drift at {index}")
        relative = safe_bundle_relative(row.get("cache"))
        require(relative.endswith(".mxr"), f"segment cache extension drift: {relative}")
        require(relative not in seen, f"duplicate cache path: {relative}")
        seen.add(relative)
        cache_identity = row.get("cache_identity")
        require(isinstance(cache_identity, dict), f"cache identity missing at segment {index}")
        size_bytes = cache_identity.get("size_bytes")
        digest = cache_identity.get("sha256")
        require(isinstance(size_bytes, int) and size_bytes > 0, f"invalid cache size at segment {index}")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"invalid cache SHA at segment {index}")
        file_row = files_by_path.get(relative)
        require(file_row is not None, f"cache absent from manifest files table: {relative}")
        require(
            (file_row.get("size_bytes"), file_row.get("sha256")) == (size_bytes, digest),
            f"segment/files cache identity disagreement: {relative}",
        )
        caches.append(
            {
                "index": index,
                "label": EXPECTED_LABELS[index],
                "relative_path": relative,
                "container_path": container_path(relative),
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    return manifest_identity, caches, files_by_path


def validate_cache_integrity(bundle_root: Path, caches: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {row["relative_path"]: row for row in caches}
    actual: dict[str, Path] = {}
    symlinks: list[str] = []
    for path in sorted(bundle_root.rglob("*.mxr")):
        relative = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file():
            actual[relative] = path
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    identities: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    checked_bytes = 0
    for relative in sorted(set(expected) & set(actual)):
        observed = identity(actual[relative])
        checked_bytes += int(observed["size_bytes"])
        wanted = expected[relative]
        passed = (observed["size_bytes"], observed["sha256"]) == (
            wanted["size_bytes"],
            wanted["sha256"],
        )
        item = {"relative_path": relative, **observed, "passed": passed}
        identities.append(item)
        if not passed:
            drift.append(item)
    return {
        "expected_cache_count": len(expected),
        "actual_mxr_file_count": len(actual),
        "checked_cache_bytes": checked_bytes,
        "missing": missing,
        "unexpected": unexpected,
        "symlinks": symlinks,
        "identity_drift": drift,
        "cache_identities": identities,
        "path_set_exact": not missing and not unexpected and len(actual) == 25,
        "all_identities_exact": not drift and len(identities) == 25,
    }


def validate_runtime_boundary(bundle_manifest: dict[str, Any], run_root: Path) -> dict[str, Any]:
    comparison_path = run_root / "phase11_runtime_fingerprint_comparison_machine2.json"
    comparison_identity = identity(comparison_path)
    require(
        (comparison_identity["size_bytes"], comparison_identity["sha256"])
        == RUNTIME_COMPARISON_IDENTITY,
        f"runtime comparison identity drift: {comparison_identity}",
    )
    comparison = read_json(comparison_path)
    images = comparison.get("images", {})
    claims = comparison.get("claims", {})
    require(comparison.get("status") == "critical_runtime_content_match", "runtime match status failed")
    require(images.get("origin") == ORIGIN_IMAGE and images.get("target") == TARGET_IMAGE, "runtime image IDs drift")
    require(images.get("exact_image_identity_match") is False, "exact-image boundary must remain false")
    require(images.get("exact_layer_manifest_match") is False, "exact-layer boundary must remain false")
    require(comparison.get("selected_runtime_content_match") is True, "selected runtime content mismatch")
    require(
        comparison.get("selected_runtime_content_sha256") == SELECTED_RUNTIME_CONTENT_SHA256,
        "selected runtime content digest drift",
    )
    require(claims.get("critical_ort_migraphx_runtime_content_match") is True, "runtime claim failed")
    require(claims.get("exact_image_identity_match") is False, "comparison overclaims exact image identity")

    inspect = read_json(run_root / "image_inspect.json")
    image_row = inspect[0] if isinstance(inspect, list) and len(inspect) == 1 else inspect
    require(isinstance(image_row, dict), "image inspect must contain one object")
    require(image_row.get("Id") == TARGET_IMAGE, "target smoke image identity drift")
    require(bundle_manifest.get("official_image_id") == ORIGIN_IMAGE, "bundle origin image drift")
    return {
        "origin_image_id": ORIGIN_IMAGE,
        "target_image_id": TARGET_IMAGE,
        "exact_image_identity_match": False,
        "exact_layer_manifest_match": False,
        "critical_runtime_content_match": True,
        "selected_runtime_content_sha256": SELECTED_RUNTIME_CONTENT_SHA256,
        "comparison_identity": comparison_identity,
        "claim_boundary": (
            "The target is a matched-runtime rebuild with different image ID and layer manifest; "
            "this is not exact-image identity."
        ),
    }


def manifest_file_identity(
    bundle_root: Path, files_by_path: dict[str, dict[str, Any]], relative: str
) -> dict[str, Any]:
    expected = files_by_path.get(relative)
    require(expected is not None, f"required frozen reference absent from manifest: {relative}")
    observed = identity(bundle_root / relative)
    require(
        (observed["size_bytes"], observed["sha256"])
        == (expected.get("size_bytes"), expected.get("sha256")),
        f"frozen reference identity drift: {relative}",
    )
    return {"relative_path": relative, **observed}


def provider_counts_from_profile(path: Path) -> Counter[str]:
    events = read_json(path)
    require(isinstance(events, list), f"ORT profile must be an event list: {path}")
    return Counter(
        str(event["args"]["provider"])
        for event in events
        if isinstance(event, dict)
        and event.get("cat") == "Node"
        and isinstance(event.get("args"), dict)
        and event["args"].get("provider")
    )


def validate_probe(
    bundle_root: Path, run_root: Path, files_by_path: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    exit_text = (run_root / "probe.exit").read_text(encoding="utf-8").strip()
    require(exit_text == "0", f"probe exit is not zero: {exit_text!r}")
    result_path = run_root / "output" / "result.json"
    result_identity = identity(result_path)
    result = read_json(result_path)
    require(result.get("status") == "passed", f"probe status failed: {result.get('status')!r}")
    require(
        result.get("variant") == "int8_backbone_25_cached_sessions_concurrently_resident_single",
        "probe variant drift",
    )
    claims = result.get("claims", {})
    require(claims.get("all_25_sessions_concurrently_resident") is True, "resident claim failed")
    require(claims.get("device_resident_single_request_pipeline") is True, "device pipeline claim failed")
    for boundary in (
        "task_utility_test90",
        "strict_numeric_admission",
        "performance",
        "native_int8_kernel_verified",
        "deployment_ready",
    ):
        require(claims.get(boundary) is False, f"smoke overclaims {boundary}")

    gates = result.get("gates")
    require(isinstance(gates, dict), "probe gates missing")
    require(set(REQUIRED_PROBE_GATES).issubset(gates), "required probe gates missing")
    require(gates and all(value is True for value in gates.values()), "not all probe gates are true")
    runtime = result.get("runtime", {})
    require(runtime.get("onnxruntime") == "1.19.2", "probe ORT version drift")
    require(runtime.get("resident_session_count") == 25, "resident session count is not 25")
    require(runtime.get("intersegment_transport") == "direct_OrtValue_device_binding", "OrtValue transport drift")

    load_rows = result.get("session_loads")
    inference_rows = result.get("segment_inference")
    require(isinstance(load_rows, list) and len(load_rows) == 25, "session load rows must contain 25 entries")
    require(isinstance(inference_rows, list) and len(inference_rows) == 25, "segment inference rows must contain 25 entries")
    require([row.get("index") for row in load_rows] == list(range(25)), "session load order drift")
    require([row.get("label") for row in load_rows] == list(EXPECTED_LABELS), "session labels drift")
    require([row.get("index") for row in inference_rows] == list(range(25)), "inference order drift")

    frozen_references = [
        manifest_file_identity(bundle_root, files_by_path, relative)
        for relative in FROZEN_REFERENCE_FILES
    ]
    probe_identities = result.get("identities", {})
    reference_by_relative = {row["relative_path"]: row for row in frozen_references}
    for key, relative in (
        ("passed_sequential_session_device_single", FROZEN_REFERENCE_FILES[0]),
        ("sequential_session_device_logits", FROZEN_REFERENCE_FILES[1]),
    ):
        described = probe_identities.get(key, {})
        frozen = reference_by_relative[relative]
        require(
            (described.get("size_bytes"), described.get("sha256"))
            == (frozen["size_bytes"], frozen["sha256"]),
            f"probe frozen reference description drift: {key}",
        )

    profiles = result.get("profiles")
    require(isinstance(profiles, list) and len(profiles) == 25, "probe must contain 25 profiles")
    require([row.get("index") for row in profiles] == list(range(25)), "profile index/order drift")
    profile_rows: list[dict[str, Any]] = []
    seen_profile_paths: set[str] = set()
    for index, row in enumerate(profiles):
        described_path = row.get("path")
        require(isinstance(described_path, str), f"profile path missing at segment {index}")
        parsed_path = PurePosixPath(described_path)
        require(parsed_path.parent == CONTAINER_RUN_ROOT / "output", f"profile path escapes run output: {described_path}")
        require(described_path not in seen_profile_paths, f"duplicate profile path: {described_path}")
        seen_profile_paths.add(described_path)
        host_path = run_root / "output" / parsed_path.name
        observed = identity(host_path)
        require(
            (observed["size_bytes"], observed["sha256"])
            == (row.get("size_bytes"), row.get("sha256")),
            f"profile identity drift at segment {index}",
        )
        counts = provider_counts_from_profile(host_path)
        described_counts = row.get("provider_event_counts")
        require(isinstance(described_counts, dict), f"profile provider counts missing at segment {index}")
        require(dict(counts) == described_counts, f"raw/profile provider count disagreement at segment {index}")
        require(counts[MGX] > 0 and counts[CPU] == 0, f"provider placement failed at segment {index}")
        require(set(counts) == {MGX}, f"unexpected execution provider at segment {index}: {dict(counts)}")
        require(row.get("passed") is True, f"profile placement flag failed at segment {index}")
        profile_rows.append(
            {
                "index": index,
                "file": parsed_path.name,
                **observed,
                "provider_event_counts": dict(counts),
            }
        )

    comparison = result.get("comparison_vs_sequential_session_device_pipeline", {})
    mae = comparison.get("mae")
    max_abs = comparison.get("max_abs")
    agreement = comparison.get("pixel_class_agreement")
    changed = comparison.get("changed_pixels")
    require(isinstance(mae, (int, float)) and math.isfinite(mae), "non-finite/missing frozen-logit MAE")
    require(isinstance(max_abs, (int, float)) and math.isfinite(max_abs), "non-finite/missing frozen-logit max_abs")
    require(isinstance(agreement, (int, float)) and math.isfinite(agreement), "non-finite/missing class agreement")
    require(mae <= 1e-6, f"frozen node3 logits MAE gate failed: {mae}")
    require(max_abs <= 1e-5, f"frozen node3 logits max_abs gate failed: {max_abs}")
    require(agreement == 1.0 and changed == 0, "frozen node3 predictions are not exact")
    return {
        "exit_code": 0,
        "result_identity": result_identity,
        "status": result["status"],
        "resident_session_count": runtime["resident_session_count"],
        "intersegment_transport": runtime["intersegment_transport"],
        "probe_gates": gates,
        "relevant_claims": {
            "all_25_sessions_concurrently_resident": True,
            "device_resident_single_request_pipeline": True,
        },
        "frozen_reference_identities": frozen_references,
        "comparison_vs_frozen_node3_device_logits": {
            "mae": mae,
            "max_abs": max_abs,
            "pixel_class_agreement": agreement,
            "changed_pixels": changed,
        },
        "profile_count": len(profile_rows),
        "profiles": profile_rows,
    }


def _event_path(annotation: str | None, fd_map: dict[int, str], fd: int) -> str | None:
    annotated = normalized_trace_path(annotation)
    if annotated and annotated.startswith("/"):
        return annotated
    return fd_map.get(fd)


def parse_cache_traces(run_root: Path, caches: list[dict[str, Any]]) -> dict[str, Any]:
    trace_files = sorted(
        (path for path in run_root.glob("cache_trace*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.name,
    )
    require(bool(trace_files), "no cache_trace files found")
    expected = {row["container_path"]: row for row in caches}
    evidence: dict[str, dict[str, Any]] = {
        path: {"readonly_opens": [], "consumption": []} for path in expected
    }
    unexpected_mxr_accesses: list[dict[str, Any]] = []
    forbidden_mxr_opens: list[dict[str, Any]] = []
    total_lines = 0
    mxr_open_events = 0
    mxr_consumption_events = 0

    for trace_path in trace_files:
        fd_map: dict[int, str] = {}
        with trace_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                total_lines += 1
                opened = OPEN_RE.search(line)
                if opened:
                    path = normalized_trace_path(decode_strace_string(opened.group("quoted")))
                    resolved = normalized_trace_path(opened.group("resolved"))
                    fd = int(opened.group("fd"))
                    if fd >= 0:
                        fd_map[fd] = resolved if resolved and resolved.startswith("/") else path  # type: ignore[assignment]
                    if (path and path.endswith(".mxr")) or (resolved and resolved.endswith(".mxr")):
                        mxr_open_events += 1
                        selected = resolved if resolved in expected else path
                        flags = opened.group("flags").split(",", 1)[0].strip()
                        item = {
                            "trace_file": trace_path.name,
                            "line": line_number,
                            "path": selected,
                            "flags": flags,
                            "fd": fd,
                        }
                        if any(flag in flags for flag in FORBIDDEN_OPEN_FLAGS):
                            forbidden_mxr_opens.append(item)
                        if fd >= 0 and selected in expected and "O_RDONLY" in flags and not any(
                            flag in flags for flag in FORBIDDEN_OPEN_FLAGS
                        ):
                            evidence[selected]["readonly_opens"].append(item)
                        elif fd >= 0 and selected not in expected:
                            unexpected_mxr_accesses.append(item)
                    continue

                read = READ_RE.search(line)
                if read and int(read.group("ret")) > 0:
                    fd = int(read.group("fd"))
                    path = _event_path(read.group("annotated"), fd_map, fd)
                    if path in expected:
                        mxr_consumption_events += 1
                        evidence[path]["consumption"].append(
                            {
                                "syscall": "read",
                                "trace_file": trace_path.name,
                                "line": line_number,
                                "fd": fd,
                                "bytes_read": int(read.group("ret")),
                            }
                        )
                    continue

                mapped = MMAP_RE.search(line)
                if mapped and int(mapped.group("fd")) >= 0 and mapped.group("ret") != "-1":
                    fd = int(mapped.group("fd"))
                    path = _event_path(mapped.group("annotated"), fd_map, fd)
                    if path in expected and "PROT_READ" in mapped.group("prot"):
                        mxr_consumption_events += 1
                        evidence[path]["consumption"].append(
                            {
                                "syscall": "mmap",
                                "trace_file": trace_path.name,
                                "line": line_number,
                                "fd": fd,
                            }
                        )

    cache_rows: list[dict[str, Any]] = []
    for cache in caches:
        path = cache["container_path"]
        item = evidence[path]
        cache_rows.append(
            {
                "index": cache["index"],
                "relative_path": cache["relative_path"],
                "container_path": path,
                "readonly_open_count": len(item["readonly_opens"]),
                "consumption_event_count": len(item["consumption"]),
                "first_readonly_open": item["readonly_opens"][0] if item["readonly_opens"] else None,
                "first_consumption": item["consumption"][0] if item["consumption"] else None,
            }
        )
    return {
        "trace_file_count": len(trace_files),
        "trace_total_bytes": sum(path.stat().st_size for path in trace_files),
        "trace_total_lines": total_lines,
        "mxr_open_event_count": mxr_open_events,
        "mxr_consumption_event_count": mxr_consumption_events,
        "unexpected_successful_mxr_accesses": unexpected_mxr_accesses,
        "forbidden_mxr_open_attempts": forbidden_mxr_opens,
        "caches_with_readonly_open": sum(row["readonly_open_count"] > 0 for row in cache_rows),
        "caches_with_read_or_mmap": sum(row["consumption_event_count"] > 0 for row in cache_rows),
        "cache_evidence": cache_rows,
    }


def validate(bundle_root: Path, run_root: Path) -> dict[str, Any]:
    gates = {key: False for key in GATE_KEYS}
    errors: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "bundle_root": str(bundle_root),
        "run_root": str(run_root),
    }

    manifest: dict[str, Any] | None = None
    caches: list[dict[str, Any]] = []
    files_by_path: dict[str, dict[str, Any]] = {}
    try:
        manifest_identity, caches, files_by_path = validate_manifest(bundle_root)
        manifest = read_json(bundle_root / "BUNDLE_MANIFEST.json")
        summary["manifest"] = {"identity": manifest_identity, "segment_count": len(caches)}
        gates["manifest_identity_locked"] = True
        gates["manifest_segment_contract_exact"] = True
    except Exception as exc:
        errors.append({"section": "manifest", "message": str(exc)})

    if caches:
        try:
            cache_integrity = validate_cache_integrity(bundle_root, caches)
            summary["cache_integrity"] = cache_integrity
            gates["cache_path_set_exact_25"] = cache_integrity["path_set_exact"]
            gates["cache_identities_unchanged_after_run"] = cache_integrity["all_identities_exact"]
            gates["no_cache_symlinks"] = not cache_integrity["symlinks"]
            gates["no_unexpected_mxr_files"] = not cache_integrity["unexpected"]
            require(
                all(
                    gates[key]
                    for key in (
                        "cache_path_set_exact_25",
                        "cache_identities_unchanged_after_run",
                        "no_cache_symlinks",
                        "no_unexpected_mxr_files",
                    )
                ),
                "cache integrity/path-set validation failed",
            )
        except Exception as exc:
            errors.append({"section": "cache_integrity", "message": str(exc)})

    if manifest is not None:
        try:
            runtime_boundary = validate_runtime_boundary(manifest, run_root)
            summary["runtime_boundary"] = runtime_boundary
            gates["critical_runtime_content_match"] = runtime_boundary["critical_runtime_content_match"]
            gates["exact_image_identity_is_false"] = runtime_boundary["exact_image_identity_match"] is False
            gates["target_image_identity_locked"] = runtime_boundary["target_image_id"] == TARGET_IMAGE
        except Exception as exc:
            errors.append({"section": "runtime_boundary", "message": str(exc)})

    try:
        host_pre = (run_root / "host_pre.txt").read_text(encoding="utf-8", errors="strict")
        gates["bundle_read_only_mount_declared"] = "bundle_mount=read_only" in host_pre
        require(gates["bundle_read_only_mount_declared"], "read-only bundle mount declaration missing")
    except Exception as exc:
        errors.append({"section": "read_only_mount", "message": str(exc)})

    if files_by_path:
        try:
            probe = validate_probe(bundle_root, run_root, files_by_path)
            summary["probe"] = probe
            gates["probe_exit_zero"] = probe["exit_code"] == 0
            gates["probe_result_status_passed"] = probe["status"] == "passed"
            gates["all_probe_gates_true"] = all(value is True for value in probe["probe_gates"].values())
            gates["all_25_sessions_concurrently_resident"] = (
                probe["resident_session_count"] == 25
                and probe["relevant_claims"]["all_25_sessions_concurrently_resident"] is True
            )
            gates["device_ortvalue_pipeline"] = (
                probe["intersegment_transport"] == "direct_OrtValue_device_binding"
                and probe["relevant_claims"]["device_resident_single_request_pipeline"] is True
            )
            gates["all_25_profiles_migraphx_positive_cpu_zero"] = probe["profile_count"] == 25
            comparison = probe["comparison_vs_frozen_node3_device_logits"]
            gates["node3_frozen_logits_mae_le_1e_6"] = comparison["mae"] <= 1e-6
            gates["node3_frozen_logits_max_abs_le_1e_5"] = comparison["max_abs"] <= 1e-5
            gates["node3_frozen_predictions_exact"] = (
                comparison["pixel_class_agreement"] == 1.0 and comparison["changed_pixels"] == 0
            )
        except Exception as exc:
            errors.append({"section": "probe", "message": str(exc)})

    if caches:
        try:
            trace = parse_cache_traces(run_root, caches)
            summary["strace"] = trace
            gates["all_25_caches_opened_read_only"] = trace["caches_with_readonly_open"] == 25
            gates["all_25_caches_consumed_by_read_or_mmap"] = trace["caches_with_read_or_mmap"] == 25
            gates["no_mxr_write_create_truncate_access"] = not trace["forbidden_mxr_open_attempts"]
            require(not trace["unexpected_successful_mxr_accesses"], "unexpected successful .mxr access in trace")
            require(gates["all_25_caches_opened_read_only"], "not all 25 caches have successful O_RDONLY openat evidence")
            require(
                gates["all_25_caches_consumed_by_read_or_mmap"],
                "not all 25 caches have subsequent read/mmap evidence",
            )
            require(gates["no_mxr_write_create_truncate_access"], "write/create/truncate .mxr open observed")
        except Exception as exc:
            errors.append({"section": "strace", "message": str(exc)})

    passed = not errors and all(gates.values())
    claims = {
        "origin_caches_portable_on_machine2": passed,
        "origin_cache_portable_on_target_machine2": passed,
        "cache_portability_verified": passed,
        "all_25_origin_caches_loaded_on_target": passed,
        "critical_runtime_content_match": gates["critical_runtime_content_match"],
        "exact_image_identity_match": False,
        "exact_layer_manifest_match": False,
        "task_utility_test90_verified_on_target": False,
        "cross_node_performance_verified": False,
        "native_int8_kernel_verified": False,
        "deployment_ready": False,
    }
    summary.update(
        {
            "status": "passed" if passed else "failed",
            "gates": gates,
            "claims": claims,
            "errors": errors,
            "evidence_boundary": (
                "A pass verifies origin-cache loading and one all-resident device-OrtValue smoke on "
                "machine2 under matched critical runtime content. It does not establish exact-image "
                "identity, target test90 task utility, target performance, native INT8 kernels, or "
                "deployment readiness."
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    output = (
        args.output.resolve(strict=False)
        if args.output is not None
        else run_root / "cache_portability_validation.json"
    )
    if output == bundle_root or bundle_root in output.parents:
        raise ValidationError("validation output must be outside the immutable bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = validate(bundle_root, run_root)
    serialized = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
