#!/usr/bin/env python3
"""Derive evaluator-ready M0-M4 manifests after FP16 cache compilation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUILD_SCHEMA = "phase11_sensitivity_mixed_precision_build_v1"
MANIFEST_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
CACHE_SCHEMA = "phase11_mixed_fp16_shared_cache_compile_v1"
FINAL_STAGE_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_cache_final_v1"
FINALIZE_SCHEMA = "phase11_sensitivity_mixed_precision_cache_finalize_v1"
EXPECTED_CANDIDATES = ("M0", "M1", "M2", "M3", "M4")
EXPECTED_FP16_BLOCKS = (0, 14, 17, 18)
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
    row = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (row["size_bytes"], row["sha256"]) != expected:
        raise RuntimeError(f"identity drift for {path}: {row}; expected={expected}")
    return row


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def resolve_record_path(owner: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def relative_posix(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(strict=True), parent.resolve())).as_posix()


def require_under(path: Path, root: Path, role: str) -> None:
    path = path.resolve(strict=True)
    root = root.resolve(strict=True)
    if Path(os.path.commonpath((str(root), str(path)))) != root:
        raise RuntimeError(f"{role} escapes the authorized build root: {path}")


def audit_build(build_root: Path) -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    summary_path = build_root / "build_summary.json"
    summary = load_json(summary_path)
    if summary.get("schema") != BUILD_SCHEMA or summary.get("status") != "created_static_pass":
        raise RuntimeError("construction build summary schema/status drift")
    rows = summary.get("candidates", [])
    if [row.get("candidate_id") for row in rows] != list(EXPECTED_CANDIDATES):
        raise RuntimeError("construction build must contain M0-M4 in order")
    manifests = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        path = resolve_record_path(summary_path, str(row["path"]))
        expected = (int(row["identity"]["size_bytes"]), str(row["identity"]["sha256"]))
        identity(path, expected)
        payload = load_json(path)
        if (
            payload.get("schema") != MANIFEST_SCHEMA
            or payload.get("status") != "created_static_pass"
            or payload.get("candidate_id") != candidate_id
        ):
            raise RuntimeError(f"construction manifest drift for {candidate_id}")
        segments = payload.get("segments", [])
        if len(segments) != 25 or [item.get("index") for item in segments] != list(range(25)):
            raise RuntimeError(f"construction segment order drift for {candidate_id}")
        for segment in segments:
            index = int(segment["index"])
            model_path = resolve_record_path(path, str(segment["model"]))
            require_under(model_path, build_root, f"{candidate_id} model {index}")
            identity(
                model_path,
                (
                    int(segment["model_identity"]["size_bytes"]),
                    str(segment["model_identity"]["sha256"]),
                ),
            )
            if segment.get("cache") is None:
                if segment.get("precision") != "fp16" or segment.get("cache_identity") is not None:
                    raise RuntimeError(f"invalid null cache contract for {candidate_id} segment {index}")
            else:
                cache_path = resolve_record_path(path, str(segment["cache"]))
                require_under(cache_path, build_root, f"{candidate_id} cache {index}")
                identity(
                    cache_path,
                    (
                        int(segment["cache_identity"]["size_bytes"]),
                        str(segment["cache_identity"]["sha256"]),
                    ),
                )
        manifests[candidate_id] = (path, payload)
    return summary, manifests


def audit_caches(cache_result_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = load_json(cache_result_path)
    claims = payload.get("claims", {})
    if payload.get("schema") != CACHE_SCHEMA or payload.get("status") != "passed":
        raise RuntimeError("FP16 cache compile result schema/status drift")
    if not (
        claims.get("four_fp16_caches_compiled")
        and claims.get("four_fp16_segments_strict_migraphx_placement")
        and claims.get("four_fp16_outputs_finite")
    ):
        raise RuntimeError("FP16 cache compile essential claims are not passed")
    rows = payload.get("segments", [])
    if tuple(row.get("index") for row in rows) != EXPECTED_FP16_BLOCKS:
        raise RuntimeError("FP16 cache result segment order drift")
    audited = {}
    for row in rows:
        index = int(row["index"])
        essential = row.get("essential_gates", {})
        if not essential or not all(essential.values()) or not row.get("cache_and_placement_passed"):
            raise RuntimeError(f"FP16 cache essential gates failed for block {index}")
        counts = row.get("migraphx_profile", {}).get("provider_event_counts", {})
        if int(counts.get(MGX, 0)) <= 0 or int(counts.get(CPU, 0)) != 0:
            raise RuntimeError(f"FP16 cache provider placement drift for block {index}")
        cache_path = resolve_record_path(cache_result_path, str(row["cache"]["path"]))
        cache_identity = identity(
            cache_path,
            (int(row["cache"]["size_bytes"]), str(row["cache"]["sha256"])),
        )
        model_path = resolve_record_path(cache_result_path, str(row["model"]["path"]))
        model_identity = identity(
            model_path,
            (int(row["model"]["size_bytes"]), str(row["model"]["sha256"])),
        )
        profile_path = resolve_record_path(cache_result_path, str(row["migraphx_profile"]["path"]))
        profile_identity = identity(
            profile_path,
            (
                int(row["migraphx_profile"]["size_bytes"]),
                str(row["migraphx_profile"]["sha256"]),
            ),
        )
        audited[index] = {
            "cache_path": cache_path,
            "cache_identity": cache_identity,
            "model_path": model_path,
            "model_identity": model_identity,
            "profile_path": profile_path,
            "profile_identity": profile_identity,
            "provider_event_counts": counts,
            "comparison_cpu_vs_migraphx": row.get("comparison_cpu_vs_migraphx"),
            "diagnostic_strict_gates_not_an_admission_precondition": row.get(
                "diagnostic_strict_gates_not_an_admission_precondition"
            ),
        }
    return payload, audited


def build_final_payload(
    candidate_id: str,
    construction_path: Path,
    construction: dict[str, Any],
    cache_result_path: Path,
    caches: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    payload = copy.deepcopy(construction)
    # Keep the public candidate schema stable for the common evaluator.  The
    # lifecycle stage is recorded separately so construction manifests remain
    # distinguishable from cache-finalized manifests.
    payload["schema"] = MANIFEST_SCHEMA
    payload["status"] = "cache_finalized_static_pass"
    payload["manifest_stage"] = FINAL_STAGE_SCHEMA
    payload["construction_manifest"] = {
        "path": construction_path.name,
        **identity(construction_path),
    }
    payload["cache_finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    fp16_blocks = tuple(int(item) for item in payload.get("fp16_backbone_blocks", []))
    for row in payload["segments"]:
        index = int(row["index"])
        if row.get("precision") != "fp16":
            if row.get("cache") is None or row.get("cache_identity") is None:
                raise RuntimeError(f"non-FP16 cache missing in {candidate_id} segment {index}")
            continue
        evidence = caches.get(index)
        if evidence is None:
            raise RuntimeError(f"compiled FP16 cache missing for {candidate_id} segment {index}")
        if row["model_identity"] != evidence["model_identity"]:
            raise RuntimeError(f"FP16 model/cache lineage mismatch for {candidate_id} segment {index}")
        row["cache"] = relative_posix(evidence["cache_path"], construction_path.parent)
        row["cache_identity"] = evidence["cache_identity"]
        row["cache_action"] = "reuse_newly_compiled_fp16"
        row["cache_placement_evidence"] = {
            "profile": {
                "path": relative_posix(evidence["profile_path"], construction_path.parent),
                **evidence["profile_identity"],
            },
            "provider_event_counts": evidence["provider_event_counts"],
            "comparison_cpu_vs_migraphx": evidence["comparison_cpu_vs_migraphx"],
            "diagnostic_strict_gates_not_an_admission_precondition": evidence[
                "diagnostic_strict_gates_not_an_admission_precondition"
            ],
        }
    if tuple(row["index"] for row in payload["segments"] if row["precision"] == "fp16") != fp16_blocks:
        raise RuntimeError(f"finalized FP16 segment set drift for {candidate_id}")
    if any(row.get("cache") is None or row.get("cache_identity") is None for row in payload["segments"]):
        raise RuntimeError(f"finalized manifest retains null cache fields for {candidate_id}")
    payload["present_unique_cache_bytes"] = sum(
        size
        for size, _ in {
            (int(row["cache_identity"]["size_bytes"]), str(row["cache_identity"]["sha256"]))
            for row in payload["segments"]
        }
    )
    payload["new_fp16_cache_count_required"] = 0
    payload["fp16_cache_compile_evidence"] = {
        "path": relative_posix(cache_result_path, construction_path.parent),
        **identity(cache_result_path),
        "used_block_indices": list(fp16_blocks),
    }
    payload["claims"]["all_required_mxr_caches_present"] = True
    payload["claims"]["fp16_cache_compilation_and_placement_passed"] = True
    payload["claims"]["single_sample_migraphx_admitted"] = False
    payload["claims"]["task_accuracy_90"] = False
    payload["claim_boundary"] = (
        "All 25 model/cache identities are now populated and each new FP16 cache passed an isolated "
        "MIGraphX-positive/CPU-zero finite-output probe. End-to-end device OrtValue admission, fixed-90 "
        "task accuracy, repeatability, performance, and kernel precision remain unclaimed. The frozen "
        "INT8 strict-logit failure remains unchanged."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cache-compile-result", type=Path, required=True)
    args = parser.parse_args()
    build_root = args.build_root.resolve(strict=True)
    cache_result_path = args.cache_compile_result.resolve(strict=True)
    summary, manifests = audit_build(build_root)
    _, caches = audit_caches(cache_result_path)
    require_under(cache_result_path, build_root, "cache compile result")
    for index, row in caches.items():
        require_under(row["model_path"], build_root, f"FP16 model {index}")
        require_under(row["cache_path"], build_root, f"FP16 cache {index}")
        require_under(row["profile_path"], build_root, f"FP16 profile {index}")

    writes = []
    for candidate_id in EXPECTED_CANDIDATES:
        construction_path, construction = manifests[candidate_id]
        output_path = construction_path.parent / "mixed_precision_manifest.final.json"
        if output_path.exists():
            raise FileExistsError(output_path)
        payload = build_final_payload(
            candidate_id,
            construction_path,
            construction,
            cache_result_path,
            caches,
        )
        writes.append((candidate_id, output_path, payload))
    finalize_summary_path = build_root / "cache_finalize_summary.json"
    if finalize_summary_path.exists():
        raise FileExistsError(finalize_summary_path)

    finalized_rows = []
    for candidate_id, output_path, payload in writes:
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        finalized_rows.append(
            {
                "candidate_id": candidate_id,
                "path": str(output_path.relative_to(build_root)).replace("\\", "/"),
                "identity": identity(output_path),
                "fp16_backbone_blocks": payload["fp16_backbone_blocks"],
                "int8_backbone_blocks": payload["int8_backbone_blocks"],
                "all_25_caches_present": True,
            }
        )
    finalize_summary = {
        "schema": FINALIZE_SCHEMA,
        "status": "cache_finalized_static_pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "construction_build_summary": identity(build_root / "build_summary.json"),
        "cache_compile_result": identity(cache_result_path),
        "candidates": finalized_rows,
        "claims": {
            "m0_m4_final_manifests_created": True,
            "all_25_cache_fields_populated": True,
            "four_new_fp16_caches_placement_gated": True,
            "end_to_end_admission_complete": False,
            "task_accuracy_90_complete": False,
            "performance_complete": False,
        },
    }
    finalize_summary_path.write_text(
        json.dumps(finalize_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "cache_finalized_static_pass",
                "summary": {"path": str(finalize_summary_path), **identity(finalize_summary_path)},
                "candidates": finalized_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
