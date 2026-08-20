#!/usr/bin/env python3
"""Finalize the optional M5 manifest after five incremental FP16 caches pass."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUILD_SCHEMA = "phase11_sensitivity_mixed_precision_m5_build_v1"
MANIFEST_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
CACHE_SCHEMA = "phase11_mixed_fp16_m5_incremental_cache_compile_v1"
FINAL_STAGE_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_cache_final_v1"
FINALIZE_SCHEMA = "phase11_sensitivity_mixed_precision_m5_cache_finalize_v1"
M5_FP16_BLOCKS = (0, 14, 15, 16, 17, 18, 19, 20, 21)
REUSED_FP16_BLOCKS = (0, 14, 17, 18)
NEW_FP16_BLOCKS = (15, 16, 19, 20, 21)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, expected: tuple[int, str] | dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    row = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None:
        if isinstance(expected, dict):
            expected_pair = (int(expected["size_bytes"]), str(expected["sha256"]))
        else:
            expected_pair = (int(expected[0]), str(expected[1]))
        if (row["size_bytes"], row["sha256"]) != expected_pair:
            raise RuntimeError(f"identity drift for {path}: {row}; expected={expected_pair}")
    return row


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


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
        raise RuntimeError(f"{role} escapes the authorized M5 build root: {path}")


def audit_build(build_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    summary_path = build_root / "build_summary.json"
    summary = load_json(summary_path)
    if summary.get("schema") != BUILD_SCHEMA or summary.get("status") != "created_static_pass":
        raise RuntimeError("M5 construction build summary schema/status drift")
    candidate = summary.get("candidate", {})
    if candidate.get("candidate_id") != "M5":
        raise RuntimeError("M5 construction summary candidate drift")
    construction_path = resolve_record_path(summary_path, str(candidate.get("path")))
    require_under(construction_path, build_root, "M5 construction manifest")
    identity(construction_path, candidate.get("identity"))
    payload = load_json(construction_path)
    if (
        payload.get("schema") != MANIFEST_SCHEMA
        or payload.get("status") != "created_static_pass"
        or payload.get("candidate_id") != "M5"
    ):
        raise RuntimeError("M5 construction manifest schema/status/ID drift")
    if tuple(payload.get("fp16_backbone_blocks", ())) != M5_FP16_BLOCKS:
        raise RuntimeError("M5 construction FP16 mapping drift")
    segments = payload.get("segments", [])
    if len(segments) != 25 or [row.get("index") for row in segments] != list(range(25)):
        raise RuntimeError("M5 construction segment order drift")
    for row in segments:
        index = int(row["index"])
        model_path = resolve_record_path(construction_path, str(row["model"]))
        require_under(model_path, build_root, f"M5 model {index}")
        identity(model_path, row["model_identity"])
        cache_value = row.get("cache")
        cache_identity = row.get("cache_identity")
        if index in NEW_FP16_BLOCKS:
            if row.get("precision") != "fp16" or cache_value is not None or cache_identity is not None:
                raise RuntimeError(f"M5 new FP16 construction cache contract drift at {index}")
            if row.get("cache_action") != "compile_new_fp16_m5_incremental":
                raise RuntimeError(f"M5 new FP16 cache action drift at {index}")
        else:
            if cache_value is None or cache_identity is None:
                raise RuntimeError(f"M5 reused cache missing at segment {index}")
            cache_path = resolve_record_path(construction_path, str(cache_value))
            require_under(cache_path, build_root, f"M5 reused cache {index}")
            identity(cache_path, cache_identity)
    return construction_path, payload, identity(summary_path)


def audit_caches(
    cache_result_path: Path,
    build_root: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = load_json(cache_result_path)
    claims = payload.get("claims", {})
    if payload.get("schema") != CACHE_SCHEMA or payload.get("status") != "passed":
        raise RuntimeError("M5 incremental cache result schema/status drift")
    required_claims = (
        "five_new_fp16_caches_compiled",
        "five_new_fp16_segments_strict_migraphx_placement",
        "five_new_fp16_outputs_finite",
        "four_m4_fp16_caches_reused_not_recompiled",
    )
    if not all(claims.get(key) for key in required_claims):
        raise RuntimeError("M5 incremental cache essential claims are not passed")
    rows = payload.get("segments", [])
    if tuple(row.get("index") for row in rows) != NEW_FP16_BLOCKS:
        raise RuntimeError("M5 incremental cache segment order drift")
    audited = {}
    for row in rows:
        index = int(row["index"])
        essential = row.get("essential_gates", {})
        if not essential or not all(essential.values()) or not row.get("cache_and_placement_passed"):
            raise RuntimeError(f"M5 FP16 cache essential gates failed for block {index}")
        counts = row.get("migraphx_profile", {}).get("provider_event_counts", {})
        if int(counts.get(MGX, 0)) <= 0 or int(counts.get(CPU, 0)) != 0:
            raise RuntimeError(f"M5 FP16 cache provider placement drift for block {index}")
        cache_path = resolve_record_path(cache_result_path, str(row["cache"]["path"]))
        model_path = resolve_record_path(cache_result_path, str(row["model"]["path"]))
        profile_path = resolve_record_path(cache_result_path, str(row["migraphx_profile"]["path"]))
        for role, path in (
            (f"M5 new cache {index}", cache_path),
            (f"M5 new model {index}", model_path),
            (f"M5 placement profile {index}", profile_path),
        ):
            require_under(path, build_root, role)
        audited[index] = {
            "cache_path": cache_path,
            "cache_identity": identity(cache_path, row["cache"]),
            "model_path": model_path,
            "model_identity": identity(model_path, row["model"]),
            "profile_path": profile_path,
            "profile_identity": identity(profile_path, row["migraphx_profile"]),
            "provider_event_counts": counts,
            "comparison_cpu_vs_migraphx": row.get("comparison_cpu_vs_migraphx"),
            "diagnostic_strict_gates_not_an_admission_precondition": row.get(
                "diagnostic_strict_gates_not_an_admission_precondition"
            ),
        }
    return payload, audited


def finalize_payload(
    construction_path: Path,
    construction: dict[str, Any],
    cache_result_path: Path,
    caches: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    payload = copy.deepcopy(construction)
    payload["schema"] = MANIFEST_SCHEMA
    payload["status"] = "cache_finalized_static_pass"
    payload["manifest_stage"] = FINAL_STAGE_SCHEMA
    payload["construction_manifest"] = {
        "path": construction_path.name,
        **identity(construction_path),
    }
    payload["cache_finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    for row in payload["segments"]:
        index = int(row["index"])
        if index not in NEW_FP16_BLOCKS:
            if row.get("cache") is None or row.get("cache_identity") is None:
                raise RuntimeError(f"M5 reused cache missing at segment {index}")
            continue
        evidence = caches.get(index)
        if evidence is None:
            raise RuntimeError(f"M5 compiled FP16 cache missing for block {index}")
        if row["model_identity"] != evidence["model_identity"]:
            raise RuntimeError(f"M5 FP16 model/cache lineage mismatch for block {index}")
        row["cache"] = relative_posix(evidence["cache_path"], construction_path.parent)
        row["cache_identity"] = evidence["cache_identity"]
        row["cache_action"] = "reuse_newly_compiled_fp16_m5_incremental"
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

    if tuple(row["index"] for row in payload["segments"] if row["precision"] == "fp16") != M5_FP16_BLOCKS:
        raise RuntimeError("finalized M5 FP16 segment mapping drift")
    if any(row.get("cache") is None or row.get("cache_identity") is None for row in payload["segments"]):
        raise RuntimeError("finalized M5 manifest retains a null cache")
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
        "newly_compiled_block_indices": list(NEW_FP16_BLOCKS),
        "reused_m4_block_indices": list(REUSED_FP16_BLOCKS),
    }
    payload["claims"]["all_required_mxr_caches_present"] = True
    payload["claims"]["fp16_cache_compilation_and_placement_passed"] = True
    payload["claims"]["four_existing_fp16_caches_reused"] = True
    payload["claims"]["five_new_fp16_caches_compiled"] = True
    payload["claims"]["single_sample_migraphx_admitted"] = False
    payload["claims"]["task_accuracy_90"] = False
    payload["claim_boundary"] = (
        "All 25 M5 model/cache identities are populated. The four existing FP16 caches were reused "
        "byte-for-byte from finalized M4, and the five new FP16 caches passed isolated "
        "MIGraphX-positive/CPU-zero finite-output probes. End-to-end device OrtValue admission, "
        "fixed-90 task accuracy, repeatability, performance, and kernel precision remain unclaimed. "
        "Frozen strict-logit failures remain unchanged."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cache-compile-result", type=Path, required=True)
    args = parser.parse_args()

    build_root = args.build_root.resolve(strict=True)
    cache_result_path = args.cache_compile_result.resolve(strict=True)
    require_under(cache_result_path, build_root, "M5 incremental cache result")
    construction_path, construction, build_summary_identity = audit_build(build_root)
    _, caches = audit_caches(cache_result_path, build_root)

    output_path = construction_path.parent / "mixed_precision_manifest.final.json"
    summary_path = build_root / "cache_finalize_summary.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing M5 final manifest/summary")
    payload = finalize_payload(construction_path, construction, cache_result_path, caches)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    final_identity = identity(output_path)
    summary = {
        "schema": FINALIZE_SCHEMA,
        "status": "cache_finalized_static_pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "construction_build_summary": build_summary_identity,
        "cache_compile_result": identity(cache_result_path),
        "candidate": {
            "candidate_id": "M5",
            "path": "M5/mixed_precision_manifest.final.json",
            "identity": final_identity,
            "fp16_backbone_blocks": list(M5_FP16_BLOCKS),
            "int8_backbone_blocks": [
                index for index in range(24) if index not in M5_FP16_BLOCKS
            ],
            "all_25_caches_present": True,
        },
        "claims": {
            "m5_final_manifest_created": True,
            "all_25_cache_fields_populated": True,
            "four_m4_fp16_caches_reused": True,
            "five_new_fp16_caches_placement_gated": True,
            "end_to_end_admission_complete": False,
            "task_accuracy_90_complete": False,
            "performance_complete": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "cache_finalized_static_pass",
                "summary": {"path": str(summary_path), **identity(summary_path)},
                "candidate": {"path": str(output_path), **final_identity},
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
