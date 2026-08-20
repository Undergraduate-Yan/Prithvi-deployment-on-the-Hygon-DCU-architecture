#!/usr/bin/env python3
"""Build the optional M5 candidate without modifying the frozen M0--M4 bundle.

M5 uses FP16 for encoder blocks 0 and 14--21, INT8 QDQ for the other
backbone blocks, and the already admitted FP32 fpn4-barrier head.  Blocks
0/14/17/18 (models and MXR caches), every retained INT8 segment, and the head
are materialized byte-for-byte from the finalized M4 bundle.  Only blocks
15/16/19/20/21 are extracted from the same frozen FP16 source; their new MXR
caches remain deliberately null until the incremental K100 cache step.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

import build_phase11_sensitivity_mixed_precision_m0_m4 as base
import phase11_mixed_precision_common as common


BUILD_SCHEMA = "phase11_sensitivity_mixed_precision_m5_build_v1"
MANIFEST_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
BASE_FP16_BLOCKS = (0, 14, 17, 18)
EXTRA_FP16_BLOCKS = (15, 16, 19, 20, 21)
M5_FP16_BLOCKS = (0, 14, 15, 16, 17, 18, 19, 20, 21)
EXPECTED_INT8_BLOCKS = tuple(index for index in range(24) if index not in M5_FP16_BLOCKS)


def resolve_record_path(owner: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def require_under(path: Path, root: Path, role: str, strict: bool = True) -> None:
    path = path.resolve(strict=strict)
    root = root.resolve(strict=True)
    if Path(os.path.commonpath((str(root), str(path)))) != root:
        raise RuntimeError(f"{role} escapes the authorized root: {path}")


def copy_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def materialize_verified(
    source: Path,
    target: Path,
    expected: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    base.identity(source, (int(expected["size_bytes"]), str(expected["sha256"])))
    base.materialize_file(source, target, mode)
    return base.identity(target, (int(expected["size_bytes"]), str(expected["sha256"])))


def audit_base_m4(
    build_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = build_root / "build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != base.BUILD_SCHEMA
        or summary.get("status") != "created_static_pass"
    ):
        raise RuntimeError("base M0--M4 construction summary schema/status drift")
    if [row.get("candidate_id") for row in summary.get("candidates", [])] != list(base.CANDIDATES):
        raise RuntimeError("base construction summary is not the frozen ordered M0--M4 build")

    finalize_path = build_root / "cache_finalize_summary.json"
    finalize = json.loads(finalize_path.read_text(encoding="utf-8"))
    if (
        finalize.get("schema") != "phase11_sensitivity_mixed_precision_cache_finalize_v1"
        or finalize.get("status") != "cache_finalized_static_pass"
    ):
        raise RuntimeError("base M0--M4 cache-finalize summary schema/status drift")
    construction_identity = finalize.get("construction_build_summary", {})
    base.identity(
        summary_path,
        (int(construction_identity["size_bytes"]), str(construction_identity["sha256"])),
    )
    finalized_rows = finalize.get("candidates", [])
    if [row.get("candidate_id") for row in finalized_rows] != list(base.CANDIDATES):
        raise RuntimeError("base cache-finalize summary is not the frozen ordered M0--M4 set")

    manifest_path = (build_root / "M4" / "mixed_precision_manifest.final.json").resolve(strict=True)
    payload, manifest_identity, rows = common.load_candidate(build_root, manifest_path)
    finalized_m4 = finalized_rows[-1]
    finalized_identity = finalized_m4.get("identity", {})
    base.identity(
        manifest_path,
        (int(finalized_identity["size_bytes"]), str(finalized_identity["sha256"])),
    )
    if payload.get("candidate_id") != "M4":
        raise RuntimeError("base manifest is not M4")
    if tuple(payload.get("fp16_backbone_blocks", ())) != BASE_FP16_BLOCKS:
        raise RuntimeError("base M4 FP16 mapping drift")
    contracts = common.validate_onnx_contracts(rows)
    return manifest_path, payload, manifest_identity, rows, contracts


def audit_reused_fp16_segment(index: int, row: dict[str, Any]) -> dict[str, Any]:
    model, node_types = base.audit_onnx(row["model_path"])
    inputs, outputs = base.graph_contract(model)
    if inputs != row["inputs"] or outputs != row["outputs"]:
        raise RuntimeError(f"reused FP16 block {index} contract drift")
    base.require_fp32_static_contract(inputs + outputs, f"reused FP16 block {index}")
    if node_types.get("LayerNormalization", 0):
        raise RuntimeError(f"reused FP16 block {index} retains LayerNormalization")
    if node_types.get("QuantizeLinear", 0) or node_types.get("DequantizeLinear", 0):
        raise RuntimeError(f"reused FP16 block {index} unexpectedly contains QDQ")
    initializer_types = Counter(int(item.data_type) for item in model.graph.initializer)
    if initializer_types.get(TensorProto.FLOAT16, 0) <= 0:
        raise RuntimeError(f"reused FP16 block {index} has no FP16 initializer")
    metadata = {item.key: item.value for item in model.metadata_props}
    expected_metadata = {
        "phase11_block_index": str(index),
        "phase11_external_io": "float32",
        "phase11_internal_precision": "float16_with_fp32_layernorm_statistic_islands",
        "phase11_fp16_source_sha256": base.FP16_SOURCE[1],
        "phase11_source_checkpoint_sha256": base.SOURCE_CHECKPOINT_SHA256,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError(f"reused FP16 block {index} metadata lineage drift")
    statistic_cast_count = sum(
        node.op_type == "Cast" and node.name.startswith(f"phase11_mixed_b{index:02d}_ln_")
        for node in model.graph.node
    )
    if statistic_cast_count != 8:
        raise RuntimeError(f"reused FP16 block {index} LayerNorm island audit drift")
    del model
    return {
        "index": index,
        "node_types": node_types,
        "initializer_data_types": {
            str(key): value for key, value in sorted(initializer_types.items())
        },
        "layernorm_statistic_island_cast_count": statistic_cast_count,
        "metadata": expected_metadata,
    }


def materialize_profile_evidence(
    base_manifest_path: Path,
    candidate_dir: Path,
    shared_root: Path,
    index: int,
    row: dict[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    evidence = copy.deepcopy(row.get("cache_placement_evidence"))
    if not isinstance(evidence, dict):
        return None
    profile = evidence.get("profile")
    if not isinstance(profile, dict) or not profile.get("path"):
        raise RuntimeError(f"reused FP16 block {index} has incomplete placement profile evidence")
    source = resolve_record_path(base_manifest_path, str(profile["path"]))
    target = shared_root / "reused_fp16_cache_evidence" / f"segment_{index:02d}_migraphx_profile.json"
    ident = materialize_verified(source, target, profile, mode)
    evidence["profile"] = {
        "path": base.relative_posix(target, candidate_dir),
        **ident,
    }
    return evidence


def materialize_base_segment(
    base_manifest_path: Path,
    candidate_dir: Path,
    shared_root: Path,
    row: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = int(row["index"])
    precision = str(row["precision"])
    if index == 24:
        model_target = shared_root / "models" / "24_upernet_decoder_head_fpn4barrier.onnx"
        cache_target = shared_root / "caches" / "segment_24_head_fpn4barrier.mxr"
        cache_action = "reuse_validated_head_from_frozen_m4"
        source_lineage = "frozen_m4_validated_fp32_head_with_fpn4_maxpool_graph_output_barrier"
    elif precision == "fp16":
        model_target = shared_root / "reused_fp16_segments" / f"{index:02d}_encoder_block_{index:02d}_fp16_ln32io32.onnx"
        cache_target = shared_root / "reused_fp16_caches" / f"segment_{index:02d}_fp16.mxr"
        cache_action = "reuse_admitted_m4_fp16"
        source_lineage = "frozen_m4_fp16_same_checkpoint_ln_fp32_island_io_fp32"
    elif precision == "int8_qdq":
        model_target = shared_root / "models" / f"{index:02d}_encoder_block_{index:02d}_int8_qdq.onnx"
        cache_target = shared_root / "caches" / f"segment_{index:02d}_int8_qdq.mxr"
        cache_action = "reuse_frozen_m4_int8"
        source_lineage = "frozen_m4_int8_backbone_qdq_compat_segment"
    else:
        raise RuntimeError(f"unexpected base precision at segment {index}: {precision}")

    model_ident = materialize_verified(row["model_path"], model_target, row["model_identity"], mode)
    cache_ident = materialize_verified(row["cache_path"], cache_target, row["cache_identity"], mode)
    segment = {
        "index": index,
        "label": row["label"],
        "precision": precision,
        "model": base.relative_posix(model_target, candidate_dir),
        "model_identity": model_ident,
        "cache": base.relative_posix(cache_target, candidate_dir),
        "cache_identity": cache_ident,
        "cache_action": cache_action,
        "inputs": copy.deepcopy(row["inputs"]),
        "outputs": copy.deepcopy(row["outputs"]),
        "source_lineage": source_lineage,
    }
    if precision == "fp16":
        segment["cache_placement_evidence"] = materialize_profile_evidence(
            base_manifest_path, candidate_dir, shared_root, index, row, mode
        )
    return segment, {
        "index": index,
        "precision": precision,
        "model": model_ident,
        "cache": cache_ident,
    }


def make_new_fp16_segment(
    candidate_dir: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "index": int(row["index"]),
        "label": str(row["label"]),
        "precision": "fp16",
        "model": base.relative_posix(row["path"], candidate_dir),
        "model_identity": copy_identity(row["identity"]),
        "cache": None,
        "cache_identity": None,
        "cache_action": "compile_new_fp16_m5_incremental",
        "inputs": copy.deepcopy(row["inputs"]),
        "outputs": copy.deepcopy(row["outputs"]),
        "source_lineage": "frozen_fp16_full_same_checkpoint_extracted_ln_fp32_island_io_fp32",
    }


def validate_candidate_segments(staging_root: Path, segments: list[dict[str, Any]]) -> None:
    if len(segments) != 25 or [row["index"] for row in segments] != list(range(25)):
        raise RuntimeError("M5 must contain exactly 25 ordered segments")
    if tuple(row["index"] for row in segments[:24] if row["precision"] == "fp16") != M5_FP16_BLOCKS:
        raise RuntimeError("M5 FP16 mapping drift")
    if tuple(row["index"] for row in segments[:24] if row["precision"] == "int8_qdq") != EXPECTED_INT8_BLOCKS:
        raise RuntimeError("M5 INT8 mapping drift")
    for previous, following in zip(segments, segments[1:24]):
        if previous["outputs"] != following["inputs"]:
            raise RuntimeError(f"M5 adjacent contract mismatch: {previous['index']} -> {following['index']}")
    if [row["name"] for row in segments[24]["inputs"]] != base.HEAD_INPUTS:
        raise RuntimeError("M5 head input contract drift")
    for row in segments:
        for key in ("model", "cache"):
            raw = row.get(key)
            if raw is None:
                continue
            resolved = (staging_root / "M5" / str(raw)).resolve(strict=True)
            require_under(resolved, staging_root, f"M5 segment {row['index']} {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-build-root", type=Path, required=True)
    parser.add_argument("--fp16-source", type=Path, required=True)
    parser.add_argument("--fp16-source-report", type=Path, required=True)
    parser.add_argument("--int8-quantization-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--payload-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    base_build_root = args.base_build_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite M5 output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(output_root.name + f".building-{os.getpid()}")
    if staging_root.exists():
        raise FileExistsError(staging_root)
    staging_root.mkdir()

    try:
        base_manifest_path, m4, m4_identity, base_rows, base_contracts = audit_base_m4(
            base_build_root
        )
        provenance = base.audit_provenance(
            args.fp16_source_report.resolve(strict=True),
            args.int8_quantization_report.resolve(strict=True),
        )
        m4_lineage = m4.get("source_identities", {}).get("common_lineage", {})
        for key in ("source_fp32_sha256", "source_checkpoint_sha256"):
            if m4_lineage.get(key) != provenance[key]:
                raise RuntimeError(f"M4/frozen-source provenance mismatch for {key}")

        shared_root = staging_root / "_shared"
        shared_root.mkdir()
        candidate_dir = staging_root / "M5"
        candidate_dir.mkdir()

        reused_fp16_audits = [
            audit_reused_fp16_segment(index, base_rows[index]) for index in BASE_FP16_BLOCKS
        ]
        base.FP16_BLOCKS = EXTRA_FP16_BLOCKS
        extraction_source, fp16_source_audit = base.prepare_extraction_source(
            args.fp16_source.resolve(strict=True), shared_root
        )
        new_fp16_rows = base.build_fp16_segments(
            extraction_source,
            shared_root / "new_fp16_segments",
            base_rows,
        )
        if extraction_source.parent == shared_root and extraction_source.name.startswith(".fp16_source"):
            extraction_source.unlink()
        if tuple(sorted(new_fp16_rows)) != EXTRA_FP16_BLOCKS:
            raise RuntimeError("new M5 FP16 extraction set drift")

        materialized = []
        segments = []
        for index in range(25):
            if index in EXTRA_FP16_BLOCKS:
                segments.append(make_new_fp16_segment(candidate_dir, new_fp16_rows[index]))
                continue
            segment, audit = materialize_base_segment(
                base_manifest_path,
                candidate_dir,
                shared_root,
                base_rows[index],
                args.payload_mode,
            )
            segments.append(segment)
            materialized.append(audit)
        validate_candidate_segments(staging_root, segments)

        unique_models = {
            (int(row["model_identity"]["size_bytes"]), str(row["model_identity"]["sha256"]))
            for row in segments
        }
        unique_caches = {
            (int(row["cache_identity"]["size_bytes"]), str(row["cache_identity"]["sha256"]))
            for row in segments
            if row["cache_identity"] is not None
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "created_static_pass",
            "candidate_id": "M5",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fp16_backbone_blocks": list(M5_FP16_BLOCKS),
            "int8_backbone_blocks": list(EXPECTED_INT8_BLOCKS),
            "head_precision": "fp32_compat_barrier",
            "segment_count": 25,
            "retain_encoder_outputs_after_segments": [5, 11, 17, 23],
            "external_io_contract": copy.deepcopy(m4["external_io_contract"]),
            "source_identities": {
                **copy.deepcopy(m4.get("source_identities", {})),
                "fp16_full_model": {
                    **fp16_source_audit["identity"],
                    "runtime_payload": False,
                },
                "base_m4_final_manifest": {
                    "size_bytes": m4_identity["size_bytes"],
                    "sha256": m4_identity["sha256"],
                    "runtime_payload": False,
                },
                "common_lineage": provenance,
            },
            "segments": segments,
            "unique_model_bytes": sum(size for size, _ in unique_models),
            "present_unique_cache_bytes": sum(size for size, _ in unique_caches),
            "new_fp16_cache_count_required": len(EXTRA_FP16_BLOCKS),
            "claims": {
                "static_onnx_valid": True,
                "all_segment_identities_locked": True,
                "all_external_segment_io_fp32_static_batch1": True,
                "fp16_blocks_derived_from_frozen_same_checkpoint_source": True,
                "fp16_layernorm_fp32_statistic_islands_present": True,
                "base_m4_fp16_models_and_caches_reused_byte_for_byte": True,
                "int8_segments_reused_byte_for_byte": True,
                "repaired_head_reused_byte_for_byte": True,
                "all_required_mxr_caches_present": False,
                "cpu_sequential_finite": False,
                "single_sample_migraphx_admitted": False,
                "task_accuracy_90": False,
                "three_run_task_repeatability": False,
                "performance": False,
                "native_int8_kernel_verified": False,
                "deployment_ready": False,
            },
            "claim_boundary": (
                "Conditional M5 construction and static contracts only. Blocks 0/14/17/18 reuse "
                "the finalized M4 FP16 model/cache artifacts; blocks 15/16/19/20/21 require new K100 "
                "MXR compilation. End-to-end placement, fixed-90 task admission, repeatability, "
                "performance, and kernel precision remain unclaimed. Frozen strict-logit failures are "
                "not overridden."
            ),
        }
        construction_path = candidate_dir / "mixed_precision_manifest.json"
        common.json_dump(construction_path, manifest)

        summary = {
            "schema": BUILD_SCHEMA,
            "status": "created_static_pass",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "candidate": {
                "candidate_id": "M5",
                "path": "M5/mixed_precision_manifest.json",
                "identity": copy_identity(base.identity(construction_path)),
                "fp16_backbone_blocks": list(M5_FP16_BLOCKS),
                "int8_backbone_blocks": list(EXPECTED_INT8_BLOCKS),
            },
            "base_m4": {
                "build_root": str(base_build_root),
                "manifest": copy_identity(m4_identity),
                "validated_onnx_contract_count": len(base_contracts),
            },
            "reused_fp16_segments": reused_fp16_audits,
            "new_fp16_segments": [
                {
                    "index": index,
                    "label": row["label"],
                    "path": base.relative_posix(row["path"], staging_root),
                    "identity": copy_identity(row["identity"]),
                    "layernorm": row["layernorm"],
                    "boundary_casts": row["boundary_casts"],
                }
                for index, row in sorted(new_fp16_rows.items())
            ],
            "runtime_payload_materialization": {
                "mode": args.payload_mode,
                "reused_segment_count": len(materialized),
                "reused_model_count": len(materialized),
                "reused_cache_count": len(materialized),
                "all_runtime_payload_under_output_root": True,
            },
            "claims": {
                "m5_manifest_generation_complete": True,
                "source_lineage_audited": True,
                "frozen_m4_models_and_caches_audited": True,
                "four_existing_fp16_models_and_caches_reused": True,
                "five_new_fp16_segments_static_valid": True,
                "runtime_admission_complete": False,
            },
        }
        summary_path = staging_root / "build_summary.json"
        common.json_dump(summary_path, summary)
        summary_identity = copy_identity(base.identity(summary_path))
        construction_identity = copy_identity(base.identity(construction_path))
        staging_root.rename(output_root)
        print(
            json.dumps(
                {
                    "status": "created_static_pass",
                    "output_root": str(output_root),
                    "build_summary": {
                        "path": str(output_root / "build_summary.json"),
                        **summary_identity,
                    },
                    "candidate_manifest": {
                        "path": str(output_root / "M5" / "mixed_precision_manifest.json"),
                        **construction_identity,
                    },
                    "reused_fp16_blocks": list(BASE_FP16_BLOCKS),
                    "new_fp16_blocks": list(EXTRA_FP16_BLOCKS),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        failure = {
            "schema": BUILD_SCHEMA,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_boundary": "No artifact from this partial M5 staging directory is admitted.",
        }
        (staging_root / "BUILD_FAILED.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
