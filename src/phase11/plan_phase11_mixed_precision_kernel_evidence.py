#!/usr/bin/env python3
"""Build the v3 deduplicated prepass/direct-hipprof plan for M0--M5 blocks."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import phase11_mixed_precision_common as common


SCHEMA = "phase11_mixed_precision_kernel_profile_plan_v3"
EXPECTED = ("M0", "M1", "M2", "M3", "M4", "M5")
PROFILE_PRECISIONS = {"int8_qdq", "fp16"}


def parse_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--manifest must be CANDIDATE=PATH")
    candidate, raw_path = value.split("=", 1)
    candidate = candidate.strip().upper()
    if candidate not in common.EXPECTED_VARIANTS:
        raise argparse.ArgumentTypeError(f"unsupported candidate: {candidate!r}")
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("manifest path is empty")
    return candidate, Path(raw_path)


def relative_under(root: Path, path: Path) -> str:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"manifest is outside --bundle: {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=parse_manifest, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-profile-tasks",
        type=int,
        default=0,
        help="Select only the first N identity-sorted tasks for a non-formal smoke run (0=all).",
    )
    parser.add_argument(
        "--require-all-m0-m5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly M0--M5 (disable only for a final Pareto subset).",
    )
    args = parser.parse_args()
    if args.max_profile_tasks < 0:
        parser.error("--max-profile-tasks must be 0 or a positive integer")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle.resolve(strict=True)

    manifest_args = dict(args.manifest)
    if len(manifest_args) != len(args.manifest):
        raise RuntimeError("duplicate candidate manifest argument")
    ordered_candidates = tuple(item for item in EXPECTED if item in manifest_args)
    if args.require_all_m0_m5 and ordered_candidates != EXPECTED:
        raise RuntimeError(f"full kernel plan requires {EXPECTED}; got {ordered_candidates}")
    if not ordered_candidates:
        raise RuntimeError("no candidate manifests selected")

    candidates: dict[str, dict] = {}
    candidate_rows: dict[str, list[dict]] = {}
    artifact_references: dict[tuple, list[dict]] = defaultdict(list)
    for candidate_id in ordered_candidates:
        manifest_path = manifest_args[candidate_id].resolve(strict=True)
        manifest, manifest_identity, rows = common.load_candidate(bundle, manifest_path)
        if manifest.get("candidate_id") != candidate_id:
            raise RuntimeError(f"candidate/manifest mismatch for {candidate_id}")
        manifest_relative = relative_under(bundle, manifest_path)
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "manifest_relative_to_bundle": manifest_relative,
            "manifest_identity": manifest_identity,
            "fp16_backbone_blocks": list(manifest["fp16_backbone_blocks"]),
            "int8_backbone_blocks": list(manifest["int8_backbone_blocks"]),
        }
        candidate_rows[candidate_id] = rows
        for row in rows[:24]:
            precision = row["precision"]
            if precision not in PROFILE_PRECISIONS:
                raise RuntimeError(f"unexpected backbone precision: {precision}")
            key = (
                int(row["index"]),
                precision,
                int(row["verified_model_identity"]["size_bytes"]),
                row["verified_model_identity"]["sha256"],
                int(row["verified_cache_identity"]["size_bytes"]),
                row["verified_cache_identity"]["sha256"],
            )
            artifact_references[key].append(
                {
                    "candidate_id": candidate_id,
                    "manifest_relative_to_bundle": manifest_relative,
                    "manifest_identity": manifest_identity,
                    "segment_index": int(row["index"]),
                    "label": row["label"],
                }
            )

    tasks = []
    key_to_evidence: dict[tuple, str] = {}
    for ordinal, (key, references) in enumerate(
        sorted(artifact_references.items(), key=lambda item: (item[0][0], item[0][1], item[0][3])),
        start=1,
    ):
        index, precision, model_size, model_sha, cache_size, cache_sha = key
        canonical = min(references, key=lambda row: EXPECTED.index(row["candidate_id"]))
        canonical_rows = candidate_rows[canonical["candidate_id"]]
        target_row = canonical_rows[index]
        if len(target_row["input_contracts"]) != 1:
            raise RuntimeError(f"encoder block {index} must have exactly one input contract")
        upstream_lineage = [
            {
                "index": int(row["index"]),
                "precision": row["precision"],
                "model_identity": row["verified_model_identity"],
                "cache_identity": row["verified_cache_identity"],
            }
            for row in canonical_rows[:index]
        ]
        evidence_key = (
            f"block_{index:02d}_{precision}_{model_sha[:10]}_{cache_sha[:10]}"
        )
        if not re.fullmatch(r"[a-z0-9_]+", evidence_key):
            raise RuntimeError(f"unsafe evidence key: {evidence_key}")
        key_to_evidence[key] = evidence_key
        tasks.append(
            {
                "ordinal": ordinal,
                "evidence_key": evidence_key,
                "segment_index": index,
                "label": canonical["label"],
                "precision": precision,
                "expected_kernel_token": "I8II" if precision == "int8_qdq" else "HBH",
                "canonical_candidate_id": canonical["candidate_id"],
                "canonical_manifest_relative_to_bundle": canonical[
                    "manifest_relative_to_bundle"
                ],
                "canonical_manifest_identity": canonical["manifest_identity"],
                "model_identity": {"size_bytes": model_size, "sha256": model_sha},
                "cache_identity": {"size_bytes": cache_size, "sha256": cache_sha},
                "boundary_input_contract": target_row["input_contracts"][0],
                "boundary_prepass_lineage": {
                    "canonical_candidate_id": canonical["candidate_id"],
                    "canonical_manifest_identity": canonical["manifest_identity"],
                    "upstream_segments": upstream_lineage,
                    "target_segment": {
                        "index": index,
                        "precision": precision,
                        "model_identity": {"size_bytes": model_size, "sha256": model_sha},
                        "cache_identity": {"size_bytes": cache_size, "sha256": cache_sha},
                    },
                },
                "referenced_by_candidates": [row["candidate_id"] for row in references],
                "reference_count": len(references),
                "reuse_basis": (
                    "The same manifest-locked static ONNX and MXR identities select kernels. "
                    "One canonical candidate materializes a realistic boundary input in an "
                    "untracked prepass; the resulting trace is reused only for this identical "
                    "block index, precision, ONNX, and MXR identity."
                ),
            }
        )

    matrix = []
    for candidate_id in ordered_candidates:
        for row in candidate_rows[candidate_id][:24]:
            key = (
                int(row["index"]),
                row["precision"],
                int(row["verified_model_identity"]["size_bytes"]),
                row["verified_model_identity"]["sha256"],
                int(row["verified_cache_identity"]["size_bytes"]),
                row["verified_cache_identity"]["sha256"],
            )
            matrix.append(
                {
                    "candidate_id": candidate_id,
                    "segment_index": int(row["index"]),
                    "label": row["label"],
                    "precision": row["precision"],
                    "evidence_key": key_to_evidence[key],
                }
            )

    if args.require_all_m0_m5:
        int8_task_count = sum(row["precision"] == "int8_qdq" for row in tasks)
        fp16_task_count = sum(row["precision"] == "fp16" for row in tasks)
        if (len(tasks), int8_task_count, fp16_task_count, len(matrix)) != (33, 24, 9, 144):
            raise RuntimeError(
                "full M0-M5 artifact reuse drift: expected 33 unique tasks "
                f"(24 INT8 + 9 FP16) and 144 matrix rows; got "
                f"{len(tasks)}, {int8_task_count}, {fp16_task_count}, {len(matrix)}"
            )

    full_counts = {
        "candidate_count": len(ordered_candidates),
        "matrix_rows": len(matrix),
        "unique_profile_tasks": len(tasks),
        "int8_qdq_tasks": sum(row["precision"] == "int8_qdq" for row in tasks),
        "fp16_tasks": sum(row["precision"] == "fp16" for row in tasks),
    }
    smoke_mode = args.max_profile_tasks > 0
    if smoke_mode:
        if args.max_profile_tasks > len(tasks):
            raise RuntimeError(
                f"--max-profile-tasks={args.max_profile_tasks} exceeds {len(tasks)} planned tasks"
            )
        tasks = tasks[: args.max_profile_tasks]
        selected_keys = {row["evidence_key"] for row in tasks}
        matrix = [row for row in matrix if row["evidence_key"] in selected_keys]

    plan = {
        "schema": SCHEMA,
        "status": (
            "planned_identity_locked_smoke_no_full_kernel_claim"
            if smoke_mode
            else "planned_identity_locked_no_kernel_claim"
        ),
        "plan_mode": "selected_task_smoke" if smoke_mode else "formal_full_selected_candidates",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle),
        "generator": common.identity(Path(__file__)),
        "common_module": common.identity(Path(common.__file__)),
        "candidates": [candidates[item] for item in ordered_candidates],
        "profile_tasks": tasks,
        "candidate_block_matrix": matrix,
        "counts": {
            "candidate_count": len(ordered_candidates),
            "matrix_rows": len(matrix),
            "unique_profile_tasks": len(tasks),
            "int8_qdq_tasks": sum(row["precision"] == "int8_qdq" for row in tasks),
            "fp16_tasks": sum(row["precision"] == "fp16" for row in tasks),
        },
        "full_counts_before_smoke_filter": full_counts,
        "claims": {
            "all_manifest_model_cache_identities_verified": True,
            "all_boundary_prepass_lineages_planned": True,
            "formal_full_candidate_mapping_planned": not smoke_mode,
            "kernel_precision_verified": False,
            "migraphx_provider_placement_is_kernel_proof": False,
            "strict_numeric_equivalence_confirmed": False,
        },
        "claim_boundary": (
            "This file is only an identity-locked execution plan. Every selected task requires an untracked "
            "device-OrtValue prepass followed by a direct outer hipprof target that executes no "
            "upstream segment. A selected-task smoke plan never proves the full candidate mapping. "
            "No I8II, HBH, native-INT8, speedup, strict-equivalence, or deployment claim can be made "
            "until the applicable hipprof records have been parsed and admitted."
        ),
    }
    plan_path = args.output_dir / "profile_plan.json"
    common.json_dump(plan_path, plan)

    tsv_path = args.output_dir / "profile_plan.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        for task in tasks:
            writer.writerow(
                [
                    task["ordinal"],
                    task["evidence_key"],
                    task["canonical_candidate_id"],
                    task["canonical_manifest_relative_to_bundle"],
                    task["segment_index"],
                    task["precision"],
                    task["model_identity"]["sha256"],
                    task["cache_identity"]["sha256"],
                ]
            )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan": common.identity(plan_path),
                "tsv": common.identity(tsv_path),
                "counts": plan["counts"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
