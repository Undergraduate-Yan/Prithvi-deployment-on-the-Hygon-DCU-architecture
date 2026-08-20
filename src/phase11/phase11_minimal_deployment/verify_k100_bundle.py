#!/usr/bin/env python3
"""Offline, fail-closed verification of a Phase 11 minimal K100 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "phase11_k100_minimal_deployment_bundle_v1"
LSMOD_SIZE = 819_664
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"
FP16_MODEL_PAIR = (
    638_970_735,
    "8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7",
)
FP16_CACHE_PAIR = (
    708_193_607,
    "9fe8985dc8ce4a3cf9d829ad66a91de41bb67b22e01b8181c43bf15c74f7dd0a",
)
FP16_EVIDENCE_PAIRS = {
    "source_manifest": (
        15_591,
        "93293dc0909f5158f983ce9dc5b7953cf4956a116f73e54aab7e86e2a7cd3fba",
    ),
    "single_admission": (
        3_316,
        "5e93e828ad44ca703c1eac5b08fd2c332d70002cec13c6d3c3c1b688f5a35a89",
    ),
    "fresh_task_result": (
        6_042,
        "c5a55d19320a1a7b52e2fd8add2192b3015c258420b5cc5996e9ca1953a32397",
    ),
    "task_admission": (
        3_720,
        "9b9ef29eea9c953fdfbd0e1bbf85cff1f689e2bef14b931a953f235c7444da5c",
    ),
}
EXPECTED_VARIANTS = {
    "M0": (),
    "M1": (0,),
    "M2": (0, 14),
    "M3": (0, 14, 17),
    "M4": (0, 14, 17, 18),
    "M5": (0, 14, 15, 16, 17, 18, 19, 20, 21),
}
EXPECTED_HEAD_LINEAGES = {
    **{
        candidate: "validated_int8_backbone_fp32_head_with_fpn4_maxpool_graph_output_barrier"
        for candidate in ("M0", "M1", "M2", "M3", "M4")
    },
    "M5": "frozen_m4_validated_fp32_head_with_fpn4_maxpool_graph_output_barrier",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, child: Path) -> bool:
    return child == root or root in child.parents


def resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"invalid relative payload path: {relative!r}")
    path = (root / relative).resolve(strict=True)
    if not contained(root, path):
        raise RuntimeError(f"payload escapes bundle: {relative!r}")
    return path


def pair(row: dict) -> tuple[int, str]:
    return int(row["size_bytes"]), str(row["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = args.bundle.resolve(strict=True)
    manifest_path = resolve(root, "manifest.json")
    manifest_digest = sha256(manifest_path)
    if args.expected_manifest_sha256 and manifest_digest != args.expected_manifest_sha256:
        raise RuntimeError(
            f"manifest SHA drift: {manifest_digest}; expected={args.expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "static_bundle_identity_locked":
        raise RuntimeError("bundle schema/status drift")
    claims = manifest.get("claims", {})
    if not claims.get("static_bundle_ready"):
        raise RuntimeError("static_bundle_ready claim is false")
    # Static packaging must never silently promote runtime/deployment claims.
    if claims.get("runtime_validated") or claims.get("deployment_ready"):
        raise RuntimeError("static manifest must not claim runtime validation or deployment readiness")
    boundaries = manifest.get("claim_boundaries", {})
    if boundaries.get("historical_int8_strict_logits") != "failed_immutable":
        raise RuntimeError("historical INT8 strict-failure boundary was not preserved")
    if boundaries.get("native_int8_kernel") not in {"unverified", "not_applicable"}:
        raise RuntimeError("unsupported native-kernel claim in static bundle")
    input_contract = manifest.get("input_contract", {})
    if input_contract.get("dtype") != "float32" or input_contract.get("shape") != [1, 6, 224, 224]:
        raise RuntimeError("external input contract drift")
    if input_contract.get("external_mean_std_normalization") is not False:
        raise RuntimeError("bundle must forbid external mean/std normalization")
    source_authorization = manifest.get("source_authorization", {})
    if not (
        source_authorization.get("all_model_and_cache_paths_resolved_below_root")
        and source_authorization.get("all_formal_admission_evidence_paths_resolved_below_root")
        and source_authorization.get("symlinks_rejected")
    ):
        raise RuntimeError("source-root authorization claim drift")

    rows = manifest.get("files", [])
    expected = {str(row["path"]): row for row in rows}
    if not expected or len(expected) != len(rows):
        raise RuntimeError("empty/duplicate file inventory")
    if len(rows) != int(manifest.get("payload_file_count", -1)):
        raise RuntimeError("payload file-count field drift")
    actual = []
    symlinks = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file() and relative != "manifest.json":
            actual.append(relative)
    if symlinks:
        raise RuntimeError(f"symlinks are forbidden: {symlinks}")
    if set(actual) != set(expected):
        raise RuntimeError(
            f"payload path-set drift: missing={sorted(set(expected)-set(actual))}, "
            f"unexpected={sorted(set(actual)-set(expected))}"
        )
    total = 0
    for relative in actual:
        path = resolve(root, relative)
        observed = (path.stat().st_size, sha256(path))
        if observed != pair(expected[relative]):
            raise RuntimeError(f"payload identity drift: {relative}: {observed}")
        total += observed[0]
    if total != int(manifest.get("payload_total_bytes", -1)):
        raise RuntimeError("payload byte total drift")

    execution = manifest.get("execution", {})
    kind = execution.get("kind")
    references = []
    if kind == "segment25":
        if boundaries.get("native_int8_kernel") != "unverified":
            raise RuntimeError("mixed segment25 bundle must retain unverified native-kernel status")
        segments = execution.get("segments", [])
        if len(segments) != 25 or [row.get("index") for row in segments] != list(range(25)):
            raise RuntimeError("segment25 order/count drift")
        if tuple(execution.get("retain_encoder_outputs_after_segments", ())) != (5, 11, 17, 23):
            raise RuntimeError("retained-output boundary drift")
        candidate_id = execution.get("source_candidate_id")
        if candidate_id not in EXPECTED_VARIANTS:
            raise RuntimeError("mixed source candidate ID drift")
        fp16_indices = tuple(
            row["index"] for row in segments[:24] if row.get("precision") == "fp16"
        )
        if fp16_indices != EXPECTED_VARIANTS[candidate_id]:
            raise RuntimeError("mixed bundle FP16 block mapping drift")
        if segments[24].get("precision") != "fp32_compat_barrier":
            raise RuntimeError("mixed bundle repaired-head precision drift")
        if segments[24].get("source_lineage") != EXPECTED_HEAD_LINEAGES[candidate_id]:
            raise RuntimeError("mixed bundle repaired-head lineage drift")
        for segment in segments:
            references.extend((segment["model"], segment["cache"]))
        references.extend((execution["source_manifest"], execution["task_admission"]))
        evidence_by_name = {
            row.get("name"): row for row in manifest.get("evidence", [])
        }
        required_task_evidence = {"task_admission"}
        for trial_index in (1, 2, 3):
            required_task_evidence.update(
                {
                    f"task_trial_{trial_index}_result",
                    f"task_trial_{trial_index}_predictions_and_targets",
                    f"task_trial_{trial_index}_per_sample_metrics",
                }
            )
        if not required_task_evidence.issubset(evidence_by_name):
            raise RuntimeError("mixed raw three-run task evidence is incomplete")
        references.extend(evidence_by_name[name] for name in required_task_evidence)
    elif kind == "fp16_full":
        if boundaries.get("native_int8_kernel") != "not_applicable":
            raise RuntimeError("FP16 bundle native-INT8-kernel boundary drift")
        if boundaries.get("fp16_full_precision_semantics") != "verified_frozen_artifact" or boundaries.get(
            "fp16_full_predeployment_state"
        ) != "strict_logits_failed_task_accuracy_passed_runtime_acceptance_pending":
            raise RuntimeError("FP16 precision/predeployment claim boundary drift")
        if pair(execution.get("model", {})) != FP16_MODEL_PAIR or pair(
            execution.get("cache", {})
        ) != FP16_CACHE_PAIR:
            raise RuntimeError("FP16 frozen model/cache identity drift")
        semantics = execution.get("precision_semantics", {})
        if semantics.get("state") != (
            "verified_frozen_fp16_internal_with_fp32_layernorm_statistic_islands"
        ):
            raise RuntimeError("FP16 internal precision semantics are not verified")
        if int(semantics.get("initializer_dtype_counts", {}).get("FLOAT16", 0)) <= 0 or int(
            semantics.get("cast_target_dtype_counts", {}).get("FLOAT16", 0)
        ) <= 0 or int(semantics.get("cast_target_dtype_counts", {}).get("FLOAT", 0)) <= 0:
            raise RuntimeError("FP16 internal semantic proof counts drift")
        if semantics.get("compatibility_op_counts", {}).get("LayerNormalization") != 0 or semantics.get(
            "compatibility_op_counts", {}
        ).get("ConvTranspose") != 0 or int(
            semantics.get("compatibility_op_counts", {}).get("MaxPool", 0)
        ) <= 0:
            raise RuntimeError("FP16 compatibility rewrite evidence drift")
        if execution.get("admission_state") != (
            "task_admitted_runtime_deployment_acceptance_pending"
        ):
            raise RuntimeError("FP16 admission state drift")
        for key, expected_pair in FP16_EVIDENCE_PAIRS.items():
            if pair(execution.get(key, {})) != expected_pair:
                raise RuntimeError(f"FP16 {key} evidence identity drift")
        if manifest.get("capacity", {}).get("precision_segment_counts") != {
            "fp16_full_verified": 1
        }:
            raise RuntimeError("FP16 verified precision count drift")
        references.extend((execution["model"], execution["cache"]))
        references.extend(execution[key] for key in FP16_EVIDENCE_PAIRS)
    else:
        raise RuntimeError(f"unsupported execution kind: {kind!r}")
    fixed = manifest.get("fixed_sample", {})
    references.extend((fixed["input"], fixed["expected_output"]))
    references.append(manifest["runtime_contract"]["origin_runtime_fingerprint"])
    shim = manifest["runtime_contract"]["static_lsmod"]
    if shim.get("path") != "tools/bin/lsmod" or pair(shim) != (LSMOD_SIZE, LSMOD_SHA256):
        raise RuntimeError("locked static lsmod contract drift")
    if not os.access(resolve(root, shim["path"]), os.X_OK):
        raise RuntimeError("locked static lsmod is not executable")
    references.append(shim)
    for reference in references:
        relative = str(reference["path"])
        if relative not in expected or pair(reference) != pair(expected[relative]):
            raise RuntimeError(f"referenced payload is not locked by inventory: {relative}")

    result = {
        "schema": "phase11_k100_bundle_static_verification_v1",
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(root),
        "bundle_id": manifest["bundle_id"],
        "execution_kind": kind,
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_digest,
        },
        "payload_file_count": len(actual),
        "payload_total_bytes": total,
        "claims": {
            "static_bundle_identity_verified": True,
            "runtime_validated": False,
            "cache_portability_verified": False,
            "deployment_ready": False,
        },
    }
    if args.receipt:
        receipt = args.receipt.resolve(strict=False)
        if receipt.exists() or contained(root, receipt):
            raise RuntimeError("receipt must be new and outside the immutable bundle")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
