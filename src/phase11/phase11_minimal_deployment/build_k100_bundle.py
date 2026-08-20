#!/usr/bin/env python3
"""Stage a candidate-agnostic minimal K100 deployment bundle and lock every byte."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "phase11_k100_minimal_deployment_bundle_v1"
RUNTIME_SCHEMA = "phase11_k100_deployment_runtime_fingerprint_v1"
MIXED_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
MIXED_FINAL_STAGE = "phase11_sensitivity_mixed_precision_candidate_cache_final_v1"
TASK_ADMISSION_SCHEMA = "phase11_mixed_precision_test90_three_run_summary_v1"
MIXED_RUN_SCHEMA = "phase11_mixed_precision_test90_v1"
MIXED_AGGREGATOR = {
    "size_bytes": 9_628,
    "sha256": "717aa6c5e10b889c082acb4626105ad708c9006acf70380793f0b93c71fbd159",
}
MIXED_COMMON = {
    "size_bytes": 23_030,
    "sha256": "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8",
}
MIXED_TEST90 = {
    "size_bytes": 17_348,
    "sha256": "8bc54ffe04719cce53d22648b0ffc2fd1e8daaac8e2e4acb61c12910c61d4091",
}

# This is the only FP16-full compatibility artifact currently admitted for the
# formal deployment track.  A different FP16 model must be admitted as a new
# protocol/version instead of being relabelled through this generic builder.
FP16_MODEL = {
    "size_bytes": 638_970_735,
    "sha256": "8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7",
}
FP16_CACHE = {
    "size_bytes": 708_193_607,
    "sha256": "9fe8985dc8ce4a3cf9d829ad66a91de41bb67b22e01b8181c43bf15c74f7dd0a",
}
FP16_BUILD_REPORT = {
    "size_bytes": 15_591,
    "sha256": "93293dc0909f5158f983ce9dc5b7953cf4956a116f73e54aab7e86e2a7cd3fba",
}
FP16_SINGLE_ADMISSION = {
    "size_bytes": 3_316,
    "sha256": "5e93e828ad44ca703c1eac5b08fd2c332d70002cec13c6d3c3c1b688f5a35a89",
}
FP16_TASK_ADMISSION = {
    "size_bytes": 3_720,
    "sha256": "9b9ef29eea9c953fdfbd0e1bbf85cff1f689e2bef14b931a953f235c7444da5c",
}
FP16_FRESH_TASK_RESULT = {
    "size_bytes": 6_042,
    "sha256": "c5a55d19320a1a7b52e2fd8add2192b3015c258420b5cc5996e9ca1953a32397",
}
FP16_FROZEN_TASK_RESULT = {
    "size_bytes": 5_993,
    "sha256": "48de1f62acf10116254574851d5974528c4ce3df7363504b2fe59424de3a72dc",
}
FP16_VARIANT = (
    "fp16_internal_with_fp32_layernorm_statistic_islands_and_single_"
    "fpn4_maxpool_barrier"
)
FP16_SOURCE_SHA256 = "10df534d4dbaabf8336e4195a60d2ebd719b14c7a3e34362d48a0609425e6dbd"
LSMOD_SIZE = 819_664
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"
EXPECTED_VARIANTS = {
    "M0": (),
    "M1": (0,),
    "M2": (0, 14),
    "M3": (0, 14, 17),
    "M4": (0, 14, 17, 18),
    "M5": (0, 14, 15, 16, 17, 18, 19, 20, 21),
}
HEAD_LINEAGE = "validated_int8_backbone_fp32_head_with_fpn4_maxpool_graph_output_barrier"
EXPECTED_HEAD_LINEAGES = {
    **{candidate: HEAD_LINEAGE for candidate in ("M0", "M1", "M2", "M3", "M4")},
    "M5": "frozen_m4_validated_fp32_head_with_fpn4_maxpool_graph_output_barrier",
}
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
TOOLS = (
    "infer_k100.py",
    "verify_k100_bundle.py",
    "capture_k100_runtime_fingerprint.py",
    "acceptance_k100.py",
    "run_cold_start_5.sh",
    "run_stability_60min.sh",
    "run_recovery_3.sh",
    "run_cross_node_smoke.sh",
    "launch_cross_node_smoke_container.py",
    "README.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"size_bytes": path.stat().st_size, "sha256": sha256(path)}


def check_expected(path: Path, expected: dict | None) -> None:
    if expected is None:
        return
    observed = identity(path)
    wanted = {"size_bytes": int(expected["size_bytes"]), "sha256": str(expected["sha256"])}
    if observed != wanted:
        raise RuntimeError(f"source identity drift: {path}: {observed}; expected={wanted}")


def contained(root: Path, child: Path) -> bool:
    root = root.resolve(strict=True)
    child = child.resolve(strict=True)
    return child == root or root in child.parents


def require_source_artifact(path: Path, source_root: Path, role: str) -> Path:
    unresolved = path
    path = path.resolve(strict=True)
    if not contained(source_root, path):
        raise RuntimeError(f"{role} escapes authorized --source-root: {unresolved} -> {path}")
    if unresolved.is_symlink() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} must be a regular non-symlink file: {unresolved}")
    return path


def source_path(
    manifest_path: Path, value: str | dict, source_root: Path
) -> tuple[Path, dict | None]:
    if isinstance(value, dict):
        relative = value.get("path")
        expected = value if "size_bytes" in value and "sha256" in value else None
    else:
        relative = value
        expected = None
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"source artifact must be a relative path: {relative!r}")
    path = require_source_artifact(
        manifest_path.parent / relative, source_root, "manifest-referenced artifact"
    )
    return path, expected


def artifact_from_record(record: dict, source_root: Path, role: str) -> Path:
    """Resolve an evidence-record path below source_root and verify its identity."""
    if not isinstance(record, dict):
        raise RuntimeError(f"{role} identity record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{role} identity path is missing")
    path = require_source_artifact(Path(raw_path), source_root, role)
    check_expected(path, record)
    return path


def validate_mixed_final_manifest(path: Path, payload: dict) -> tuple[str, list[dict]]:
    if payload.get("schema") != MIXED_SCHEMA:
        raise RuntimeError(f"mixed candidate schema drift: {payload.get('schema')!r}")
    if payload.get("status") != "cache_finalized_static_pass":
        raise RuntimeError(f"mixed candidate is not cache-finalized: {payload.get('status')!r}")
    if payload.get("manifest_stage") != MIXED_FINAL_STAGE:
        raise RuntimeError("mixed candidate final-stage marker drift")
    candidate_id = str(payload.get("candidate_id", ""))
    if candidate_id not in EXPECTED_VARIANTS:
        raise RuntimeError(f"unsupported mixed candidate ID: {candidate_id!r}")
    if int(payload.get("segment_count", -1)) != 25:
        raise RuntimeError("mixed candidate segment_count must be 25")
    if tuple(payload.get("retain_encoder_outputs_after_segments", ())) != (5, 11, 17, 23):
        raise RuntimeError("mixed candidate retained-output contract drift")
    if payload.get("head_precision") != "fp32_compat_barrier":
        raise RuntimeError("mixed candidate repaired FP32 head contract drift")
    fp16_blocks = tuple(int(value) for value in payload.get("fp16_backbone_blocks", ()))
    if fp16_blocks != EXPECTED_VARIANTS[candidate_id]:
        raise RuntimeError(
            f"{candidate_id} FP16 block mapping drift: {fp16_blocks}; "
            f"expected={EXPECTED_VARIANTS[candidate_id]}"
        )
    int8_blocks = tuple(int(value) for value in payload.get("int8_backbone_blocks", ()))
    expected_int8 = tuple(index for index in range(24) if index not in fp16_blocks)
    if int8_blocks != expected_int8:
        raise RuntimeError(f"{candidate_id} INT8 block mapping drift")
    rows = payload.get("segments", [])
    if len(rows) != 25 or [row.get("index") for row in rows] != list(range(25)):
        raise RuntimeError("mixed candidate segment order/count drift")
    for index, row in enumerate(rows):
        expected_label = (
            f"encoder_block_{index:02d}" if index < 24 else "upernet_decoder_head_fpn4barrier"
        )
        if row.get("label") != expected_label:
            raise RuntimeError(f"mixed segment {index} label drift: {row.get('label')!r}")
        expected_precision = (
            "fp32_compat_barrier"
            if index == 24
            else "fp16"
            if index in fp16_blocks
            else "int8_qdq"
        )
        if row.get("precision") != expected_precision:
            raise RuntimeError(f"mixed segment {index} precision drift")
        if row.get("model") is None or row.get("model_identity") is None:
            raise RuntimeError(f"mixed segment {index} model lineage is incomplete")
        if row.get("cache") is None or row.get("cache_identity") is None:
            raise RuntimeError(f"mixed segment {index} cache lineage is incomplete")
    head = rows[24]
    if head.get("source_lineage") != EXPECTED_HEAD_LINEAGES[candidate_id]:
        raise RuntimeError("mixed repaired-head lineage drift")
    claims = payload.get("claims", {})
    if claims.get("all_required_mxr_caches_present") is not True:
        raise RuntimeError("mixed candidate does not claim all required caches present")
    if claims.get("fp16_cache_compilation_and_placement_passed") is not True:
        raise RuntimeError("mixed candidate FP16 cache placement prerequisite is not passed")
    return candidate_id, rows


def validate_task_admission(
    path: Path,
    payload: dict,
    candidate_id: str,
    manifest_path: Path,
    source_root: Path,
) -> list[tuple[str, Path, dict]]:
    if payload.get("schema") != TASK_ADMISSION_SCHEMA or payload.get("status") != "passed":
        raise RuntimeError("mixed fixed-90 three-run task admission schema/status drift")
    claims = payload.get("claims", {})
    if claims.get("task_equivalence_repeatability_confirmed") is not True or claims.get(
        "task_equivalence_performance_track_may_proceed"
    ) is not True:
        raise RuntimeError("mixed fixed-90 task/repeatability admission is not passed")
    runs = payload.get("runs", [])
    if len(runs) != 3 or [row.get("trial_index") for row in runs] != [1, 2, 3]:
        raise RuntimeError("mixed task admission must contain ordered trials 1,2,3")
    lineage = payload.get("candidate_lineage", {})
    if lineage.get("candidate_id") != candidate_id:
        raise RuntimeError("task-admission candidate ID differs from source manifest")
    recorded_manifest = lineage.get("manifest", {})
    observed_manifest = identity(manifest_path)
    if (
        int(recorded_manifest.get("size_bytes", -1)),
        str(recorded_manifest.get("sha256", "")),
    ) != (observed_manifest["size_bytes"], observed_manifest["sha256"]):
        raise RuntimeError("task-admission manifest identity differs from source manifest")
    gates = payload.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise RuntimeError(f"task-admission gates are incomplete/failed: {path}")

    identities = payload.get("identities", {})
    if {
        key: {
            "size_bytes": int(identities.get(key, {}).get("size_bytes", -1)),
            "sha256": str(identities.get(key, {}).get("sha256", "")),
        }
        for key in ("aggregate_script", "common_module")
    } != {"aggregate_script": MIXED_AGGREGATOR, "common_module": MIXED_COMMON}:
        raise RuntimeError("mixed task-admission generator identity drift")

    evidence: list[tuple[str, Path, dict]] = []
    observed_result_hashes = set()
    for trial_index, summary_run in enumerate(runs, start=1):
        artifacts = summary_run.get("artifacts", {})
        result_record = artifacts.get("result", {})
        result_path = artifact_from_record(
            result_record, source_root, f"mixed trial {trial_index} result"
        )
        observed_result_hashes.add(result_record.get("sha256"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("schema") != MIXED_RUN_SCHEMA or result.get("status") != "diagnostic_completed":
            raise RuntimeError(f"mixed trial {trial_index} result schema/status drift")
        result_lineage = result.get("candidate_lineage", {})
        if result_lineage.get("candidate_id") != candidate_id:
            raise RuntimeError(f"mixed trial {trial_index} candidate ID drift")
        result_manifest = result_lineage.get("manifest", {})
        if (
            int(result_manifest.get("size_bytes", -1)),
            str(result_manifest.get("sha256", "")),
        ) != (observed_manifest["size_bytes"], observed_manifest["sha256"]):
            raise RuntimeError(f"mixed trial {trial_index} manifest identity drift")
        result_identities = result.get("identities", {})
        if (
            int(result_identities.get("evaluation_script", {}).get("size_bytes", -1)),
            str(result_identities.get("evaluation_script", {}).get("sha256", "")),
        ) != (MIXED_TEST90["size_bytes"], MIXED_TEST90["sha256"]):
            raise RuntimeError(f"mixed trial {trial_index} evaluator identity drift")
        if (
            int(result_identities.get("common_module", {}).get("size_bytes", -1)),
            str(result_identities.get("common_module", {}).get("sha256", "")),
        ) != (MIXED_COMMON["size_bytes"], MIXED_COMMON["sha256"]):
            raise RuntimeError(f"mixed trial {trial_index} common-module identity drift")
        if result.get("task_equivalence_admission_passed") is not True:
            raise RuntimeError(f"mixed trial {trial_index} task gate is not passed")
        if not result.get("task_gates") or not all(
            value is True for value in result["task_gates"].values()
        ):
            raise RuntimeError(f"mixed trial {trial_index} contains a failed task gate")
        claims = result.get("claims", {})
        if claims.get("task_equivalence_admission_passed") is not True or claims.get(
            "historical_m0_strict_failure_overridden"
        ) is True:
            raise RuntimeError(f"mixed trial {trial_index} claim boundary drift")
        dataset = result.get("dataset", {})
        if int(dataset.get("samples", -1)) != 90 or int(dataset.get("valid_pixels", -1)) != 3_927_398:
            raise RuntimeError(f"mixed trial {trial_index} fixed dataset contract drift")
        if int(result.get("nonfinite_output_count", -1)) != 0:
            raise RuntimeError(f"mixed trial {trial_index} has non-finite output")
        if summary_run.get("metrics") != result.get("metrics") or summary_run.get(
            "deltas_vs_frozen_fp32_pp"
        ) != result.get("deltas_vs_frozen_fp32_pp") or summary_run.get(
            "prediction_agreement_vs_fp32"
        ) != result.get("valid_prediction_agreement_vs_frozen_fp32"):
            raise RuntimeError(f"mixed trial {trial_index} summary/result value drift")

        evidence.append((f"task_trial_{trial_index}_result", result_path, result_record))
        for key in ("predictions_and_targets", "per_sample_metrics"):
            summary_record = artifacts.get(key, {})
            raw_record = result.get("artifacts", {}).get(key, {})
            if (
                int(summary_record.get("size_bytes", -1)),
                str(summary_record.get("sha256", "")),
            ) != (
                int(raw_record.get("size_bytes", -2)),
                str(raw_record.get("sha256", "drift")),
            ):
                raise RuntimeError(f"mixed trial {trial_index} {key} identity drift")
            artifact_path = artifact_from_record(
                summary_record, source_root, f"mixed trial {trial_index} {key}"
            )
            evidence.append((f"task_trial_{trial_index}_{key}", artifact_path, summary_record))
    if len(observed_result_hashes) != 3:
        raise RuntimeError("mixed task-admission results are not three independent artifacts")
    return evidence


def expected_pair(row: dict) -> tuple[int, str]:
    return int(row.get("size_bytes", -1)), str(row.get("sha256", ""))


def validate_fp16_build_report(payload: dict) -> None:
    if payload.get("status") != "created_static_pass" or payload.get("variant") != FP16_VARIANT:
        raise RuntimeError("FP16 frozen build-report schema/status/variant drift")
    source = payload.get("source", {})
    if int(source.get("size", -1)) != 638_894_819 or source.get("sha256") != FP16_SOURCE_SHA256:
        raise RuntimeError("FP16 build-report frozen source identity drift")
    candidate = payload.get("candidate", {})
    if (
        int(candidate.get("size", -1)),
        str(candidate.get("sha256", "")),
    ) != (FP16_MODEL["size_bytes"], FP16_MODEL["sha256"]):
        raise RuntimeError("FP16 build-report candidate identity drift")
    if candidate.get("outputs") != [
        "logits",
        "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0",
    ]:
        raise RuntimeError("FP16 repaired-head output lineage drift")
    if payload.get("layernorm", {}).get("count") != 49 or payload.get("layernorm", {}).get(
        "remaining"
    ) != 0:
        raise RuntimeError("FP16 LayerNorm FP32-island rewrite lineage drift")
    if payload.get("convtranspose", {}).get("count") != 3 or payload.get(
        "convtranspose", {}
    ).get("remaining") != 0:
        raise RuntimeError("FP16 ConvTranspose compatibility rewrite lineage drift")
    if payload.get("claims", {}).get("static_graph_valid") is not True:
        raise RuntimeError("FP16 build report did not pass static graph validation")


def validate_fp16_single_admission(payload: dict, official_image_id: str) -> None:
    # The historical single-sample result is deliberately failed on strict
    # numeric gates.  It is retained here only as model/cache pairing and EP
    # placement evidence and must never be rewritten as a strict pass.
    if payload.get("status") != "failed":
        raise RuntimeError("FP16 frozen single-result status drift")
    inputs = payload.get("inputs", {})
    if expected_pair(inputs.get("candidate", {})) != (
        FP16_MODEL["size_bytes"],
        FP16_MODEL["sha256"],
    ) or expected_pair(inputs.get("build_report", {})) != (
        FP16_BUILD_REPORT["size_bytes"],
        FP16_BUILD_REPORT["sha256"],
    ):
        raise RuntimeError("FP16 single-result model/build lineage drift")
    if expected_pair(payload.get("compiled_cache", {})) != (
        FP16_CACHE["size_bytes"],
        FP16_CACHE["sha256"],
    ):
        raise RuntimeError("FP16 single-result model/cache pairing drift")
    runtime = payload.get("runtime", {})
    if runtime.get("onnxruntime") != "1.19.2" or runtime.get(
        "container_image_id"
    ) != official_image_id:
        raise RuntimeError("FP16 single-result runtime/image lineage drift")
    counts = payload.get("profile", {}).get("provider_event_counts", {})
    if int(counts.get("MIGraphXExecutionProvider", 0)) <= 0 or int(
        counts.get("CPUExecutionProvider", 0)
    ) != 0:
        raise RuntimeError("FP16 single-result provider placement drift")
    claims = payload.get("claims", {})
    if claims.get("strict_migraphx_single_sample_placement") is not True or claims.get(
        "formal_exact_transform_equivalence"
    ) is not False:
        raise RuntimeError("FP16 single-result strict/placement claim boundary drift")
    if payload.get("formal_exact_transform_gate", {}).get("passed") is not False:
        raise RuntimeError("FP16 strict numeric failure was not preserved")


def validate_fp16_fresh_task_result(payload: dict) -> None:
    if payload.get("status") != "diagnostic_completed":
        raise RuntimeError("FP16 fresh fixed-90 result status drift")
    lineage = payload.get("lineage", {})
    if expected_pair(lineage.get("candidate", {})) != (
        FP16_MODEL["size_bytes"],
        FP16_MODEL["sha256"],
    ) or expected_pair(lineage.get("compiled_cache", {})) != (
        FP16_CACHE["size_bytes"],
        FP16_CACHE["sha256"],
    ) or expected_pair(lineage.get("failed_single_gate", {})) != (
        FP16_SINGLE_ADMISSION["size_bytes"],
        FP16_SINGLE_ADMISSION["sha256"],
    ):
        raise RuntimeError("FP16 fresh fixed-90 model/cache/single lineage drift")
    dataset = payload.get("dataset", {})
    if int(dataset.get("sample_count", -1)) != 90 or int(dataset.get("valid_pixels", -1)) != 3_927_398:
        raise RuntimeError("FP16 fresh fixed-90 dataset contract drift")
    task = payload.get("descriptive_task_gates", {})
    if task.get("passed") is not True or not task.get("gates") or not all(
        value is True for value in task["gates"].values()
    ):
        raise RuntimeError("FP16 fresh fixed-90 task admission failed")
    counts = payload.get("placement", {}).get("provider_event_counts", {})
    if int(counts.get("MIGraphXExecutionProvider", -1)) != 90 or int(
        counts.get("CPUExecutionProvider", 0)
    ) != 0:
        raise RuntimeError("FP16 fresh fixed-90 provider placement drift")
    if payload.get("claims", {}).get("formal_fp16_deployment_passed") is not False:
        raise RuntimeError("FP16 pre-deployment evidence boundary was rewritten")


def validate_fp16_task_admission(payload: dict) -> None:
    if payload.get("status") != "task_level_confirmation_passed":
        raise RuntimeError("FP16 task-confirmation status drift")
    protocol = payload.get("protocol", {})
    if not protocol or not all(value is True for value in protocol.values()):
        raise RuntimeError("FP16 task-confirmation protocol drift")
    frozen = payload.get("frozen_evidence", {})
    if expected_pair(frozen.get("diagnostic_result", {})) != (
        FP16_FROZEN_TASK_RESULT["size_bytes"],
        FP16_FROZEN_TASK_RESULT["sha256"],
    ) or expected_pair(frozen.get("failed_single_gate", {})) != (
        FP16_SINGLE_ADMISSION["size_bytes"],
        FP16_SINGLE_ADMISSION["sha256"],
    ):
        raise RuntimeError("FP16 task-confirmation frozen evidence lineage drift")
    if expected_pair(payload.get("fresh_evidence", {}).get("result", {})) != (
        FP16_FRESH_TASK_RESULT["size_bytes"],
        FP16_FRESH_TASK_RESULT["sha256"],
    ):
        raise RuntimeError("FP16 task-confirmation fresh-result lineage drift")
    gates = payload.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise RuntimeError("FP16 task-confirmation gates are incomplete/failed")
    claims = payload.get("claims", {})
    required = {
        "strict_logits_numeric_equivalence": False,
        "task_level_accuracy_confirmed": True,
        "compatibility_variant_only": True,
        "paired_performance_experiment_allowed": True,
        "performance_result_available": False,
        "deployment_complete": False,
    }
    if any(claims.get(key) is not value for key, value in required.items()):
        raise RuntimeError("FP16 task-confirmation claim boundary drift")
    if int(payload.get("metrics", {}).get("valid_pixels", -1)) != 3_927_398:
        raise RuntimeError("FP16 task-confirmation valid-pixel contract drift")
    counts = payload.get("placement", {}).get("provider_event_counts", {})
    if int(counts.get("MIGraphXExecutionProvider", -1)) != 90 or int(
        counts.get("CPUExecutionProvider", 0)
    ) != 0:
        raise RuntimeError("FP16 task-confirmation provider placement drift")


def copy_locked(source: Path, root: Path, relative: str, expected: dict | None = None) -> dict:
    source = source.resolve(strict=True)
    check_expected(source, expected)
    target = (root / relative).resolve(strict=False)
    if root not in target.parents:
        raise RuntimeError(f"target escapes bundle root: {relative}")
    if target.exists():
        raise RuntimeError(f"duplicate bundle target: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    row = {"path": relative.replace("\\", "/"), **identity(target)}
    check_expected(target, expected)
    return row


def normalize_contract(value: Any, context: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty list")
    result = []
    for item in value:
        if isinstance(item, str):
            result.append({"name": item})
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            result.append({key: item[key] for key in ("name", "elem_type", "shape") if key in item})
        else:
            raise RuntimeError(f"invalid {context}: {item!r}")
    return result


def inspect_segment_onnx(path: Path, declared_inputs: list[dict], declared_outputs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate and canonicalize all external segment boundaries as static float32."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    graph_inputs = {item.name: item for item in model.graph.input}
    graph_outputs = {item.name: item for item in model.graph.output}
    if set(graph_inputs) != {row["name"] for row in declared_inputs}:
        raise RuntimeError(f"declared ONNX inputs drift for {path}")
    if set(graph_outputs) != {row["name"] for row in declared_outputs}:
        raise RuntimeError(f"declared ONNX outputs drift for {path}")

    def canonical(rows: list[dict], actual: dict) -> list[dict]:
        result = []
        by_name = {row["name"]: row for row in rows}
        for name in by_name:
            info = actual[name]
            if info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                raise RuntimeError(f"deployment boundary is not float32: {path}/{name}")
            shape = []
            for dim in info.type.tensor_type.shape.dim:
                if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
                    raise RuntimeError(f"deployment boundary is not positive/static: {path}/{name}")
                shape.append(int(dim.dim_value))
            declared = by_name[name]
            if "elem_type" in declared and int(declared["elem_type"]) != onnx.TensorProto.FLOAT:
                raise RuntimeError(f"manifest boundary dtype drift: {path}/{name}")
            if "shape" in declared and list(declared["shape"]) != shape:
                raise RuntimeError(f"manifest boundary shape drift: {path}/{name}")
            result.append({"name": name, "elem_type": int(onnx.TensorProto.FLOAT), "shape": shape})
        return result

    return canonical(declared_inputs, graph_inputs), canonical(declared_outputs, graph_outputs)


def load_fixed_input(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as pack:
            if pack.files != ["input"]:
                raise RuntimeError("fixed NPZ must contain only an 'input' array")
            array = pack["input"]
    else:
        raise RuntimeError("fixed input must be .npy or .npz")
    if array.dtype != np.float32 or tuple(array.shape) != (1, 6, 224, 224):
        raise RuntimeError(f"fixed input contract failure: dtype={array.dtype}, shape={array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("fixed input contains NaN/Inf")
    return np.ascontiguousarray(array)


def load_expected(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() != ".npz":
        raise RuntimeError("expected output must be .npz")
    with np.load(path, allow_pickle=False) as pack:
        if set(pack.files) != {"logits", "prediction"}:
            raise RuntimeError("expected output must contain exactly logits and prediction")
        logits = pack["logits"]
        prediction = pack["prediction"]
    if logits.dtype != np.float32 or tuple(logits.shape) != (1, 2, 224, 224):
        raise RuntimeError("expected logits contract failure")
    if prediction.dtype != np.uint8 or tuple(prediction.shape) != (1, 224, 224):
        raise RuntimeError("expected prediction contract failure")
    if not np.isfinite(logits).all():
        raise RuntimeError("expected logits contain NaN/Inf")
    if not np.array_equal(prediction, np.argmax(logits, axis=1).astype(np.uint8)):
        raise RuntimeError("expected prediction is not argmax(expected logits)")
    return np.ascontiguousarray(logits), np.ascontiguousarray(prediction)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inspect_full_onnx(path: Path) -> tuple[str, str, list[str], dict]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise RuntimeError("FP16 full model must have exactly one graph input")
    output = next((item for item in model.graph.output if item.name == "logits"), None)
    if output is None:
        raise RuntimeError("FP16 full model must expose a graph output named logits")
    input_info = model.graph.input[0]
    if input_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("FP16 full external input must be float32")
    if output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("FP16 full external logits must be float32")

    def shape(info) -> list[int]:
        result = []
        for dim in info.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                raise RuntimeError("deployment ONNX boundary shapes must be static")
            result.append(int(dim.dim_value))
        return result

    if shape(input_info) != [1, 6, 224, 224] or shape(output) != [1, 2, 224, 224]:
        raise RuntimeError("FP16 full ONNX external shape contract drift")
    ignored_outputs = [item.name for item in model.graph.output if item.name != output.name]
    initializer_dtype_counts: dict[str, int] = {}
    for initializer in model.graph.initializer:
        key = onnx.TensorProto.DataType.Name(initializer.data_type)
        initializer_dtype_counts[key] = initializer_dtype_counts.get(key, 0) + 1
    cast_to_counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        attribute = next((item for item in node.attribute if item.name == "to"), None)
        if attribute is None:
            raise RuntimeError("FP16 candidate contains a Cast without a target dtype")
        key = onnx.TensorProto.DataType.Name(int(onnx.helper.get_attribute_value(attribute)))
        cast_to_counts[key] = cast_to_counts.get(key, 0) + 1
    op_counts = {
        op_type: sum(node.op_type == op_type for node in model.graph.node)
        for op_type in ("LayerNormalization", "ConvTranspose", "MaxPool")
    }
    if initializer_dtype_counts.get("FLOAT16", 0) <= 0:
        raise RuntimeError("frozen FP16 model has no internal FLOAT16 initializer evidence")
    if cast_to_counts.get("FLOAT16", 0) <= 0 or cast_to_counts.get("FLOAT", 0) <= 0:
        raise RuntimeError("frozen FP16 model lacks FP16/FP32-island Cast evidence")
    if op_counts["LayerNormalization"] != 0 or op_counts["ConvTranspose"] != 0:
        raise RuntimeError("frozen FP16 model compatibility rewrites are incomplete")
    if op_counts["MaxPool"] <= 0:
        raise RuntimeError("frozen FP16 repaired-head MaxPool barrier is absent")
    semantics = {
        "state": "verified_frozen_fp16_internal_with_fp32_layernorm_statistic_islands",
        "model_identity": FP16_MODEL,
        "initializer_dtype_counts": dict(sorted(initializer_dtype_counts.items())),
        "cast_target_dtype_counts": dict(sorted(cast_to_counts.items())),
        "compatibility_op_counts": op_counts,
        "external_input_and_logits_dtype": "float32",
    }
    return input_info.name, output.name, ignored_outputs, semantics


def evidence_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("evidence must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not SAFE_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(f"unsafe evidence name: {name!r}")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("segment25", "fp16_full"), required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-image-id", required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--lsmod-shim", type=Path, required=True)
    parser.add_argument("--fixed-input", type=Path, required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--single-admission", type=Path)
    parser.add_argument("--task-admission", type=Path)
    parser.add_argument("--fp16-task-result", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--evidence", type=evidence_argument, action="append", default=[])
    args = parser.parse_args()
    if not SAFE_NAME.fullmatch(args.bundle_id):
        raise RuntimeError("bundle ID may contain only A-Z, a-z, 0-9, dot, underscore and dash")
    if not IMAGE_PATTERN.fullmatch(args.official_image_id):
        raise RuntimeError("official image ID must be a full sha256 digest")
    source_root = args.source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise RuntimeError("--source-root must be an existing directory")
    if source_root.parent == source_root:
        raise RuntimeError("filesystem root is too broad for --source-root authorization")
    shim = args.lsmod_shim.resolve(strict=True)
    check_expected(shim, {"size_bytes": LSMOD_SIZE, "sha256": LSMOD_SHA256})
    final_output = args.output_dir.resolve(strict=False)
    if final_output.exists():
        raise RuntimeError(f"refusing to overwrite bundle directory: {final_output}")
    output = final_output.with_name(final_output.name + f".partial-{os.getpid()}")
    if output.exists():
        raise RuntimeError(f"stale partial bundle must be inspected/removed explicitly: {output}")
    output.mkdir(parents=True, exist_ok=False)

    runtime = json.loads(args.runtime_fingerprint.resolve(strict=True).read_text(encoding="utf-8"))
    if runtime.get("schema") != RUNTIME_SCHEMA or runtime.get("status") != "captured":
        raise RuntimeError("runtime fingerprint schema/status drift")
    portable = runtime.get("portable_fields", {})
    if portable.get("official_image_id") != args.official_image_id:
        raise RuntimeError("runtime fingerprint official image ID drift")
    if portable.get("onnxruntime") != "1.19.2":
        raise RuntimeError("runtime fingerprint ONNX Runtime version drift")
    observation = runtime.get("observation", {})
    if observation.get("image_identity_attested") is not True or observation.get(
        "attested_image_id"
    ) != args.official_image_id:
        raise RuntimeError("origin runtime fingerprint image identity is not attested")
    if portable.get("static_lsmod") != {
        "size_bytes": LSMOD_SIZE,
        "sha256": LSMOD_SHA256,
    }:
        raise RuntimeError("runtime fingerprint static lsmod contract drift")
    runtime_shim = runtime.get("static_lsmod", {})
    if (
        int(runtime_shim.get("size_bytes", -1)),
        str(runtime_shim.get("sha256", "")),
    ) != (LSMOD_SIZE, LSMOD_SHA256):
        raise RuntimeError("runtime fingerprint did not use the locked static lsmod")

    tool_root = Path(__file__).resolve().parent
    expected_fingerprint_generator = identity(
        tool_root / "capture_k100_runtime_fingerprint.py"
    )
    recorded_generator = runtime.get("generator", {})
    if (
        int(recorded_generator.get("size_bytes", -1)),
        str(recorded_generator.get("sha256", "")),
    ) != (
        expected_fingerprint_generator["size_bytes"],
        expected_fingerprint_generator["sha256"],
    ):
        raise RuntimeError("runtime fingerprint generator identity drift")
    for name in TOOLS:
        source = (tool_root / name).resolve(strict=True)
        copy_locked(source, output, f"tools/{name}")
    for path in (output / "tools").iterdir():
        if path.suffix in {".py", ".sh"}:
            path.chmod(0o755)
    shim_row = copy_locked(
        shim,
        output,
        "tools/bin/lsmod",
        {"size_bytes": LSMOD_SIZE, "sha256": LSMOD_SHA256},
    )
    (output / "tools/bin/lsmod").chmod(0o755)

    runtime_row = copy_locked(
        args.runtime_fingerprint, output, "runtime/origin_runtime_fingerprint.json"
    )
    fixed_array = load_fixed_input(args.fixed_input.resolve(strict=True))
    expected_logits, expected_prediction = load_expected(args.expected_output.resolve(strict=True))
    fixed_suffix = args.fixed_input.suffix.lower()
    fixed_row = copy_locked(args.fixed_input, output, f"test/fixed_input{fixed_suffix}")
    expected_row = copy_locked(args.expected_output, output, "test/expected_output.npz")

    evidence_rows = []
    evidence_names = set()
    for name, path in args.evidence:
        if name in {
            "task_admission",
            "single_admission",
            "source_candidate_manifest",
            "fp16_task_result",
        } or name.startswith("task_trial_"):
            raise RuntimeError(f"evidence name {name!r} is reserved")
        if name in evidence_names:
            raise RuntimeError(f"duplicate evidence name: {name}")
        evidence_names.add(name)
        source = path.resolve(strict=True)
        row = copy_locked(source, output, f"evidence/{name}{source.suffix.lower()}")
        evidence_rows.append({"name": name, **row})

    if args.kind == "segment25":
        if (
            args.source_manifest is None
            or args.task_admission is None
            or args.single_admission is not None
            or args.fp16_task_result is not None
            or args.model is not None
            or args.cache is not None
        ):
            raise RuntimeError("segment25 requires --source-manifest and --task-admission only")
        source_manifest = require_source_artifact(
            args.source_manifest, source_root, "mixed candidate manifest"
        )
        source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        candidate_id, source_rows = validate_mixed_final_manifest(
            source_manifest, source_payload
        )
        task_admission = require_source_artifact(
            args.task_admission, source_root, "mixed task-admission evidence"
        )
        task_payload = json.loads(task_admission.read_text(encoding="utf-8"))
        raw_task_evidence = validate_task_admission(
            task_admission,
            task_payload,
            candidate_id,
            source_manifest,
            source_root,
        )
        source_manifest_row = copy_locked(
            source_manifest, output, "evidence/source_candidate_manifest.json"
        )
        task_admission_row = copy_locked(
            task_admission, output, "evidence/task_admission.json"
        )
        evidence_rows.append({"name": "task_admission", **task_admission_row})
        for name, source, expected in raw_task_evidence:
            suffix = source.suffix.lower()
            copied = copy_locked(source, output, f"evidence/{name}{suffix}", expected)
            evidence_rows.append({"name": name, **copied})
        segments = []
        for index, row in enumerate(source_rows):
            model_source, model_inline_identity = source_path(
                source_manifest, row["model"], source_root
            )
            cache_source, cache_inline_identity = source_path(
                source_manifest, row["cache"], source_root
            )
            model_expected = row.get("model_identity") or model_inline_identity
            cache_expected = row.get("cache_identity") or cache_inline_identity
            declared_inputs = normalize_contract(row.get("inputs"), f"segment {index} inputs")
            declared_outputs = normalize_contract(row.get("outputs"), f"segment {index} outputs")
            checked_inputs, checked_outputs = inspect_segment_onnx(
                model_source, declared_inputs, declared_outputs
            )
            model_row = copy_locked(
                model_source, output, f"models/segment_{index:02d}.onnx", model_expected
            )
            cache_row = copy_locked(
                cache_source, output, f"caches/segment_{index:02d}.mxr", cache_expected
            )
            segments.append(
                {
                    "index": index,
                    "label": str(row.get("label", f"segment_{index:02d}")),
                    "precision": str(row.get("precision", "unspecified")),
                    "source_lineage": str(row.get("source_lineage", "")),
                    "model": model_row,
                    "cache": cache_row,
                    "inputs": checked_inputs,
                    "outputs": checked_outputs,
                }
            )
        if segments[0]["inputs"] != [
            {"name": segments[0]["inputs"][0]["name"], "elem_type": 1, "shape": [1, 6, 224, 224]}
        ]:
            raise RuntimeError("first segment must have one static float32[1,6,224,224] input")
        for index in range(1, 24):
            if len(segments[index - 1]["outputs"]) != 1 or segments[index]["inputs"] != segments[index - 1]["outputs"]:
                raise RuntimeError(f"encoder intersegment boundary drift between {index-1} and {index}")
        retained_names = {segments[index]["outputs"][0]["name"] for index in (5, 11, 17, 23)}
        if {row["name"] for row in segments[24]["inputs"]} != retained_names:
            raise RuntimeError("head inputs do not match retained encoder outputs 5/11/17/23")
        head_logits = next(
            (row for row in segments[24]["outputs"] if row["name"] == "logits"), None
        )
        if head_logits != {"name": "logits", "elem_type": 1, "shape": [1, 2, 224, 224]}:
            raise RuntimeError("head must expose static float32 logits[1,2,224,224]")
        execution = {
            "kind": "segment25",
            "source_candidate_id": candidate_id,
            "source_manifest": source_manifest_row,
            "task_admission": task_admission_row,
            "segments": segments,
            "retain_encoder_outputs_after_segments": [5, 11, 17, 23],
            "device_intersegment_io": "OrtValue with I/O Binding",
            "concurrently_resident_sessions": 25,
        }
        precision_counts: dict[str, int] = {}
        for row in segments:
            precision_counts[row["precision"]] = precision_counts.get(row["precision"], 0) + 1
        capacity = {
            "onnx_bytes": sum(int(row["model"]["size_bytes"]) for row in segments),
            "mxr_bytes": sum(int(row["cache"]["size_bytes"]) for row in segments),
            "precision_segment_counts": dict(sorted(precision_counts.items())),
        }
        native_kernel_state = "unverified"
    else:
        if (
            args.source_manifest is None
            or args.single_admission is None
            or args.task_admission is None
            or args.fp16_task_result is None
            or args.model is None
            or args.cache is None
        ):
            raise RuntimeError(
                "fp16_full requires --model, --cache, --source-manifest, "
                "--single-admission, --fp16-task-result and --task-admission"
            )
        model = require_source_artifact(args.model, source_root, "FP16 full ONNX")
        cache = require_source_artifact(args.cache, source_root, "FP16 full MXR")
        build_report = require_source_artifact(
            args.source_manifest, source_root, "FP16 frozen build report"
        )
        single_admission = require_source_artifact(
            args.single_admission, source_root, "FP16 frozen single-result evidence"
        )
        task_result = require_source_artifact(
            args.fp16_task_result, source_root, "FP16 fresh fixed-90 result evidence"
        )
        task_admission = require_source_artifact(
            args.task_admission, source_root, "FP16 task-confirmation evidence"
        )
        check_expected(model, FP16_MODEL)
        check_expected(cache, FP16_CACHE)
        check_expected(build_report, FP16_BUILD_REPORT)
        check_expected(single_admission, FP16_SINGLE_ADMISSION)
        check_expected(task_result, FP16_FRESH_TASK_RESULT)
        check_expected(task_admission, FP16_TASK_ADMISSION)
        validate_fp16_build_report(json.loads(build_report.read_text(encoding="utf-8")))
        validate_fp16_single_admission(
            json.loads(single_admission.read_text(encoding="utf-8")),
            args.official_image_id,
        )
        validate_fp16_fresh_task_result(
            json.loads(task_result.read_text(encoding="utf-8"))
        )
        validate_fp16_task_admission(
            json.loads(task_admission.read_text(encoding="utf-8"))
        )
        input_name, output_name, ignored_outputs, fp16_semantics = inspect_full_onnx(model)
        source_manifest_row = copy_locked(
            build_report,
            output,
            "evidence/fp16_frozen_build_report.json",
            FP16_BUILD_REPORT,
        )
        single_admission_row = copy_locked(
            single_admission,
            output,
            "evidence/fp16_single_admission.json",
            FP16_SINGLE_ADMISSION,
        )
        task_result_row = copy_locked(
            task_result,
            output,
            "evidence/fp16_fresh_task_result.json",
            FP16_FRESH_TASK_RESULT,
        )
        task_admission_row = copy_locked(
            task_admission,
            output,
            "evidence/fp16_task_admission.json",
            FP16_TASK_ADMISSION,
        )
        evidence_rows.extend(
            (
                {"name": "source_candidate_manifest", **source_manifest_row},
                {"name": "single_admission", **single_admission_row},
                {"name": "fp16_task_result", **task_result_row},
                {"name": "task_admission", **task_admission_row},
            )
        )
        execution = {
            "kind": "fp16_full",
            "model": copy_locked(model, output, "models/fp16_full.onnx", FP16_MODEL),
            "cache": copy_locked(cache, output, "caches/fp16_full.mxr", FP16_CACHE),
            "source_manifest": source_manifest_row,
            "single_admission": single_admission_row,
            "fresh_task_result": task_result_row,
            "task_admission": task_admission_row,
            "precision_semantics": fp16_semantics,
            "admission_state": "task_admitted_runtime_deployment_acceptance_pending",
            "input_name": input_name,
            "output_name": output_name,
            "ignored_diagnostic_graph_outputs": ignored_outputs,
            "device_io": "OrtValue with I/O Binding",
            "concurrently_resident_sessions": 1,
        }
        capacity = {
            "onnx_bytes": int(execution["model"]["size_bytes"]),
            "mxr_bytes": int(execution["cache"]["size_bytes"]),
            "precision_segment_counts": {"fp16_full_verified": 1},
        }
        native_kernel_state = "not_applicable"

    file_rows = []
    total_bytes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in bundle: {path}")
        relative = str(path.relative_to(output)).replace("\\", "/")
        row = {"path": relative, **identity(path)}
        file_rows.append(row)
        total_bytes += row["size_bytes"]

    manifest = {
        "schema": SCHEMA,
        "status": "static_bundle_identity_locked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_id": args.bundle_id,
        "source_authorization": {
            "root": str(source_root),
            "all_model_and_cache_paths_resolved_below_root": True,
            "all_formal_admission_evidence_paths_resolved_below_root": True,
            "symlinks_rejected": True,
        },
        "input_contract": {
            "npz_key": "input",
            "dtype": "float32",
            "shape": [1, 6, 224, 224],
            "band_order": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "constant_scale_already_applied": 0.0001,
            "external_mean_std_normalization": False,
            "finite_required": True,
            "silent_dtype_or_shape_conversion": False,
        },
        "output_contract": {
            "logits": {"dtype": "float32", "shape": [1, 2, 224, 224]},
            "prediction": {
                "dtype": "uint8",
                "shape": [1, 224, 224],
                "rule": "argmax(logits, axis=1)",
                "class_order": ["background", "water"],
            },
        },
        "runtime_contract": {
            "official_image_id": args.official_image_id,
            "origin_runtime_fingerprint": runtime_row,
            "static_lsmod": shim_row,
            "portable_runtime_signature_sha256": runtime["portable_runtime_signature_sha256"],
            "onnxruntime": "1.19.2",
            "provider": "MIGraphXExecutionProvider",
            "cpu_ep_fallback_disabled": True,
            "load_compiled_cache": True,
            "save_compiled_cache": False,
            "image_attestation_environment_variable": "PHASE11_K100_IMAGE_ID",
            "cache_portability": "not_assumed; each target node must pass a fresh fixed-sample smoke",
        },
        "execution": execution,
        "capacity": capacity,
        "fixed_sample": {
            "input": fixed_row,
            "input_array_sha256": array_sha256(fixed_array),
            "expected_output": expected_row,
            "expected_logits_array_sha256_descriptive_only": array_sha256(expected_logits),
            "expected_prediction_array_sha256": array_sha256(expected_prediction),
            "cross_node_gate": "prediction array SHA must match; strict logits SHA is not a portability gate",
        },
        "evidence": evidence_rows,
        "payload_file_count": len(file_rows),
        "payload_total_bytes": total_bytes,
        "files": file_rows,
        "claims": {
            "static_bundle_ready": True,
            "runtime_validated": False,
            "cold_start_5_passed": False,
            "stability_60min_passed": False,
            "recovery_3_passed": False,
            "cross_node_smoke_passed": False,
            "deployment_ready": False,
        },
        "claim_boundaries": {
            "historical_int8_strict_logits": "failed_immutable",
            "candidate_strict_logits": "diagnostic_only; not inferred by packaging or deployment smoke",
            "native_int8_kernel": native_kernel_state,
            "fp16_full_precision_semantics": (
                "verified_frozen_artifact"
                if args.kind == "fp16_full"
                else "not_applicable"
            ),
            "fp16_full_predeployment_state": (
                "strict_logits_failed_task_accuracy_passed_runtime_acceptance_pending"
                if args.kind == "fp16_full"
                else "not_applicable"
            ),
            "provider_placement_does_not_prove_kernel_precision": True,
            "static_bundle_readiness_is_not_runtime_or_deployment_readiness": True,
            "fp16_full_remains_default_until_a_mixed_candidate_passes_all_upgrade_gates": True,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Publish only after every payload and the complete manifest are durable.
    os.replace(output, final_output)
    manifest_path = final_output / "manifest.json"
    receipt = {
        "status": "static_bundle_identity_locked",
        "bundle": str(final_output),
        "bundle_id": args.bundle_id,
        "kind": args.kind,
        "manifest": {"path": str(manifest_path), **identity(manifest_path)},
        "payload_file_count": len(file_rows),
        "payload_total_bytes": total_bytes,
        "next_required_action": "run verify_k100_bundle.py, then target-node runtime acceptance",
    }
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
