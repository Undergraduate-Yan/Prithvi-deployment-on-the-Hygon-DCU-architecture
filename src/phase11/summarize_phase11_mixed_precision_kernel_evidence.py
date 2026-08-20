#!/usr/bin/env python3
"""Parse v3 direct-outer-hipprof traces into per-block kernel evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import phase11_mixed_precision_common as common


SCHEMA = "phase11_mixed_precision_kernel_evidence_summary_v3"
PLAN_SCHEMA = "phase11_mixed_precision_kernel_profile_plan_v3"
TARGET_SCHEMA = "phase11_mixed_precision_block_external_hipprof_target_v3"
PREPASS_SCHEMA = "phase11_mixed_precision_boundary_prepass_v3"
RUNTIME_SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"
IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
KERNEL_RULE_VERSION = "phase11_kernel_category_rules_v1"
EXPECTED_TARGET_IDENTITY = (
    16_603,
    "58893c10fc21c10a09116579e39a0b9e6ad3cae3ab86528cbd107d067f3cba71",
)
EXPECTED_PREPASS_IDENTITY = (
    11_960,
    "b28d9e9ea7851f877c835095ab0b48900607c5ff7d8164ec20042ef37f45b3e6",
)


def pair(value: dict[str, Any]) -> tuple[int, str]:
    return int(value["size_bytes"]), str(value["sha256"])


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in row.items()}


def field(row: dict[str, str], *names: str) -> str:
    normalized = normalized_row(row)
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in normalized:
            return normalized[key]
    raise KeyError(f"none of {names!r} found in CSV fields {tuple(row)}")


def as_int(value: str) -> int:
    return int(round(float(str(value).strip())))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_kernel(name: str, precision: str) -> str:
    lowered = name.lower()
    if re.search(r"(?:^|_)i8ii(?:_|$)", lowered):
        return "int8_gemm_i8ii"
    if re.search(r"(?:^|_)hbh(?:_|$)", lowered):
        return "fp16_gemm_hbh"
    if re.search(r"(?:^|_)sb(?:_|$)", lowered):
        return "fp32_gemm_sb"
    if re.search(r"transpose|permute|layout|reorder|contiguous|pack|unpack", lowered):
        return "layout_conversion"
    if precision == "int8_qdq" and re.search(
        r"quant|dequant|nearbyint|clip.*convert|convert.*clip|convert_mul|mul.*convert",
        lowered,
    ):
        return "qdq_conversion"
    if re.search(r"cast|convert", lowered):
        return "cast_conversion"
    if re.search(r"memcpy|copy_kernel|memset", lowered):
        return "memory_kernel"
    return "other_kernel"


def parse_kernel_csv(path: Path, precision: str) -> dict[str, Any]:
    raw_rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            name = field(raw, "Name").strip()
            if not name or name.lower() == "total":
                continue
            calls = as_int(field(raw, "Calls"))
            duration = as_int(field(raw, "TotalDurationNs", "Total Duration Ns"))
            category = classify_kernel(name, precision)
            raw_rows.append(
                {
                    "name": name,
                    "calls": calls,
                    "total_duration_ns": duration,
                    "average_ns": as_int(field(raw, "AverageNs", "Average Ns")),
                    "reported_percentage": float(field(raw, "Percentage")),
                    "category": category,
                }
            )
    if not raw_rows:
        raise RuntimeError(f"kernel CSV contains no data rows: {path}")
    total_duration = sum(row["total_duration_ns"] for row in raw_rows)
    total_calls = sum(row["calls"] for row in raw_rows)
    category_duration: dict[str, int] = defaultdict(int)
    category_calls: dict[str, int] = defaultdict(int)
    for row in raw_rows:
        category_duration[row["category"]] += row["total_duration_ns"]
        category_calls[row["category"]] += row["calls"]
    categories = {}
    for name in (
        "int8_gemm_i8ii",
        "fp16_gemm_hbh",
        "fp32_gemm_sb",
        "qdq_conversion",
        "cast_conversion",
        "layout_conversion",
        "memory_kernel",
        "other_kernel",
    ):
        duration = category_duration[name]
        categories[name] = {
            "calls": category_calls[name],
            "total_duration_ns": duration,
            "percent_of_gpu_kernel_time": 100.0 * duration / total_duration,
        }
    return {
        "identity": common.identity(path),
        "kernel_row_count": len(raw_rows),
        "gpu_kernel_calls": total_calls,
        "gpu_kernel_total_duration_ns": total_duration,
        "categories": categories,
        "raw_rows": raw_rows,
        "classification_rule_version": KERNEL_RULE_VERSION,
    }


def parse_stats_tables(text: str) -> dict[str, Any]:
    tables: dict[str, list[dict[str, Any]]] = {"hip_api": [], "hsa_api": []}
    totals: dict[str, int | None] = {"hip_api": None, "hsa_api": None}
    headers_found = {"hip_api": False, "hsa_api": False}
    active: str | None = None
    for line in text.splitlines():
        if "HIP PROF:HIP API statistics" in line:
            active = "hip_api"
            headers_found[active] = True
            continue
        if "HIP PROF:HSA API statistics" in line:
            active = "hsa_api"
            headers_found[active] = True
            continue
        if line.startswith("HIP PROF:") and "statistics" in line and active is not None:
            active = None
            continue
        if active is None or not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] in {"Name", ""}:
            continue
        try:
            row = {
                "name": cells[0],
                "calls": as_int(cells[1]) if cells[1].strip() else 0,
                "total_duration_ns": as_int(cells[2]) if cells[2].strip() else 0,
                "average_ns": as_int(cells[3]) if cells[3].strip() else 0,
                "reported_percentage": float(cells[4]) if cells[4].strip() else 0.0,
            }
        except ValueError:
            continue
        if row["name"].lower() == "total":
            totals[active] = row["total_duration_ns"]
        else:
            tables[active].append(row)

    for table in tables:
        if totals[table] is None:
            totals[table] = sum(row["total_duration_ns"] for row in tables[table])

    hip_memcpy = [row for row in tables["hip_api"] if row["name"].lower().startswith("hipmemcpy")]
    hsa_copy = [
        row
        for row in tables["hsa_api"]
        if "memory_async_copy" in row["name"].lower()
        or "memory_copy" in row["name"].lower()
    ]

    def summarize(rows: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
        duration = sum(row["total_duration_ns"] for row in rows)
        return {
            "calls": sum(row["calls"] for row in rows),
            "total_duration_ns": duration,
            "percent_of_api_table_duration": 100.0 * duration / denominator if denominator else 0.0,
            "rows": rows,
        }

    return {
        "statistics_headers_found": headers_found,
        "hip_api_total_duration_ns": totals["hip_api"],
        "hsa_api_total_duration_ns": totals["hsa_api"],
        "hip_memcpy_api": summarize(hip_memcpy, int(totals["hip_api"] or 0)),
        "hsa_async_copy_api": summarize(hsa_copy, int(totals["hsa_api"] or 0)),
        "time_domain_boundary": (
            "Memcpy values are HIP/HSA API-call durations from hipprof statistics, not GPU kernel "
            "durations and not proven DMA-engine wall time. They are reported with separate API "
            "denominators and are never added to GPU-kernel category percentages."
        ),
    }


def task_evidence(
    task: dict[str, Any],
    profiles_root: Path,
    runtime_identity: dict[str, Any],
    image_id: str,
) -> dict[str, Any]:
    evidence_dir = (profiles_root / task["evidence_key"]).resolve(strict=True)
    prepass_path = (evidence_dir / "prepass" / "result.json").resolve(strict=True)
    boundary_path = (evidence_dir / "prepass" / "boundary_input.npy").resolve(strict=True)
    result_path = (evidence_dir / "target" / "result.json").resolve(strict=True)
    prepass = read_json(prepass_path)
    result = read_json(result_path)
    if (
        prepass.get("schema") != PREPASS_SCHEMA
        or prepass.get("status") != "boundary_input_materialized_untracked_prepass_passed"
    ):
        raise RuntimeError("boundary prepass did not pass")
    if pair(prepass.get("generator", {})) != EXPECTED_PREPASS_IDENTITY:
        raise RuntimeError("boundary prepass script identity drift")
    if (
        result.get("schema") != TARGET_SCHEMA
        or result.get("status") != "target_passed_pending_external_hipprof_parse"
    ):
        raise RuntimeError("profile target did not pass")
    if pair(result.get("generator", {})) != EXPECTED_TARGET_IDENTITY:
        raise RuntimeError("profile target script identity drift")
    expected_common = pair(common.identity(Path(common.__file__)))
    if pair(result.get("common_module", {})) != expected_common:
        raise RuntimeError("profile target common-module identity drift")
    if pair(prepass.get("common_module", {})) != expected_common:
        raise RuntimeError("boundary prepass common-module identity drift")

    selection = result.get("selection", {})
    prepass_selection = prepass.get("selection", {})
    if (
        selection.get("candidate_id") != task["canonical_candidate_id"]
        or int(selection.get("segment_index", -1)) != int(task["segment_index"])
        or selection.get("expected_precision") != task["precision"]
    ):
        raise RuntimeError("profile target selection differs from plan")
    if (
        prepass_selection.get("candidate_id") != task["canonical_candidate_id"]
        or int(prepass_selection.get("segment_index", -1)) != int(task["segment_index"])
        or prepass_selection.get("expected_precision") != task["precision"]
    ):
        raise RuntimeError("boundary prepass selection differs from plan")
    if result.get("container_image_id") != image_id:
        raise RuntimeError("profile target container image identity drift")
    if prepass.get("container_image_id") != image_id:
        raise RuntimeError("boundary prepass container image identity drift")
    if pair(result.get("runtime_fingerprint", {})) != pair(runtime_identity):
        raise RuntimeError("profile target runtime fingerprint identity drift")
    if pair(prepass.get("runtime_fingerprint", {})) != pair(runtime_identity):
        raise RuntimeError("boundary prepass runtime fingerprint identity drift")
    if pair(result.get("manifest", {})) != pair(task["canonical_manifest_identity"]):
        raise RuntimeError("profile target manifest identity drift")
    if pair(prepass.get("manifest", {})) != pair(task["canonical_manifest_identity"]):
        raise RuntimeError("boundary prepass manifest identity drift")
    selected = result.get("selected_artifacts", {})
    prepass_selected = prepass.get("selected_artifacts", {})
    if pair(selected.get("model", {})) != pair(task["model_identity"]):
        raise RuntimeError("profile target model identity drift")
    if pair(selected.get("cache", {})) != pair(task["cache_identity"]):
        raise RuntimeError("profile target cache identity drift")
    if pair(prepass_selected.get("model", {})) != pair(task["model_identity"]):
        raise RuntimeError("boundary prepass target-model identity drift")
    if pair(prepass_selected.get("cache", {})) != pair(task["cache_identity"]):
        raise RuntimeError("boundary prepass target-cache identity drift")

    boundary_identity = common.identity(boundary_path)
    receipt_path = (evidence_dir / "boundary_receipt.txt").resolve(strict=True)
    receipt = {}
    for line in receipt_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise RuntimeError("malformed boundary receipt line")
        key, value = line.split("=", 1)
        if key in receipt:
            raise RuntimeError("duplicate boundary receipt key")
        receipt[key] = value
    if (
        receipt.get("boundary_input_size_bytes") != str(boundary_identity["size_bytes"])
        or receipt.get("boundary_input_sha256") != boundary_identity["sha256"]
        or receipt.get("prepass_result_size_bytes") != str(prepass_path.stat().st_size)
        or receipt.get("prepass_result_sha256") != common.sha256(prepass_path)
    ):
        raise RuntimeError("host boundary/prepass SHA receipt drift")
    if pair(prepass.get("boundary_input", {})) != pair(boundary_identity):
        raise RuntimeError("boundary input differs from prepass record")
    if pair(result.get("boundary_input", {})) != pair(boundary_identity):
        raise RuntimeError("boundary input differs from target record")
    if pair(result.get("boundary_prepass_result", {})) != pair(common.identity(prepass_path)):
        raise RuntimeError("target did not lock the exact prepass result")
    if pair(result.get("source_sample", {})) != pair(prepass.get("source_sample", {})):
        raise RuntimeError("target/prepass source-sample identity drift")

    planned_lineage = task.get("boundary_prepass_lineage", {})
    observed_lineage = prepass.get("lineage", {})
    if observed_lineage.get("canonical_candidate_id") != planned_lineage.get(
        "canonical_candidate_id"
    ):
        raise RuntimeError("prepass canonical candidate lineage drift")
    if pair(observed_lineage.get("manifest", {})) != pair(
        planned_lineage.get("canonical_manifest_identity", {})
    ):
        raise RuntimeError("prepass manifest lineage drift")
    planned_upstream = planned_lineage.get("upstream_segments", [])
    observed_upstream = observed_lineage.get("upstream_segments", [])
    if len(planned_upstream) != int(task["segment_index"]) or len(observed_upstream) != len(
        planned_upstream
    ):
        raise RuntimeError("prepass upstream lineage length drift")
    for expected, observed in zip(planned_upstream, observed_upstream, strict=True):
        if (
            int(observed.get("index", -1)) != int(expected["index"])
            or observed.get("precision") != expected["precision"]
            or pair(observed.get("model", {})) != pair(expected["model_identity"])
            or pair(observed.get("cache", {})) != pair(expected["cache_identity"])
        ):
            raise RuntimeError(f"prepass upstream artifact lineage drift at {expected['index']}")
    target_lineage = observed_lineage.get("target_segment", {})
    if (
        int(target_lineage.get("index", -1)) != int(task["segment_index"])
        or target_lineage.get("precision") != task["precision"]
        or pair(target_lineage.get("model", {})) != pair(task["model_identity"])
        or pair(target_lineage.get("cache", {})) != pair(task["cache_identity"])
    ):
        raise RuntimeError("prepass target artifact lineage drift")
    if result.get("prepass_lineage") != observed_lineage:
        raise RuntimeError("target did not preserve exact prepass lineage")
    if prepass.get("lineage_sha256") != canonical_sha256(observed_lineage):
        raise RuntimeError("boundary prepass lineage digest drift")

    boundary = np.load(boundary_path, allow_pickle=False)
    contract = task.get("boundary_input_contract", {})
    if (
        boundary.dtype != np.float32
        or contract.get("elem_type") != 1
        or list(boundary.shape) != contract.get("shape")
        or not np.isfinite(boundary).all()
    ):
        raise RuntimeError("materialized boundary does not satisfy the planned float32 contract")
    boundary_tensor_sha = common.array_sha256(np.ascontiguousarray(boundary))
    if (
        boundary_tensor_sha != observed_lineage.get("boundary_tensor_sha256")
        or boundary_tensor_sha != prepass.get("runtime", {}).get("boundary_tensor_sha256")
    ):
        raise RuntimeError("boundary tensor SHA drift")

    prepass_claims = prepass.get("claims", {})
    prepass_runtime = prepass.get("runtime", {})
    if not (
        prepass_claims.get("untracked_prepass_completed")
        and prepass_claims.get("manifest_model_cache_lineage_locked")
        and prepass_claims.get("boundary_input_sha_locked")
        and prepass_claims.get("upstream_device_ortvalue_chain_preserved")
        and int(prepass_runtime.get("upstream_segments_executed", -1))
        == int(task["segment_index"])
        and prepass_runtime.get("upstream_device_ortvalue_chain_preserved")
        and prepass_runtime.get("all_values_finite")
    ):
        raise RuntimeError("boundary prepass operational/lineage gates failed")
    if prepass_claims.get("kernel_precision_verified"):
        raise RuntimeError("boundary prepass improperly claimed kernel precision")

    prepass_inspect_path = (
        evidence_dir / "prepass_container" / "container_inspect_started.json"
    ).resolve(strict=True)
    prepass_inspected = json.loads(prepass_inspect_path.read_text(encoding="utf-8"))
    if not isinstance(prepass_inspected, list) or len(prepass_inspected) != 1:
        raise RuntimeError("prepass container inspect is not one Docker object")
    prepass_config = prepass_inspected[0].get("Config", {})
    prepass_entrypoint = prepass_config.get("Entrypoint") or []
    prepass_command = prepass_config.get("Cmd") or []
    if isinstance(prepass_entrypoint, str):
        prepass_entrypoint = [prepass_entrypoint]
    if isinstance(prepass_command, str):
        prepass_command = [prepass_command]
    prepass_command = [str(value) for value in prepass_command]
    if prepass_entrypoint != ["/usr/bin/python3"]:
        raise RuntimeError("boundary prepass was not an unprofiled Python entrypoint")
    if "/work/tools/materialize_phase11_mixed_precision_block_input.py" not in prepass_command:
        raise RuntimeError("boundary prepass command drift")
    if any("hipprof" in value.lower() for value in prepass_entrypoint + prepass_command):
        raise RuntimeError("hipprof unexpectedly attached to boundary prepass")
    if prepass_command.count("--sample") != 1:
        raise RuntimeError("boundary prepass sample argument drift")
    sample_arg = prepass_command[prepass_command.index("--sample") + 1]
    if Path(sample_arg).suffix.lower() not in {".npy", ".npz", ".pt", ".pth"}:
        raise RuntimeError("boundary prepass sample suffix was not preserved")
    prepass_env = prepass_config.get("Env") or []
    prepass_path_value = next(
        (str(value)[5:] for value in prepass_env if str(value).startswith("PATH=")), ""
    )
    if not prepass_path_value.startswith("/work/runtime_tools:"):
        raise RuntimeError("prepass PATH does not prioritize the static lsmod shim")
    prepass_mounts = prepass_inspected[0].get("Mounts", [])
    if not any(str(mount.get("Destination", "")) == sample_arg for mount in prepass_mounts):
        raise RuntimeError("boundary prepass sample argument is not a direct suffix-preserving mount")

    claims = result.get("claims", {})
    placement = result.get("ort_provider_placement", {})
    trace_contract = result.get("trace_contract", {})
    target_runtime = result.get("runtime", {})
    if not (
        claims.get("manifest_model_cache_identity_locked")
        and claims.get("boundary_prepass_lineage_locked")
        and claims.get("selected_segment_migraphx_positive_cpu_zero")
        and claims.get("profiled_input_and_output_device_resident")
        and claims.get("selected_segment_only_no_upstream_execution")
        and claims.get("internal_dynamic_hipprof_control_absent")
        and placement.get("passed")
        and trace_contract.get("mode") == "direct_outer_hipprof_no_dynamic_session_v3"
        and int(trace_contract.get("upstream_segments_executed_in_target_process", -1)) == 0
        and int(trace_contract.get("internal_trace_control_calls", -1)) == 0
        and trace_contract.get("final_output_d2h_for_finite_sha_validation_is_inside_outer_trace")
        and int(target_runtime.get("sessions_created", -1)) == 1
        and int(target_runtime.get("upstream_segments_executed", -1)) == 0
        and int(target_runtime.get("profiled_selected_segment_calls", -1))
        == int(selection.get("repetitions", -2))
    ):
        raise RuntimeError("profile target operational/placement gates failed")
    if claims.get("kernel_precision_verified"):
        raise RuntimeError("profile target improperly claimed kernel precision before parse")
    profile_record = placement.get("profile", {})
    profile_path = Path(str(profile_record.get("path", ""))).resolve(strict=True)
    if pair(common.identity(profile_path)) != pair(profile_record):
        raise RuntimeError("ORT placement profile identity drift")
    reparsed_placement = common.parse_profile(profile_path)
    if not reparsed_placement.get("passed") or reparsed_placement.get(
        "provider_event_counts"
    ) != placement.get("provider_event_counts"):
        raise RuntimeError("ORT placement profile content drift")

    exit_path = (evidence_dir / "container.exit").resolve(strict=True)
    if exit_path.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("hipprof container exit code was nonzero")
    inspect_path = (evidence_dir / "container_inspect_started.json").resolve(strict=True)
    inspected = json.loads(inspect_path.read_text(encoding="utf-8"))
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise RuntimeError("trace container inspect is not one Docker object")
    config = inspected[0].get("Config", {})
    entrypoint = config.get("Entrypoint") or []
    command = config.get("Cmd") or []
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    if isinstance(command, str):
        command = [command]
    command = [str(value) for value in command]
    if entrypoint != ["/opt/dtk/bin/hipprof"]:
        raise RuntimeError(f"trace container is not direct outer hipprof: {entrypoint}")
    if not {"--hip-trace", "--hsa-trace", "--hiptx-trace"}.issubset(command):
        raise RuntimeError("direct outer hipprof trace flags are incomplete")
    forbidden = {"--trace-off", "--session", "--start", "--stop", "--flush"}
    if forbidden.intersection(command):
        raise RuntimeError("dynamic/session hipprof control unexpectedly present")
    if "/work/tools/profile_phase11_mixed_precision_block.py" not in command:
        raise RuntimeError("trace command does not select the v3 one-block target")
    if any(
        sample_path in command
        for sample_path in ("/work/sample.npy", "/work/sample.npz", "/work/sample.pt", "/work/sample.pth")
    ):
        raise RuntimeError("trace command unexpectedly exposes the original sample")
    if command.count("--segment-index") != 1 or command.count("--boundary-input") != 1:
        raise RuntimeError("trace command target/boundary scope drift")
    expected_boundary_arg = f"/work/run/profiles/{task['evidence_key']}/prepass/boundary_input.npy"
    expected_prepass_arg = f"/work/run/profiles/{task['evidence_key']}/prepass/result.json"
    if command[command.index("--boundary-input") + 1] != expected_boundary_arg:
        raise RuntimeError("trace command boundary input path drift")
    if command.count("--boundary-prepass-result") != 1 or command[
        command.index("--boundary-prepass-result") + 1
    ] != expected_prepass_arg:
        raise RuntimeError("trace command prepass-result path drift")
    mounts = inspected[0].get("Mounts", [])
    if any(
        str(mount.get("Destination", "")).lower()
        in {"/work/sample.npy", "/work/sample.npz", "/work/sample.pt", "/work/sample.pth"}
        for mount in mounts
    ):
        raise RuntimeError("trace container unexpectedly mounted the original sample")
    env_rows = config.get("Env") or []
    path_value = next((str(value)[5:] for value in env_rows if str(value).startswith("PATH=")), "")
    if not path_value.startswith("/work/runtime_tools:"):
        raise RuntimeError("trace container PATH does not prioritize the static lsmod shim")

    kernel_csvs = sorted(evidence_dir.glob("hipprof*.kernel.csv"))
    if len(kernel_csvs) != 1:
        raise RuntimeError(f"expected exactly one hipprof kernel CSV; got {kernel_csvs}")
    log_path = (evidence_dir / "container.log").resolve(strict=True)
    parsed = parse_kernel_csv(kernel_csvs[0], task["precision"])
    api = parse_stats_tables(log_path.read_text(encoding="utf-8", errors="replace"))
    if not all(api["statistics_headers_found"].values()):
        raise RuntimeError(
            f"hipprof HIP/HSA statistics tables are incomplete: {api['statistics_headers_found']}"
        )
    expected_category = "int8_gemm_i8ii" if task["precision"] == "int8_qdq" else "fp16_gemm_hbh"
    token_calls = int(parsed["categories"][expected_category]["calls"])
    repetitions = int(selection["repetitions"])
    kernel_proof_passed = token_calls >= repetitions
    return {
        **task,
        "status": "passed" if kernel_proof_passed else "expected_kernel_token_not_detected",
        "boundary_prepass_result": common.identity(prepass_path),
        "boundary_prepass_container_inspect": common.identity(prepass_inspect_path),
        "boundary_receipt": common.identity(receipt_path),
        "boundary_input": boundary_identity,
        "source_sample": prepass.get("source_sample"),
        "boundary_tensor_sha256": boundary_tensor_sha,
        "prepass_lineage_sha256": prepass.get("lineage_sha256"),
        "target_result": common.identity(result_path),
        "container_exit": common.identity(exit_path),
        "container_inspect_started": common.identity(inspect_path),
        "container_log": common.identity(log_path),
        "hipprof_artifacts": [
            common.identity(path)
            for path in sorted(evidence_dir.glob("hipprof*"))
            if path.is_file()
        ],
        "provider_placement": {
            "passed": True,
            "provider_event_counts": reparsed_placement.get("provider_event_counts", {}),
            "evidence": common.identity(profile_path),
        },
        "kernel_evidence": parsed,
        "copy_api_evidence": api,
        "expected_kernel_token": task["expected_kernel_token"],
        "expected_kernel_category": expected_category,
        "expected_kernel_calls": token_calls,
        "profiled_repetitions": repetitions,
        "kernel_proof_passed": kernel_proof_passed,
        "provider_placement_is_kernel_proof": False,
        "trace_mode": "direct_outer_hipprof_no_dynamic_session_v3",
        "trace_scope": {
            "upstream_segments_in_trace": 0,
            "selected_segment_calls": repetitions,
            "target_session_creation_and_boundary_upload_in_trace": True,
            "final_output_d2h_for_finite_sha_validation_in_trace": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not IMAGE_RE.fullmatch(args.container_image_id):
        parser.error("--container-image-id must be a full sha256 image ID")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claims": {
            "all_required_profiles_parsed": False,
            "selected_task_smoke_kernel_gates_passed": False,
            "all_untracked_boundary_prepasses_lineage_locked": False,
            "all_traces_direct_outer_hipprof_without_dynamic_session": False,
            "all_traces_exclude_upstream_segment_execution": False,
            "all_profiled_blocks_migraphx_positive_cpu_zero": False,
            "all_retained_int8_blocks_have_i8ii": False,
            "all_fp16_blocks_have_hbh": False,
            "native_int8_kernel_verified_for_every_retained_block": False,
            "strict_numeric_equivalence_confirmed": False,
            "speedup_proved_by_kernel_trace": False,
            "deployment_ready": False,
        },
    }
    exit_code = 2
    try:
        plan_identity = common.identity(args.plan)
        plan = read_json(args.plan)
        admitted_plan_statuses = {
            "planned_identity_locked_no_kernel_claim",
            "planned_identity_locked_smoke_no_full_kernel_claim",
        }
        if plan.get("schema") != PLAN_SCHEMA or plan.get("status") not in admitted_plan_statuses:
            raise RuntimeError("kernel profile plan schema/status drift")
        smoke_mode = plan.get("plan_mode") == "selected_task_smoke"
        if smoke_mode != (
            plan.get("status") == "planned_identity_locked_smoke_no_full_kernel_claim"
        ):
            raise RuntimeError("kernel profile plan mode/status mismatch")
        if pair(plan.get("common_module", {})) != pair(common.identity(Path(common.__file__))):
            raise RuntimeError("profile plan common-module identity drift")
        if plan.get("claims", {}).get("kernel_precision_verified"):
            raise RuntimeError("profile plan improperly contains a kernel claim")
        runtime_identity = common.identity(args.runtime_fingerprint)
        runtime = read_json(args.runtime_fingerprint)
        if runtime.get("schema") != RUNTIME_SCHEMA:
            raise RuntimeError("runtime fingerprint schema drift")
        if runtime.get("onnxruntime") != common.EXPECTED_ORT_VERSION or common.MGX not in runtime.get(
            "available_providers", []
        ):
            raise RuntimeError("runtime fingerprint is not the admitted ORT/MIGraphX runtime")
        tasks = plan.get("profile_tasks", [])
        if not tasks or len({row["evidence_key"] for row in tasks}) != len(tasks):
            raise RuntimeError("profile plan tasks are empty or non-unique")

        evidence_by_key: dict[str, dict[str, Any]] = {}
        task_failures = []
        for task in tasks:
            try:
                evidence_by_key[task["evidence_key"]] = task_evidence(
                    task, args.profiles_root.resolve(strict=True), runtime_identity, args.container_image_id
                )
            except Exception as exc:
                task_failures.append(
                    {
                        "evidence_key": task.get("evidence_key"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        matrix_rows = []
        for planned in plan.get("candidate_block_matrix", []):
            evidence = evidence_by_key.get(planned["evidence_key"])
            categories = evidence["kernel_evidence"]["categories"] if evidence else {}
            copy_api = evidence["copy_api_evidence"] if evidence else {}
            matrix_rows.append(
                {
                    **planned,
                    "profile_reused_across_candidates": bool(
                        evidence and evidence.get("reference_count", 0) > 1
                    ),
                    "boundary_input_tensor_sha256": (
                        evidence.get("boundary_tensor_sha256") if evidence else None
                    ),
                    "canonical_prepass_candidate": (
                        evidence.get("canonical_candidate_id") if evidence else None
                    ),
                    "trace_mode": evidence.get("trace_mode") if evidence else None,
                    "provider_placement_passed": bool(
                        evidence and evidence["provider_placement"]["passed"]
                    ),
                    "expected_kernel_token": evidence.get("expected_kernel_token") if evidence else None,
                    "expected_kernel_calls": evidence.get("expected_kernel_calls", 0) if evidence else 0,
                    "kernel_proof_passed": bool(evidence and evidence["kernel_proof_passed"]),
                    "gpu_kernel_total_duration_ns": (
                        evidence["kernel_evidence"]["gpu_kernel_total_duration_ns"] if evidence else 0
                    ),
                    "int8_i8ii_duration_ns": categories.get("int8_gemm_i8ii", {}).get(
                        "total_duration_ns", 0
                    ),
                    "fp16_hbh_duration_ns": categories.get("fp16_gemm_hbh", {}).get(
                        "total_duration_ns", 0
                    ),
                    "qdq_conversion_duration_ns": categories.get("qdq_conversion", {}).get(
                        "total_duration_ns", 0
                    ),
                    "cast_conversion_duration_ns": categories.get("cast_conversion", {}).get(
                        "total_duration_ns", 0
                    ),
                    "layout_conversion_duration_ns": categories.get("layout_conversion", {}).get(
                        "total_duration_ns", 0
                    ),
                    "hip_memcpy_api_duration_ns": copy_api.get("hip_memcpy_api", {}).get(
                        "total_duration_ns", 0
                    ),
                    "hsa_async_copy_api_duration_ns": copy_api.get("hsa_async_copy_api", {}).get(
                        "total_duration_ns", 0
                    ),
                }
            )

        candidate_results = []
        for candidate in plan.get("candidates", []):
            candidate_id = candidate["candidate_id"]
            rows = [row for row in matrix_rows if row["candidate_id"] == candidate_id]
            if smoke_mode and not rows:
                continue
            if not smoke_mode and len(rows) != 24:
                raise RuntimeError(f"candidate matrix does not contain 24 blocks for {candidate_id}")
            int8_rows = [row for row in rows if row["precision"] == "int8_qdq"]
            fp16_rows = [row for row in rows if row["precision"] == "fp16"]
            candidate_results.append(
                {
                    "candidate_id": candidate_id,
                    "plan_mode": plan.get("plan_mode"),
                    "profiled_block_count": len(rows),
                    "int8_block_count": len(int8_rows),
                    "fp16_block_count": len(fp16_rows),
                    "selected_profiled_blocks_provider_placement_passed": all(
                        row["provider_placement_passed"] for row in rows
                    ),
                    "all_blocks_provider_placement_passed": (not smoke_mode)
                    and len(rows) == 24
                    and all(row["provider_placement_passed"] for row in rows),
                    "selected_retained_int8_blocks_have_i8ii_or_none": all(
                        row["kernel_proof_passed"] for row in int8_rows
                    ),
                    "all_retained_int8_blocks_have_i8ii": (not smoke_mode)
                    and bool(int8_rows)
                    and all(row["kernel_proof_passed"] for row in int8_rows),
                    "all_fp16_blocks_have_hbh_or_none": all(
                        row["kernel_proof_passed"] for row in fp16_rows
                    ),
                    "selected_profiled_blocks_kernel_mapping_verified": all(
                        row["kernel_proof_passed"] for row in rows
                    ),
                    "mixed_precision_kernel_mapping_verified": (not smoke_mode)
                    and len(rows) == 24
                    and all(row["kernel_proof_passed"] for row in rows),
                    "failed_blocks": [
                        row["segment_index"] for row in rows if not row["kernel_proof_passed"]
                    ],
                }
            )

        all_tasks_parsed = not task_failures and len(evidence_by_key) == len(tasks)
        all_placement = all(
            row["provider_placement"]["passed"] for row in evidence_by_key.values()
        ) and all_tasks_parsed
        any_int8 = any(row["precision"] == "int8_qdq" for row in evidence_by_key.values())
        all_int8 = all_tasks_parsed and all(
            row["kernel_proof_passed"]
            for row in evidence_by_key.values()
            if row["precision"] == "int8_qdq"
        ) and (any_int8 or smoke_mode)
        all_fp16 = all_tasks_parsed and all(
            row["kernel_proof_passed"]
            for row in evidence_by_key.values()
            if row["precision"] == "fp16"
        )
        passed = all_tasks_parsed and all_placement and all_int8 and all_fp16

        csv_fields = [
            "candidate_id",
            "segment_index",
            "label",
            "precision",
            "evidence_key",
            "profile_reused_across_candidates",
            "canonical_prepass_candidate",
            "boundary_input_tensor_sha256",
            "trace_mode",
            "provider_placement_passed",
            "expected_kernel_token",
            "expected_kernel_calls",
            "kernel_proof_passed",
            "gpu_kernel_total_duration_ns",
            "int8_i8ii_duration_ns",
            "fp16_hbh_duration_ns",
            "qdq_conversion_duration_ns",
            "cast_conversion_duration_ns",
            "layout_conversion_duration_ns",
            "hip_memcpy_api_duration_ns",
            "hsa_async_copy_api_duration_ns",
        ]
        csv_path = args.output_dir / "candidate_block_kernel_matrix.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows([{key: row[key] for key in csv_fields} for row in matrix_rows])

        summary.update(
            {
                "status": (
                    "selected_task_smoke_passed_no_full_mapping_claim"
                    if passed and smoke_mode
                    else "passed"
                    if passed
                    else "incomplete_or_kernel_proof_failed"
                ),
                "plan_mode": plan.get("plan_mode"),
                "container_image_id": args.container_image_id,
                "identities": {
                    "generator": common.identity(Path(__file__)),
                    "common_module": common.identity(Path(common.__file__)),
                    "plan": plan_identity,
                    "runtime_fingerprint": runtime_identity,
                    "matrix_csv": common.identity(csv_path),
                },
                "profile_tasks": [
                    evidence_by_key.get(
                        task["evidence_key"],
                        {**task, "status": "missing_or_failed", "kernel_proof_passed": False},
                    )
                    for task in tasks
                ],
                "task_failures": task_failures,
                "candidate_results": candidate_results,
                "candidate_block_matrix": matrix_rows,
                "claims": {
                    **summary["claims"],
                    "all_required_profiles_parsed": all_tasks_parsed,
                    "selected_task_smoke_kernel_gates_passed": smoke_mode and passed,
                    "all_untracked_boundary_prepasses_lineage_locked": all_tasks_parsed,
                    "all_traces_direct_outer_hipprof_without_dynamic_session": all_tasks_parsed,
                    "all_traces_exclude_upstream_segment_execution": all_tasks_parsed,
                    "all_profiled_blocks_migraphx_positive_cpu_zero": all_placement,
                    "all_retained_int8_blocks_have_i8ii": (not smoke_mode) and all_int8,
                    "all_fp16_blocks_have_hbh": (not smoke_mode) and all_fp16,
                    "native_int8_kernel_verified_for_every_retained_block": (not smoke_mode)
                    and all_int8,
                },
                "evidence_boundary": {
                    "provider_placement": (
                        "ORT profile with MIGraphX>0 and CPU=0 proves placement only; it is not "
                        "accepted as I8II/HBH evidence."
                    ),
                    "kernel_precision": (
                        "I8II/HBH is claimed per static model+MXR identity only when hipprof records "
                        "the expected token at least once per profiled repetition."
                    ),
                    "trace_scope": (
                        "A normal, unprofiled prepass executes canonical upstream segments with "
                        "device OrtValues and materializes a SHA-locked float32 boundary NPY. The "
                        "direct outer hipprof process receives only that NPY and executes exactly "
                        "one selected segment; it never constructs an upstream session."
                    ),
                    "conversion_time": (
                        "Q/DQ, Cast, and layout percentages share the GPU-kernel-time denominator. "
                        "Memcpy is reported separately in HIP/HSA API time domains. Direct outer "
                        "hipprof also observes target-session creation, boundary H2D upload, output "
                        "allocation, and the final output D2H used for finite/SHA validation, so "
                        "these categories are trace-scope evidence, not isolated steady-state percentages."
                    ),
                    "not_proved": [
                        *(["full 24-block candidate kernel mapping (selected-task smoke only)"] if smoke_mode else []),
                        "strict CPU/MIGraphX logits equivalence",
                        "model-level speedup or causal speedup attribution",
                        "full/task-head INT8 admission",
                        "deployment stability",
                    ],
                },
            }
        )
        exit_code = 0 if passed else 2
    except Exception as exc:
        summary.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    summary_path = args.output_dir / "kernel_evidence_summary.json"
    common.json_dump(summary_path, summary)
    boundary_path = args.output_dir / "证据边界.md"
    boundary_path.write_text(
        "# Phase 11 混合精度 kernel 证据边界\n\n"
        f"状态：`{summary['status']}`。\n\n"
        "- **Provider placement 与 kernel 精度分开。** ORT profile 中 MIGraphX 事件大于 0、"
        "CPU 事件为 0，只能证明该分段落在 MIGraphX，不能单独证明执行了 INT8 或 FP16 kernel。\n"
        "- **跟踪范围。** 每个 block 先由未跟踪 prepass 用 device OrtValue 执行其前序段并生成"
        "带 SHA 与完整 manifest/model/MXR lineage 的 float32 边界输入；随后直接外部 `hipprof` "
        "只启动目标单段进程，不使用 `--trace-off`、`--session` 或 start/stop/flush 控制。\n"
        "- **INT8 证明口径。** 只有对应模型与 MXR 的 hipprof 记录中出现 `I8II`，并且调用数"
        "不少于被跟踪的重复次数，才把该 block 标记为原生 INT8 GEMM 已确认。\n"
        "- **FP16 证明口径。** 同理，只有出现 `HBH` 才把该 block 标记为 FP16 GEMM 已确认。\n"
        "- **开销口径。** Q/DQ、Cast、layout conversion 按 GPU kernel 时间分别统计；Memcpy "
        "采用 hipprof 的 HIP/HSA API 调用时间并使用独立分母，二者不得相加为端到端时间。直接"
        "外部 trace 还包含目标 session 创建、边界 H2D 上传、输出分配，以及用于 finite/SHA 校验的"
        "最终输出 D2H，因此转换占比不表述为纯稳态占比。\n"
        "- **复用口径。** 多个候选引用完全相同 SHA256 的静态 ONNX 与 MXR 时，只采集一次"
        " kernel 证据；该证据不跨不同 artifact 身份复用。\n"
        "- **不能推出的结论。** 本证据不覆盖既有 strict logits failure，不单独证明模型级加速，"
        "也不代表 full/task-head INT8 或部署稳定性已经通过。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
