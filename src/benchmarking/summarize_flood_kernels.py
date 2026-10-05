#!/usr/bin/env python3
'Summarize flood graph kernel evidence.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def kernel_calls(path: Path, token: str) -> tuple[int, list[dict[str, Any]]]:
    calls = 0
    matches = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("Name") or "").strip()
            if name.lower() == "total" or not re.search(
                rf"(?:^|_){token}(?:_|$)", name, re.IGNORECASE
            ):
                continue
            count = int(float(row.get("Calls") or 0))
            calls += count
            matches.append({"name": name, "calls": count})
    return calls, matches


def required_tokens(region: dict[str, Any]) -> list[str]:
    sequence = region.get("precision_sequence") or [region["precision"]]
    normalized = [str(value).lower() for value in sequence if value is not None]
    has_int8 = any("int8" in value for value in normalized)
    has_fp16 = any("fp16" in value for value in normalized)
    tokens = []
    if has_int8:
        tokens.append("I8II")
    if has_fp16:
        tokens.append("HBH")
    if not tokens:
        raise RuntimeError(f"no auditable GEMM token for region {region['session_id']}: {sequence}")
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prepass-result", required=True, type=Path)
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepass_path = args.prepass_result.resolve(strict=True)
    prepass = json.loads(prepass_path.read_text(encoding="utf-8"))
    regions = manifest["candidate_sessions"]
    if (
        int(manifest.get("expected_candidate_sessions", -1)) != 13
        or len(regions) != 13
        or prepass.get("status") != "passed"
        or prepass.get("manifest", {}).get("sha256") != identity(manifest_path)["sha256"]
        or prepass.get("formal_90_image_test_used") is not False
    ):
        raise RuntimeError("final admitted RCS13 manifest/prepass required")

    rows = []
    for ordinal, region in enumerate(regions):
        region_dir = args.trace_root / f"session_{ordinal:02d}_{region['start_block']}_{region['end_block']}"
        target_path = region_dir / "target" / "result.json"
        target = json.loads(target_path.resolve(strict=True).read_text(encoding="utf-8"))
        csvs = sorted(region_dir.glob("hipprof*.kernel.csv"))
        if len(csvs) != 1:
            raise RuntimeError(f"expected one kernel CSV for session {ordinal}: {csvs}")
        inspect_path = region_dir / "container_inspect_started.json"
        inspect = json.loads(inspect_path.resolve(strict=True).read_text(encoding="utf-8"))[0]
        command = [str(item) for item in (inspect["Config"].get("Cmd") or [])]
        entrypoint = [str(item) for item in (inspect["Config"].get("Entrypoint") or [])]
        repetitions = int(target["selection"]["repetitions"])
        tokens = required_tokens(region)
        token_evidence = {}
        for token in tokens:
            calls, matches = kernel_calls(csvs[0], token)
            token_evidence[token] = {
                "calls": calls,
                "matching_kernel_rows": matches,
                "calls_ge_repetitions": calls >= repetitions,
            }
        exact = (
            target.get("model", {}).get("sha256") == region["model"]["sha256"]
            and target.get("cache", {}).get("sha256") == region["cache"]["sha256"]
        )
        gates = {
            "target_passed": target.get("status") == "target_passed_pending_external_hipprof_parse",
            "exact_model_cache_identity": exact,
            "provider_placement_passed": all(
                target.get("gates", {}).get(key, False)
                for key in (
                    "migraphx_events_positive",
                    "cpu_events_zero",
                    "input_and_outputs_device_resident",
                )
            ),
            "all_required_kernel_tokens_ge_repetitions": all(
                row["calls_ge_repetitions"] for row in token_evidence.values()
            ),
            "direct_outer_hipprof": entrypoint == ["/opt/dtk/bin/hipprof"],
            "required_trace_flags": all(
                flag in command for flag in ("--hip-trace", "--hsa-trace", "--hiptx-trace")
            ),
            "no_dynamic_session_flags": not any(
                flag in command for flag in ("--session", "--start", "--stop", "--flush", "--trace-off")
            ),
        }
        rows.append({
            "session_ordinal": ordinal,
            "session_id": region["session_id"],
            "start_block": region["start_block"],
            "end_block": region["end_block"],
            "precision": region["precision"],
            "precision_sequence": region.get("precision_sequence"),
            "model_sha256": region["model"]["sha256"],
            "cache_sha256": region["cache"]["sha256"],
            "required_kernel_tokens": tokens,
            "token_evidence": token_evidence,
            "target_result": identity(target_path),
            "kernel_csv": identity(csvs[0]),
            "container_inspect": identity(inspect_path),
            "profiled_repetitions": repetitions,
            "gates": gates,
            "status": "passed" if all(gates.values()) else "failed",
        })

    gates = {
        "exact_13_direct_session_traces": len(rows) == 13,
        "all_session_kernel_gates_passed": all(row["status"] == "passed" for row in rows),
        "all_int8_sequences_have_I8II": all(
            "I8II" in row["required_kernel_tokens"]
            for row in rows
            if any("int8" in str(value).lower() for value in (row["precision_sequence"] or [row["precision"]]))
        ),
        "all_fp16_sequences_have_HBH": all(
            "HBH" in row["required_kernel_tokens"]
            for row in rows
            if any("fp16" in str(value).lower() for value in (row["precision_sequence"] or [row["precision"]]))
        ),
        "formal_90_image_test_not_used": True,
    }
    summary = {
        "schema": "journal_phase2d_rcs13_direct_kernel_evidence_summary_v1",
        "status": "passed" if all(gates.values()) else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": "海光 K100 AI 加速卡",
        "candidate": manifest["candidate_name"],
        "manifest": identity(manifest_path),
        "prepass": identity(prepass_path),
        "regions": rows,
        "gates": gates,
        "formal_90_image_test_used": False,
        "claim_boundary": "I8II/HBH is proven by direct outer hipprof per exact static ONNX+MXR session identity; it is not task-accuracy, latency, or whole-model precision evidence.",
    }
    (args.output_root / "kernel_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_root / "candidate_session_kernel_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "session_ordinal", "session_id", "start_block", "end_block", "precision",
            "model_sha256", "cache_sha256", "required_kernel_tokens", "status",
        ))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row[key] for key in writer.fieldnames if key in row},
                "required_kernel_tokens": "+".join(row["required_kernel_tokens"]),
            })
    (args.output_root / "kernel_evidence_boundaries_zh.md").write_text(
        "# RCS13 direct outer hipprof 核迹边界\n\n"
        f"状态：`{summary['status']}`。\n\n"
        "- 最终 13 个静态 session 均采用真实配置验证边界与 device OrtValue 独立跟踪。\n"
        "- 混合 INT8/FP16 session 同时要求 I8II 与 HBH；INT8+FP32 Head 要求 I8II。\n"
        "- I8II/HBH 只证明相应 GEMM 路径，不证明正式任务精度、端到端延迟或全模型算子精度。\n"
        "- 冻结 90 图测试集未被读取。\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "gates": gates}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
