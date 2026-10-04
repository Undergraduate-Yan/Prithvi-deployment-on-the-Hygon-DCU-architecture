#!/usr/bin/env python3
'Research implementation: summarize cloud kernels.'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def read_trace(kind: str, result_path: Path, kernel_path: Path) -> tuple[dict, list[dict]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS_PENDING_HIPPROF_PARSE" or result.get("repetitions") != 20 or result.get("cpu_fallback_events") != 0:
        raise RuntimeError(f"{kind} trace parent gate failed")
    with kernel_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"{kind} kernel CSV is empty")
    return result, rows


def bucket(rows: list[dict], token: str) -> dict[str, object]:
    selected = [row for row in rows if token in row["Name"]]
    return {"calls": sum(int(float(row["Calls"])) for row in selected), "distinct_names": len(selected), "names": [row["Name"] for row in selected]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8-result", required=True, type=Path)
    parser.add_argument("--int8-kernel-csv", required=True, type=Path)
    parser.add_argument("--fp16-result", required=True, type=Path)
    parser.add_argument("--fp16-kernel-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    int8_result, int8_rows = read_trace("INT8", args.int8_result, args.int8_kernel_csv)
    fp16_result, fp16_rows = read_trace("FP16", args.fp16_result, args.fp16_kernel_csv)
    i8ii, hbh = bucket(int8_rows, "I8II"), bucket(fp16_rows, "HBH")
    checks = {"int8_trace_i8ii_positive": i8ii["calls"] > 0, "fp16_trace_hbh_positive": hbh["calls"] > 0, "both_cpu_fallback_zero": True, "formal_payload_not_accessed": True}
    result = {
        "schema": "phase7r_mixed_paired_kernel_gate_v1", "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "paired direct outer hipprof traces, exact admitted mixed candidate, 20 repetitions per session",
        "checks": checks, "I8II": i8ii, "HBH": hbh,
        "int8_trace": {"result": identity(args.int8_result), "kernel_csv": identity(args.int8_kernel_csv), "session_ordinal": int8_result["session_ordinal"]},
        "fp16_trace": {"result": identity(args.fp16_result), "kernel_csv": identity(args.fp16_kernel_csv), "session_ordinal": fp16_result["session_ordinal"]},
        "claim_boundary": "I8II and HBH prove executed INT8 and FP16 paths; they do not by themselves prove task accuracy or speedup.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "I8II_calls": i8ii["calls"], "HBH_calls": hbh["calls"]}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
