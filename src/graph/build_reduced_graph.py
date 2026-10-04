#!/usr/bin/env python3
'Research implementation: build reduced graph.'

from __future__ import annotations

import argparse
import hashlib
import json
import socket
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


def verify(row: dict[str, Any]) -> dict[str, Any]:
    actual = identity(Path(row["path"]))
    if (actual["size_bytes"], actual["sha256"]) != (
        int(row["size_bytes"]),
        str(row["sha256"]),
    ):
        raise RuntimeError(f"identity drift: {row['path']}")
    return actual


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--replacements-json", required=True, type=Path)
    parser.add_argument("--expected-sessions", required=True, type=int)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if socket.gethostname() != "machine3":
        raise RuntimeError("minimized RCS assembly is frozen to machine3")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    base_path = args.base_manifest.resolve(strict=True)
    base = load(base_path)
    if (
        base.get("status") != "prepared_for_cache_load_validation"
        or not all(base.get("gates", {}).values())
        or base.get("selection_boundary", {}).get("formal_90_image_test_used") is not False
    ):
        raise RuntimeError("base RCS manifest is not admitted")
    replacement_path = args.replacements_json.resolve(strict=True)
    specs = load(replacement_path)
    required = {"id", "start", "end", "precision", "build_report", "compile_result"}
    if not isinstance(specs, list) or not specs or any(set(row) != required for row in specs):
        raise RuntimeError("replacement declaration schema drift")
    occupied = []
    replacement_rows = []
    compile_results = []
    for spec in specs:
        start, end = int(spec["start"]), int(spec["end"])
        if not 0 <= start < end <= 24:
            raise RuntimeError(f"invalid replacement interval: {spec}")
        occupied.extend(range(start, end + 1))
        report_path = Path(spec["build_report"]).resolve(strict=True)
        report = load(report_path)
        if report.get("status") != "passed" or report.get("formal_90_image_test_used") is not False:
            raise RuntimeError(f"replacement build is not admitted: {spec['id']}")
        region = next(row for row in report["regions"] if row["spec"]["id"] == spec["id"])
        if (
            int(region["spec"]["start"]) != start
            or int(region["spec"]["end"]) != end
            or not all(region["gates"].values())
        ):
            raise RuntimeError(f"replacement region contract failed: {spec['id']}")
        result_path = Path(spec["compile_result"]).resolve(strict=True)
        result = load(result_path)
        if (
            result.get("status") != "passed"
            or result.get("formal_90_image_test_used") is not False
            or result.get("mxr") is None
            or result.get("budget_stopped") is not False
            or result.get("oom_killed") is not False
        ):
            raise RuntimeError(f"replacement compile is not admitted: {spec['id']}")
        model = verify(region["model"])
        if (model["size_bytes"], model["sha256"]) != (
            int(result["model"]["size_bytes"]),
            str(result["model"]["sha256"]),
        ):
            raise RuntimeError(f"compiled model identity mismatch: {spec['id']}")
        replacement_rows.append({
            "session_id": "pending",
            "start_block": start,
            "end_block": end,
            "precision": spec["precision"],
            "precision_sequence": region["spec"].get("expected_precisions"),
            "model": model,
            "cache": verify(result["mxr"]),
            "source": f"phase2d_controlled_minimize_{spec['id']}",
        })
        compile_results.append(identity(result_path))
    if len(occupied) != len(set(occupied)):
        raise RuntimeError("replacement intervals overlap")

    base_rows = []
    for row in base["candidate_sessions"]:
        covered = set(range(int(row["start_block"]), int(row["end_block"]) + 1))
        overlap = covered.intersection(occupied)
        if overlap and not covered <= set(occupied):
            raise RuntimeError(f"replacement cuts through an admitted base session: {row['session_id']}")
        if not overlap:
            base_rows.append(dict(row))
    replaced_base_coverage = sorted(
        index
        for row in base["candidate_sessions"]
        if set(range(int(row["start_block"]), int(row["end_block"]) + 1)).intersection(occupied)
        for index in range(int(row["start_block"]), int(row["end_block"]) + 1)
    )
    if replaced_base_coverage != sorted(occupied):
        raise RuntimeError("replacement union does not exactly cover the removed base sessions")

    candidate = sorted(base_rows + replacement_rows, key=lambda row: int(row["start_block"]))
    for ordinal, row in enumerate(candidate):
        row["session_id"] = f"{args.candidate_label}_{ordinal:02d}"
        row["model"] = verify(row["model"])
        row["cache"] = verify(row["cache"])
    covered = [
        index
        for row in candidate
        for index in range(int(row["start_block"]), int(row["end_block"]) + 1)
    ]
    manifest = {
        "schema": f"journal_phase2d_rcs{args.expected_sessions}_candidate_manifest_v1",
        "status": "prepared_for_cache_load_validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "hardware": "海光 K100 AI 加速卡",
        "precision_map": "M5-R",
        "candidate_name": args.candidate_name,
        "candidate_label": args.candidate_label,
        "expected_candidate_sessions": args.expected_sessions,
        "reference_name": base["reference_name"],
        "input": verify(base["input"]),
        "payload_manifest": verify(base["payload_manifest"]),
        "reference_sessions": base["reference_sessions"],
        "candidate_sessions": candidate,
        "base_manifest": identity(base_path),
        "replacements_manifest": identity(replacement_path),
        "new_interval_compile_results": compile_results,
        "selection_boundary": {
            "configuration_validation_only": True,
            "formal_90_image_test_used": False,
            "not_formal_task_accuracy_or_performance_evidence": True,
        },
        "gates": {
            "exact_expected_candidate_sessions": len(candidate) == args.expected_sessions,
            "candidate_covers_0_through_24_once": covered == list(range(25)),
            "all_replacements_compile_admitted": len(compile_results) == len(specs),
            "all_paths_and_identities_verified": True,
            "formal_90_image_test_not_used": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError(f"minimized RCS gates failed: {manifest['gates']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "sessions": len(candidate), "gates": manifest["gates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
