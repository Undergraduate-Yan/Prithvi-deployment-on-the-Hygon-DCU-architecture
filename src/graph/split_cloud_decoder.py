#!/usr/bin/env python3
'Research implementation: split cloud decoder.'
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx.utils import extract_model


BOUNDARIES = [
    "/model/neck/neck.1/Reshape_2_output_0",
    "/model/neck/neck.1/Reshape_5_output_0",
    "/model/neck/neck.1/Reshape_8_output_0",
    "/model/neck/neck.1/Reshape_11_output_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def describe(path: Path) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "node_count": len(model.graph.node),
        "inputs": [item.name for item in model.graph.input],
        "outputs": [item.name for item in model.graph.output],
        "operator_types": sorted({node.op_type for node in model.graph.node}),
    }


def main() -> int:
    args = parse_args()
    source = onnx.load(args.input, load_external_data=False)
    source_inputs = [item.name for item in source.graph.input]
    source_outputs = [item.name for item in source.graph.output]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "session_12a_pre_fpn.onnx"
    suffix = args.output_dir / "session_12b_fpn_head.onnx"
    extract_model(str(args.input), str(prefix), source_inputs, BOUNDARIES, check_model=True)
    extract_model(str(args.input), str(suffix), BOUNDARIES, source_outputs, check_model=True)
    result = {
        "schema": "phase7r_session12_r2_split_candidate_v1",
        "status": "BUILT_PENDING_NUMERIC_GATE",
        "candidate": "R2_SPLIT_BEFORE_FIRST_DIVERGENT_OPERATOR",
        "single_variable": "replace original Session 12 with two sessions split at four neck reshape tensors",
        "parent_model": str(args.input.resolve()),
        "parent_sha256": digest(args.input),
        "boundary_tensors": BOUNDARIES,
        "session_count_before": 13,
        "session_count_after": 14,
        "models": [describe(prefix), describe(suffix)],
        "formal_payload_accessed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
