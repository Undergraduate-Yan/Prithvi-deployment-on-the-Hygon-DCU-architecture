#!/usr/bin/env python3
'Research implementation: extract subgraph.'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx.utils import extract_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-names", required=True, help="comma-separated boundary tensors")
    parser.add_argument("--output-names", required=True, help="comma-separated failing tensors")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [value for value in args.input_names.split(",") if value]
    outputs = [value for value in args.output_names.split(",") if value]
    extract_model(str(args.input), str(args.output), inputs, outputs, check_model=True)
    model = onnx.load(args.output, load_external_data=False)
    result = {
        "schema": "phase7r_minimal_failure_subgraph_manifest_v1",
        "status": "BUILT_PENDING_PROVIDER_CONFIRMATION",
        "parent": str(args.input.resolve()),
        "model": str(args.output.resolve()),
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
        "operator_types": sorted({node.op_type for node in model.graph.node}),
    }
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
