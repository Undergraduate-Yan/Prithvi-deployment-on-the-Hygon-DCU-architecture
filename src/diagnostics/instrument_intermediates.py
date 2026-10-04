#!/usr/bin/env python3
'Research implementation: instrument intermediates.'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--node-indices", required=True, help="comma-separated topological node indices")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


def main() -> int:
    args = parse_args()
    indices = sorted({int(value) for value in args.node_indices.split(",") if value.strip()})
    model = onnx.load(args.input, load_external_data=True)
    if not indices or indices[0] < 0 or indices[-1] >= len(model.graph.node):
        raise RuntimeError("probe index outside graph")
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
        known = value_info_map(inferred)
    except Exception:
        known = value_info_map(model)
    existing = {value.name for value in model.graph.output}
    probes = []
    for index in indices:
        node = model.graph.node[index]
        for ordinal, name in enumerate(node.output):
            if not name or name in existing:
                continue
            info = known.get(name)
            if info is None:
                info = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            model.graph.output.append(info)
            existing.add(name)
            probes.append({"node_index": index, "node_name": node.name, "op_type": node.op_type, "output_ordinal": ordinal, "output_name": name})
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    args.manifest.write_text(json.dumps({
        "schema": "phase7r_instrumented_onnx_manifest_v1",
        "parent": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "node_count": len(model.graph.node),
        "probes": probes,
        "graph_outputs": [value.name for value in model.graph.output],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "probes": len(probes), "outputs": len(model.graph.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
