#!/usr/bin/env python3
'Research implementation: lower cloud layernorm.'

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXACT_PARENT_SHA256 = "c823e2c96bbb551d3f57792df585eb0677f43471333ebaf990a314587c1e1a5c"
EXACT_PARENT_SIZE = 1_277_042_559


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def value_contract(value_info: Any) -> dict[str, Any]:
    tensor = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "element_type": int(tensor.elem_type),
        "shape": [int(dim.dim_value) for dim in tensor.shape.dim],
    }


def attributes(node: Any) -> dict[str, Any]:
    import onnx

    return {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() or temporary_output.exists() or args.report.exists():
        raise FileExistsError("lowered model or report already exists")
    parent = identity(args.input)
    if (
        parent["sha256"] != EXACT_PARENT_SHA256
        or parent["size_bytes"] != EXACT_PARENT_SIZE
    ):
        raise RuntimeError("frozen Phase 7D ONNX parent identity mismatch")

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(args.input.resolve(strict=True)))
    opsets = {item.domain: int(item.version) for item in model.opset_import}
    if opsets.get("") != 17:
        raise RuntimeError(f"expected default ONNX opset 17, got {opsets}")
    before_contract = {
        "inputs": [value_contract(item) for item in model.graph.input],
        "outputs": [value_contract(item) for item in model.graph.output],
    }
    expected_contract = {
        "inputs": [{"name": "input", "element_type": TensorProto.FLOAT, "shape": [1, 6, 224, 224]}],
        "outputs": [{"name": "logits", "element_type": TensorProto.FLOAT, "shape": [1, 4, 224, 224]}],
    }
    if before_contract != expected_contract:
        raise RuntimeError(f"external I/O contract drift: {before_contract}")

    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    existing_names.update(initializer.name for initializer in model.graph.initializer)
    lowered_nodes = []
    new_initializers = []
    audit_rows = []
    lowered_count = 0
    for ordinal, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization" or node.domain not in ("", "ai.onnx"):
            lowered_nodes.append(node)
            continue
        attrs = attributes(node)
        axis = int(attrs.get("axis", -1))
        epsilon = float(attrs.get("epsilon", 1e-5))
        stash_type = int(attrs.get("stash_type", TensorProto.FLOAT))
        if len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(
                f"unsupported LayerNormalization arity at {node.name}: "
                f"inputs={len(node.input)} outputs={len(node.output)}"
            )
        if axis != -1 or stash_type != TensorProto.FLOAT:
            raise RuntimeError(
                f"unsupported LayerNormalization attributes at {node.name}: {attrs}"
            )
        prefix = f"__phase7f_ln_{ordinal:04d}"
        generated = {
            key: f"{prefix}_{key}"
            for key in (
                "epsilon",
                "mean",
                "centered",
                "squared",
                "variance",
                "variance_epsilon",
                "denominator",
                "normalized",
                "scaled",
            )
        }
        if any(name in existing_names for name in generated.values()):
            raise RuntimeError(f"generated tensor-name collision at {node.name}")
        existing_names.update(generated.values())
        new_initializers.extend(
            [
                numpy_helper.from_array(
                    np.asarray(epsilon, dtype=np.float32),
                    name=generated["epsilon"],
                ),
            ]
        )
        x, scale, bias = node.input
        y = node.output[0]
        lowered_nodes.extend(
            [
                helper.make_node(
                    "ReduceMean",
                    [x],
                    [generated["mean"]],
                    axes=[-1],
                    keepdims=1,
                    name=f"{prefix}_ReduceMean_1",
                ),
                helper.make_node(
                    "Sub",
                    [x, generated["mean"]],
                    [generated["centered"]],
                    name=f"{prefix}_Sub",
                ),
                helper.make_node(
                    "Mul",
                    [generated["centered"], generated["centered"]],
                    [generated["squared"]],
                    name=f"{prefix}_Square",
                ),
                helper.make_node(
                    "ReduceMean",
                    [generated["squared"]],
                    [generated["variance"]],
                    axes=[-1],
                    keepdims=1,
                    name=f"{prefix}_ReduceMean_2",
                ),
                helper.make_node(
                    "Add",
                    [generated["variance"], generated["epsilon"]],
                    [generated["variance_epsilon"]],
                    name=f"{prefix}_AddEpsilon",
                ),
                helper.make_node(
                    "Sqrt",
                    [generated["variance_epsilon"]],
                    [generated["denominator"]],
                    name=f"{prefix}_Sqrt",
                ),
                helper.make_node(
                    "Div",
                    [generated["centered"], generated["denominator"]],
                    [generated["normalized"]],
                    name=f"{prefix}_Div",
                ),
                helper.make_node(
                    "Mul",
                    [generated["normalized"], scale],
                    [generated["scaled"]],
                    name=f"{prefix}_Scale",
                ),
                helper.make_node(
                    "Add",
                    [generated["scaled"], bias],
                    [y],
                    name=f"{prefix}_Bias",
                ),
            ]
        )
        audit_rows.append(
            {
                "ordinal": ordinal,
                "source_node_name": node.name,
                "source_output": y,
                "axis": axis,
                "epsilon": epsilon,
                "stash_type": stash_type,
                "replacement_node_count": 9,
            }
        )
        lowered_count += 1

    if lowered_count != 49:
        raise RuntimeError(f"expected 49 LayerNormalization nodes, found {lowered_count}")
    del model.graph.node[:]
    model.graph.node.extend(lowered_nodes)
    model.graph.initializer.extend(new_initializers)
    producer_suffix = "phase7f-layernorm-lowered-v1"
    model.producer_name = (
        f"{model.producer_name}+{producer_suffix}" if model.producer_name else producer_suffix
    )
    after_contract = {
        "inputs": [value_contract(item) for item in model.graph.input],
        "outputs": [value_contract(item) for item in model.graph.output],
    }
    if after_contract != before_contract:
        raise RuntimeError("external I/O changed during LayerNormalization lowering")
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(temporary_output))
    reloaded = onnx.load(
        str(temporary_output.resolve(strict=True)), load_external_data=False
    )
    onnx.checker.check_model(reloaded, full_check=True)
    os.replace(temporary_output, args.output)
    lowered = identity(args.output)
    op_counts = Counter(node.op_type for node in reloaded.graph.node)
    checks = {
        "parent_identity_match": True,
        "opset_17": opsets.get("") == 17,
        "external_io_unchanged": after_contract == before_contract,
        "layernormalization_count_zero": op_counts["LayerNormalization"] == 0,
        "lowered_node_count_49": lowered_count == 49,
        "onnx_full_checker_pass": True,
        "formal_test_payload_access_count_zero": True,
    }
    report = {
        "schema": "phase7f_cloud_layernorm_lowering_report_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "MIGraphX compatibility lowering; model weights and external I/O are unchanged",
        "parent_onnx": parent,
        "lowered_onnx": lowered,
        "opset_imports": opsets,
        "external_io": after_contract,
        "lowering": {
            "source_operator": "LayerNormalization",
            "source_node_count": lowered_count,
            "replacement_primitives": [
                "ReduceMean",
                "Sub",
                "Mul",
                "ReduceMean",
                "Add",
                "Sqrt",
                "Div",
                "Mul",
                "Add",
            ],
            "audit_rows": audit_rows,
        },
        "graph": {
            "node_count": len(reloaded.graph.node),
            "operation_counts": dict(sorted(op_counts.items())),
        },
        "checks": checks,
        "data_firewall": {
            "calibration_payload_access_count": 0,
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
            "transformation_uses_weights_and_graph_structure_only": True,
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
