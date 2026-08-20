#!/usr/bin/env python3
"""Rewrite non-overlapping 2x2/stride-2 ConvTranspose into equivalent primitive ops.

For group=1, dilation=1, padding=0 and output_padding=0, each input pixel maps to
an independent 2x2 output block.  The operation is exactly representable as a
1x1 Conv producing four channel groups, followed by reshape/transpose/reshape.
The source artifact is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def attrs(node) -> dict:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def dimensions(value_info) -> list[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise RuntimeError(f"static dimension required for {value_info.name}")
        dims.append(int(dim.dim_value))
    return dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise SystemExit("source and output must differ")

    model = onnx.load(str(args.source), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    info = {
        item.name: item
        for item in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
    }
    initializers = {item.name: item for item in model.graph.initializer}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1

    replacement_nodes = []
    replacement_initializers = []
    removed_initializer_names = set()
    changed = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "ConvTranspose":
            replacement_nodes.append(node)
            continue
        a = attrs(node)
        contract = {
            "group": int(a.get("group", 1)),
            "dilations": list(a.get("dilations", [1, 1])),
            "kernel_shape": list(a.get("kernel_shape", [])),
            "pads": list(a.get("pads", [0, 0, 0, 0])),
            "strides": list(a.get("strides", [1, 1])),
            "output_padding": list(a.get("output_padding", [0, 0])),
        }
        expected = {
            "group": 1,
            "dilations": [1, 1],
            "kernel_shape": [2, 2],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
            "output_padding": [0, 0],
        }
        if contract != expected or len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(f"unsupported ConvTranspose contract at {node.name}: {contract}")
        x_name, weight_name, bias_name = node.input
        if weight_name not in initializers or bias_name not in initializers:
            raise RuntimeError(f"weights must be initializers at {node.name}")
        if consumers.get(weight_name) != 1 or consumers.get(bias_name) != 1:
            raise RuntimeError(f"weights are shared at {node.name}")
        w = numpy_helper.to_array(initializers[weight_name]).astype(np.float32, copy=False)
        b = numpy_helper.to_array(initializers[bias_name]).astype(np.float32, copy=False)
        if w.ndim != 4 or w.shape[2:] != (2, 2) or b.shape != (w.shape[1],):
            raise RuntimeError(f"unexpected weight shape at {node.name}: {w.shape}, {b.shape}")
        input_shape = dimensions(info[x_name])
        output_shape = dimensions(info[node.output[0]])
        if len(input_shape) != 4 or len(output_shape) != 4:
            raise RuntimeError(f"rank-4 tensors required at {node.name}")
        n, cin, h, width = input_shape
        n2, cout, out_h, out_w = output_shape
        if n != n2 or cin != w.shape[0] or cout != w.shape[1] or out_h != 2 * h or out_w != 2 * width:
            raise RuntimeError(f"shape contract mismatch at {node.name}: {input_shape} -> {output_shape}, {w.shape}")

        # DCR ordering: q = ((kernel_y * 2 + kernel_x) * Cout + output_channel).
        conv_w = np.empty((4 * cout, cin, 1, 1), dtype=np.float32)
        conv_b = np.empty((4 * cout,), dtype=np.float32)
        for ky in range(2):
            for kx in range(2):
                start = (ky * 2 + kx) * cout
                conv_w[start : start + cout, :, 0, 0] = w[:, :, ky, kx].T
                conv_b[start : start + cout] = b

        base = f"phase11_deconv_{index}"
        new_weight = base + "_weight"
        new_bias = base + "_bias"
        shape1_name = base + "_shape1"
        shape2_name = base + "_shape2"
        conv_out = base + "_conv"
        reshape1_out = base + "_reshape1"
        transpose_out = base + "_transpose"
        replacement_initializers.extend(
            [
                numpy_helper.from_array(conv_w, new_weight),
                numpy_helper.from_array(conv_b, new_bias),
                numpy_helper.from_array(np.asarray([n, 2, 2, cout, h, width], dtype=np.int64), shape1_name),
                numpy_helper.from_array(np.asarray([n, cout, out_h, out_w], dtype=np.int64), shape2_name),
            ]
        )
        replacement_nodes.extend(
            [
                helper.make_node("Conv", [x_name, new_weight, new_bias], [conv_out], name=base + "/Conv1x1", kernel_shape=[1, 1]),
                helper.make_node("Reshape", [conv_out, shape1_name], [reshape1_out], name=base + "/ReshapeBlocks", allowzero=0),
                helper.make_node("Transpose", [reshape1_out], [transpose_out], name=base + "/TransposeBlocks", perm=[0, 3, 4, 1, 5, 2]),
                helper.make_node("Reshape", [transpose_out, shape2_name], list(node.output), name=base + "/ReshapeOutput", allowzero=0),
            ]
        )
        removed_initializer_names.update((weight_name, bias_name))
        changed.append(
            {
                "node_index": index,
                "node_name": node.name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "old_weight": {"name": weight_name, "shape": list(w.shape)},
                "new_weight": {"name": new_weight, "shape": list(conv_w.shape)},
            }
        )

    if len(changed) != 3:
        raise RuntimeError(f"expected exactly 3 ConvTranspose nodes, found {len(changed)}")
    kept_initializers = [item for item in model.graph.initializer if item.name not in removed_initializer_names]
    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers + replacement_initializers)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output), save_as_external_data=False)
    reloaded = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(reloaded)
    remaining = sum(node.op_type == "ConvTranspose" for node in reloaded.graph.node)
    if remaining:
        raise RuntimeError(f"ConvTranspose nodes remain: {remaining}")
    report = {
        "status": "created_static_pass",
        "transformation": "ConvTranspose(k=2,s=2,p=0) -> Conv1x1 + Reshape + Transpose + Reshape",
        "source": {"path": str(args.source), "size_bytes": args.source.stat().st_size, "sha256": sha256(args.source)},
        "output": {"path": str(args.output), "size_bytes": args.output.stat().st_size, "sha256": sha256(args.output)},
        "convtranspose_replaced": len(changed),
        "convtranspose_remaining": remaining,
        "changed_nodes": changed,
        "claims": {"cpu_numeric_equivalence": False, "migraphx_numeric_equivalence": False, "deployment": False},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
