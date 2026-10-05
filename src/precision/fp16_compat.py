#!/usr/bin/env python3
"""Create an FP16 MIGraphX compatibility variant without modifying the source.

Transformations:
1. LayerNormalization -> primitive FP32 statistic island -> FP16 output.
2. Non-overlapping FP16 ConvTranspose(k=2,s=2) -> FP16 Conv1x1/pixel shuffle.
3. Expose only the known fpn4 MaxPool tensor as an additional graph output.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SOURCE_SIZE = 638_894_819
SOURCE_SHA256 = "10df534d4dbaabf8336e4195a60d2ebd719b14c7a3e34362d48a0609425e6dbd"
BARRIER_NODE = "/task/model/decoder/fpn4/fpn4.0/MaxPool"
BARRIER_TENSOR = "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0"


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


def rewrite_layernorm(model: onnx.ModelProto) -> list[dict]:
    replacement = []
    changed = []
    existing = {item.name for item in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization":
            replacement.append(node)
            continue
        a = attrs(node)
        axis = int(a.get("axis", -1))
        epsilon = float(a.get("epsilon", 1e-5))
        stash_type = int(a.get("stash_type", TensorProto.FLOAT))
        if axis != -1 or stash_type != TensorProto.FLOAT or len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(f"unsupported LayerNormalization at {node.name}: {a}")
        base = f"phase11_fp16_ln_{index}"
        epsilon_name = base + "_epsilon_fp32"
        if epsilon_name in existing:
            raise RuntimeError(f"initializer collision: {epsilon_name}")
        existing.add(epsilon_name)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name)
        )
        x, scale, bias = node.input
        y = node.output[0]
        x32, scale32, bias32 = base + "_x32", base + "_scale32", base + "_bias32"
        mean, centered, squared = base + "_mean", base + "_centered", base + "_squared"
        variance, var_eps, std = base + "_variance", base + "_var_eps", base + "_std"
        norm, scaled, y32 = base + "_norm", base + "_scaled", base + "_y32"
        replacement.extend(
            [
                helper.make_node("Cast", [x], [x32], name=base + "/CastInputFP32", to=TensorProto.FLOAT),
                helper.make_node("Cast", [scale], [scale32], name=base + "/CastScaleFP32", to=TensorProto.FLOAT),
                helper.make_node("Cast", [bias], [bias32], name=base + "/CastBiasFP32", to=TensorProto.FLOAT),
                helper.make_node("ReduceMean", [x32], [mean], name=base + "/ReduceMean", axes=[-1], keepdims=1),
                helper.make_node("Sub", [x32, mean], [centered], name=base + "/Sub"),
                helper.make_node("Mul", [centered, centered], [squared], name=base + "/Square"),
                helper.make_node("ReduceMean", [squared], [variance], name=base + "/Variance", axes=[-1], keepdims=1),
                helper.make_node("Add", [variance, epsilon_name], [var_eps], name=base + "/AddEpsilon"),
                helper.make_node("Sqrt", [var_eps], [std], name=base + "/Sqrt"),
                helper.make_node("Div", [centered, std], [norm], name=base + "/Div"),
                helper.make_node("Mul", [norm, scale32], [scaled], name=base + "/Scale"),
                helper.make_node("Add", [scaled, bias32], [y32], name=base + "/Bias"),
                helper.make_node("Cast", [y32], [y], name=base + "/CastOutputFP16", to=TensorProto.FLOAT16),
            ]
        )
        changed.append(
            {
                "node_index": index,
                "node_name": node.name,
                "axis": axis,
                "epsilon": epsilon,
                "stash_type": stash_type,
                "mode": "fp16_io_fp32_statistic_island",
            }
        )
    if len(changed) != 49:
        raise RuntimeError(f"expected 49 LayerNormalization nodes, found {len(changed)}")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    return changed


def rewrite_convtranspose(model: onnx.ModelProto) -> list[dict]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    info = {
        item.name: item
        for item in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
    }
    initializers = {item.name: item for item in model.graph.initializer}
    consumers: dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1
    nodes = []
    added_initializers = []
    removed = set()
    changed = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "ConvTranspose":
            nodes.append(node)
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
            raise RuntimeError(f"unsupported ConvTranspose at {node.name}: {contract}")
        x_name, weight_name, bias_name = node.input
        if weight_name not in initializers or bias_name not in initializers:
            raise RuntimeError(f"missing initializer at {node.name}")
        if consumers.get(weight_name) != 1 or consumers.get(bias_name) != 1:
            raise RuntimeError(f"shared initializer at {node.name}")
        w = numpy_helper.to_array(initializers[weight_name])
        b = numpy_helper.to_array(initializers[bias_name])
        if w.dtype != np.float16 or b.dtype != np.float16:
            raise RuntimeError(f"FP16 weights required at {node.name}: {w.dtype}, {b.dtype}")
        if w.ndim != 4 or w.shape[2:] != (2, 2) or b.shape != (w.shape[1],):
            raise RuntimeError(f"unexpected weights at {node.name}: {w.shape}, {b.shape}")
        input_shape = dimensions(info[x_name])
        output_shape = dimensions(info[node.output[0]])
        n, cin, h, width = input_shape
        n2, cout, out_h, out_w = output_shape
        if n != n2 or cin != w.shape[0] or cout != w.shape[1] or out_h != 2 * h or out_w != 2 * width:
            raise RuntimeError(f"shape mismatch at {node.name}: {input_shape} -> {output_shape}")
        conv_w = np.empty((4 * cout, cin, 1, 1), dtype=np.float16)
        conv_b = np.empty((4 * cout,), dtype=np.float16)
        for ky in range(2):
            for kx in range(2):
                start = (ky * 2 + kx) * cout
                conv_w[start : start + cout, :, 0, 0] = w[:, :, ky, kx].T
                conv_b[start : start + cout] = b
        base = f"phase11_fp16_deconv_{index}"
        new_weight, new_bias = base + "_weight", base + "_bias"
        shape1, shape2 = base + "_shape1", base + "_shape2"
        conv_out, block_out, transposed = base + "_conv", base + "_blocks", base + "_transposed"
        added_initializers.extend(
            [
                numpy_helper.from_array(conv_w, new_weight),
                numpy_helper.from_array(conv_b, new_bias),
                numpy_helper.from_array(np.asarray([n, 2, 2, cout, h, width], np.int64), shape1),
                numpy_helper.from_array(np.asarray([n, cout, out_h, out_w], np.int64), shape2),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Conv", [x_name, new_weight, new_bias], [conv_out], name=base + "/Conv1x1", kernel_shape=[1, 1]),
                helper.make_node("Reshape", [conv_out, shape1], [block_out], name=base + "/ReshapeBlocks", allowzero=0),
                helper.make_node("Transpose", [block_out], [transposed], name=base + "/TransposeBlocks", perm=[0, 3, 4, 1, 5, 2]),
                helper.make_node("Reshape", [transposed, shape2], list(node.output), name=base + "/ReshapeOutput", allowzero=0),
            ]
        )
        removed.update((weight_name, bias_name))
        changed.append(
            {
                "node_index_after_layernorm_rewrite": index,
                "node_name": node.name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "weight_dtype": str(w.dtype),
                "old_weight_shape": list(w.shape),
                "new_weight_shape": list(conv_w.shape),
            }
        )
    if len(changed) != 3:
        raise RuntimeError(f"expected 3 ConvTranspose nodes, found {len(changed)}")
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + added_initializers)
    return changed


def add_barrier(model: onnx.ModelProto) -> dict:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    info = {
        item.name: item
        for item in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
    }
    matches = [node for node in model.graph.node if node.name == BARRIER_NODE]
    if len(matches) != 1 or matches[0].output[0] != BARRIER_TENSOR:
        raise RuntimeError("fpn4 MaxPool barrier node contract mismatch")
    if [item.name for item in model.graph.output] != ["logits"]:
        raise RuntimeError("source output contract drift")
    if BARRIER_TENSOR not in info:
        raise RuntimeError("barrier value_info unavailable")
    model.graph.output.append(copy.deepcopy(info[BARRIER_TENSOR]))
    return {
        "node_name": BARRIER_NODE,
        "tensor": BARRIER_TENSOR,
        "shape": dimensions(info[BARRIER_TENSOR]),
        "elem_type": int(info[BARRIER_TENSOR].type.tensor_type.elem_type),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve(strict=True)
    if source.stat().st_size != SOURCE_SIZE or sha256(source) != SOURCE_SHA256:
        raise RuntimeError("frozen FP16 source identity mismatch")
    if args.output.resolve() == source:
        raise RuntimeError("source and output must differ")
    model = onnx.load(str(source), load_external_data=False)
    before = {"nodes": len(model.graph.node), "initializers": len(model.graph.initializer)}
    layernorm = rewrite_layernorm(model)
    deconv = rewrite_convtranspose(model)
    barrier = add_barrier(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output), save_as_external_data=False)
    reloaded = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(reloaded)
    outputs = [item.name for item in reloaded.graph.output]
    remaining_ln = sum(node.op_type == "LayerNormalization" for node in reloaded.graph.node)
    remaining_deconv = sum(node.op_type == "ConvTranspose" for node in reloaded.graph.node)
    if remaining_ln or remaining_deconv or outputs != ["logits", BARRIER_TENSOR]:
        raise RuntimeError("saved candidate contract mismatch")
    report = {
        "status": "created_static_pass",
        "variant": "fp16_internal_with_fp32_layernorm_statistic_islands_and_single_fpn4_maxpool_barrier",
        "source": {"path": str(source), "size": source.stat().st_size, "sha256": SOURCE_SHA256},
        "candidate": {
            "path": str(args.output.resolve()),
            "size": args.output.stat().st_size,
            "sha256": sha256(args.output),
            "outputs": outputs,
        },
        "before": before,
        "after": {"nodes": len(reloaded.graph.node), "initializers": len(reloaded.graph.initializer)},
        "layernorm": {"count": len(layernorm), "remaining": remaining_ln, "changes": layernorm},
        "convtranspose": {"count": len(deconv), "remaining": remaining_deconv, "changes": deconv},
        "barrier": barrier,
        "claims": {
            "static_graph_valid": True,
            "cpu_numeric_equivalence": False,
            "migraphx_numeric_equivalence": False,
            "task_accuracy_90": False,
            "performance": False,
            "deployment_complete": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
