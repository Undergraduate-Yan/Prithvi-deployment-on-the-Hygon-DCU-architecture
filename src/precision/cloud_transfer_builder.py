#!/usr/bin/env python3
'Research implementation: cloud transfer builder.'
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FLOOD_TRANSFER_MAP_SHA256 = "4391d072f0879b2655047346b0850ad1b6ad56a821041c6dc7f29bb43f671be7"
BASE_BUILDER_SHA256 = "44943ff7dbd02e917b1cf53a19bdd34df36f316f22e52008fbc02e1110921f4c"
FP16_BLOCKS = (0, 8, 10, 14, 15, 17, 18, 19, 20)
INT8_BLOCKS = (1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 21, 22, 23)
BLOCK_RE = re.compile(r"/model/encoder/blocks\.(\d+)(?:/|$)")
ATTENTION_CAST = re.compile(
    r"^/model/encoder/blocks\.(\d+)/attn/Cast(?:_1)?$"
)
HEAD_TOKENS = (
    "/model/neck/",
    "/model/decoder/",
    "/model/head/",
    "/model/Resize",
    "phase11_fp16_deconv_",
)
BARRIER_TENSOR = "/model/decoder/fpn4/fpn4.0/MaxPool_output_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized-source", type=Path, required=True)
    parser.add_argument("--quantization-report", type=Path, required=True)
    parser.add_argument("--quantization-parameters", type=Path, required=True)
    parser.add_argument("--flood-transfer-map", type=Path, required=True)
    parser.add_argument("--cloud-topology", type=Path, required=True)
    parser.add_argument("--base-builder", type=Path, required=True)
    parser.add_argument("--fp16-compat-tool", type=Path, required=True)
    parser.add_argument("--fp16-resize-tool", type=Path, required=True)
    parser.add_argument("--synthetic-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "official"), default="official")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
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
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module(path: Path, name: str, expected_sha256: str | None = None) -> Any:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"frozen dependency identity mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def value_contract(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "element_type": int(tensor.elem_type),
        "shape": [int(dim.dim_value) for dim in tensor.shape.dim],
    }


def cast_target(node: Any) -> int:
    matches = [item for item in node.attribute if item.name == "to"]
    if len(matches) != 1:
        raise RuntimeError(f"Cast has no unique target: {node.name}")
    return int(matches[0].i)


def set_cast_target(node: Any, target: int) -> None:
    matches = [item for item in node.attribute if item.name == "to"]
    if len(matches) != 1:
        raise RuntimeError(f"Cast has no unique target: {node.name}")
    matches[0].i = target


def block_from_text(*values: str) -> int | None:
    blocks = {
        int(match.group(1))
        for value in values
        for match in BLOCK_RE.finditer(value or "")
    }
    return next(iter(blocks)) if len(blocks) == 1 else None


def node_block(node: Any) -> int | None:
    # A block's first operators legitimately consume the previous block's
    # boundary tensor.  Prefer the owning node name; use tensor names only for
    # generated Q/DQ/Cast nodes whose own names carry no block prefix.
    named = {
        int(match.group(1)) for match in BLOCK_RE.finditer(node.name or "")
    }
    if len(named) == 1:
        return next(iter(named))
    return block_from_text(*node.input, *node.output)


def graph_node_ownership(model: onnx.ModelProto) -> dict[str, int | None]:
    """Assign generated Q/DQ nodes through one-hop producer/consumer lineage."""
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    result: dict[str, int | None] = {}
    for node in model.graph.node:
        direct = node_block(node)
        if direct is not None:
            result[node.name] = direct
            continue
        adjacent = []
        adjacent.extend(
            producer
            for name in node.input
            if (producer := producers.get(name)) is not None
        )
        adjacent.extend(
            consumer
            for name in node.output
            for consumer in consumers.get(name, [])
        )
        blocks = {
            block
            for neighbor in adjacent
            if (block := node_block(neighbor)) is not None
        }
        result[node.name] = next(iter(blocks)) if len(blocks) == 1 else None
    return result


def attributes(node: Any) -> dict[str, Any]:
    return {
        item.name: helper.get_attribute_value(item)
        for item in node.attribute
    }


def lower_mixed_layernorm(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Lower LN with full FP32 islands and FP16 outputs outside INT8 blocks."""
    replacement = []
    rows = []
    existing = {item.name for item in model.graph.initializer}
    for ordinal, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization":
            replacement.append(node)
            continue
        attrs = attributes(node)
        axis = int(attrs.get("axis", -1))
        epsilon = float(attrs.get("epsilon", 1e-5))
        stash_type = int(attrs.get("stash_type", TensorProto.FLOAT))
        if axis != -1 or stash_type != TensorProto.FLOAT or len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(f"unsupported mixed LayerNormalization: {node.name}: {attrs}")
        # LayerNorm ownership follows its node path, never its consumed tensor:
        # the single neck normalization consumes block-23 output but belongs to
        # the new cloud Head.  All 48 encoder LayerNorm names carry one block.
        named_blocks = {
            int(match.group(1)) for match in BLOCK_RE.finditer(node.name or "")
        }
        if len(named_blocks) > 1:
            raise RuntimeError(f"ambiguous LayerNormalization node path: {node.name}")
        block = next(iter(named_blocks)) if named_blocks else None
        mode = "fp32_int8_block" if block in INT8_BLOCKS else "fp16_io_fp32_statistics"
        base = f"phase7f_cloud_mp_ln_{ordinal:04d}"
        epsilon_name = base + "_epsilon_fp32"
        if epsilon_name in existing:
            raise RuntimeError(f"initializer collision: {epsilon_name}")
        existing.add(epsilon_name)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name)
        )
        x, scale, bias = node.input
        y = node.output[0]
        x32 = base + "_x32"
        scale32 = base + "_scale32"
        bias32 = base + "_bias32"
        mean = base + "_mean"
        centered = base + "_centered"
        squared = base + "_squared"
        variance = base + "_variance"
        variance_epsilon = base + "_variance_epsilon"
        denominator = base + "_denominator"
        normalized32 = base + "_normalized_fp32"
        scaled = base + "_scaled"
        y32 = base + "_y32"
        prefix = []
        statistic_input = x
        if mode == "fp16_io_fp32_statistics":
            prefix.append(
                helper.make_node(
                    "Cast", [x], [x32], name=base + "/CastInputFP32", to=TensorProto.FLOAT
                )
            )
            prefix.extend(
                [
                    helper.make_node("Cast", [scale], [scale32], name=base + "/CastScaleFP32", to=TensorProto.FLOAT),
                    helper.make_node("Cast", [bias], [bias32], name=base + "/CastBiasFP32", to=TensorProto.FLOAT),
                ]
            )
            statistic_input = x32
        group = [
            helper.make_node("ReduceMean", [statistic_input], [mean], name=base + "/ReduceMean", axes=[-1], keepdims=1),
            helper.make_node("Sub", [statistic_input, mean], [centered], name=base + "/Center"),
            helper.make_node("Mul", [centered, centered], [squared], name=base + "/Square"),
            helper.make_node("ReduceMean", [squared], [variance], name=base + "/Variance", axes=[-1], keepdims=1),
            helper.make_node("Add", [variance, epsilon_name], [variance_epsilon], name=base + "/AddEpsilon"),
            helper.make_node("Sqrt", [variance_epsilon], [denominator], name=base + "/Sqrt"),
            helper.make_node("Div", [centered, denominator], [normalized32], name=base + "/Normalize"),
        ]
        if mode == "fp16_io_fp32_statistics":
            group.extend(
                [
                    helper.make_node("Mul", [normalized32, scale32], [scaled], name=base + "/ScaleFP32"),
                    helper.make_node("Add", [scaled, bias32], [y32], name=base + "/BiasFP32"),
                    helper.make_node("Cast", [y32], [y], name=base + "/CastOutputFP16", to=TensorProto.FLOAT16),
                ]
            )
        else:
            group.extend(
                [
                    helper.make_node("Mul", [normalized32, scale], [scaled], name=base + "/ScaleFP32"),
                    helper.make_node("Add", [scaled, bias], [y], name=base + "/BiasFP32"),
                ]
            )
        replacement.extend(prefix + group)
        rows.append(
            {
                "source_node": node.name,
                "source_ordinal": ordinal,
                "block": block,
                "mode": mode,
                "axis": axis,
                "epsilon": epsilon,
                "statistics_precision": "FP32",
                "affine_precision": "FP32",
            }
        )
    if len(rows) != 49:
        raise RuntimeError(f"expected 49 LayerNormalization groups, found {len(rows)}")
    mode_counts = Counter(row["mode"] for row in rows)
    block_counts = Counter(row["block"] for row in rows)
    if mode_counts["fp32_int8_block"] != 30:
        raise RuntimeError(
            "expected 30 INT8-block FP32 LayerNorm groups; "
            f"modes={dict(mode_counts)} blocks={dict(block_counts)}"
        )
    if mode_counts["fp16_io_fp32_statistics"] != 19:
        raise RuntimeError(
            "expected 19 FP16/Head LayerNorm groups; "
            f"modes={dict(mode_counts)} blocks={dict(block_counts)}"
        )
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    return rows


def tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    result: dict[str, int] = {}
    for collection in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info):
        for value in collection:
            if value.type.HasField("tensor_type"):
                result[value.name] = value.type.tensor_type.elem_type
    for item in inferred.graph.initializer:
        result[item.name] = item.data_type
    return result


def graph_stats(model: onnx.ModelProto) -> dict[str, Any]:
    counts = Counter(node.op_type for node in model.graph.node)
    initializer_types = Counter(
        TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer
    )
    return {
        "node_count": len(model.graph.node),
        "operation_counts": dict(sorted(counts.items())),
        "initializer_type_counts": dict(sorted(initializer_types.items())),
        "graph_input_count": len(model.graph.input),
        "graph_output_count": len(model.graph.output),
    }


def set_value_type(value: Any, elem_type: int) -> None:
    value.type.tensor_type.elem_type = elem_type


def boundary_elem_type(name: str) -> int:
    if name in {"input", "logits"}:
        return TensorProto.FLOAT
    if name == BARRIER_TENSOR:
        return TensorProto.FLOAT16
    block = block_from_text(name)
    if block is None:
        raise RuntimeError(f"cannot assign mixed boundary precision: {name}")
    return TensorProto.FLOAT16 if block in FP16_BLOCKS else TensorProto.FLOAT


def inject_mixed_boundaries(model: onnx.ModelProto, topology: dict[str, Any]) -> list[dict[str, Any]]:
    # A topology cut inherits the type of the executable tensor at that exact
    # point, not merely the nominal precision of the block named in the tensor.
    # Mixed conversion may insert a Cast at a cut (for example at the output of
    # an INT8 block feeding an FP16 path), so block-name inference can create a
    # contradictory value_info annotation.  Infer once from the checked graph
    # and use that executable type as the internal Session contract; the public
    # FP32 Runner contract is enforced after extraction.
    inferred_types = tensor_types(model)
    known = {
        value.name: value
        for collection in (model.graph.input, model.graph.output, model.graph.value_info)
        for value in collection
    }
    produced = {name for node in model.graph.node for name in node.output}
    consumed = {name for node in model.graph.node for name in node.input}
    changed = []
    for row in topology["topology"]["sessions"]:
        for item in row["inputs"] + row["outputs"]:
            name = item["name"]
            elem_type = inferred_types.get(name)
            if elem_type is None:
                elem_type = boundary_elem_type(name)
            if name in known:
                before = int(known[name].type.tensor_type.elem_type)
                set_value_type(known[name], elem_type)
                action = "verified" if before == elem_type else "corrected"
            else:
                if name not in produced and name not in consumed:
                    raise RuntimeError(f"mixed boundary absent from graph: {name}")
                value = helper.make_tensor_value_info(name, elem_type, item["shape"])
                model.graph.value_info.append(value)
                known[name] = value
                before = None
                action = "injected"
            changed.append(
                {
                    "name": name,
                    "shape": item["shape"],
                    "elem_type": TensorProto.DataType.Name(elem_type),
                    "before_elem_type": TensorProto.DataType.Name(before) if before else None,
                    "action": action,
                }
            )
    return changed


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    expected_status = "SMOKE_PASS" if args.mode == "smoke" else "PASS"
    quant_report = json.loads(args.quantization_report.read_text(encoding="utf-8"))
    if (
        quant_report.get("status") != expected_status
        or quant_report.get("mode") != args.mode
        or not quant_report.get("checks")
        or not all(quant_report["checks"].values())
    ):
        raise RuntimeError("cloud MP quantization report gate failed")
    quantized_source = identity(args.quantized_source)
    if (
        quantized_source["sha256"] != quant_report.get("output_onnx", {}).get("sha256")
        or quantized_source["size_bytes"] != quant_report.get("output_onnx", {}).get("size_bytes")
    ):
        raise RuntimeError("cloud MP quantized ONNX identity mismatch")
    quant_parameters = identity(args.quantization_parameters)
    if (
        quant_parameters["sha256"] != quant_report.get("quantization_parameters", {}).get("sha256")
        or quant_parameters["size_bytes"] != quant_report.get("quantization_parameters", {}).get("size_bytes")
    ):
        raise RuntimeError("cloud MP quantization-parameter identity mismatch")
    transfer_map = json.loads(args.flood_transfer_map.read_text(encoding="utf-8"))
    if sha256_file(args.flood_transfer_map) != FLOOD_TRANSFER_MAP_SHA256:
        raise RuntimeError("frozen flood transfer-map identity mismatch")
    if (
        tuple(transfer_map.get("backbone", {}).get("fp16_blocks", ())) != FP16_BLOCKS
        or tuple(transfer_map.get("backbone", {}).get("int8_qdq_blocks", ())) != INT8_BLOCKS
    ):
        raise RuntimeError("frozen flood bit-width map drift")
    topology = json.loads(args.cloud_topology.read_text(encoding="utf-8"))
    if (
        topology.get("status") != "MAPPED_FROM_FROZEN_FLOOD_BOUNDARIES"
        or topology.get("session_count") != 13
    ):
        raise RuntimeError("cloud RCS13 topology gate failed")
    base = load_module(args.base_builder.resolve(strict=True), "frozen_rcs13_builder", BASE_BUILDER_SHA256)
    compat_tool = load_module(args.fp16_compat_tool.resolve(strict=True), "frozen_fp16_compat_tool")
    resize_tool = load_module(args.fp16_resize_tool.resolve(strict=True), "frozen_fp16_resize_tool")

    import onnxconverter_common
    from onnxconverter_common import float16

    if onnxconverter_common.__version__ != "1.16.0":
        raise RuntimeError(f"onnxconverter-common version drift: {onnxconverter_common.__version__}")
    model = onnx.load(str(args.quantized_source.resolve(strict=True)), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    before_contract = {
        "inputs": [value_contract(item) for item in model.graph.input],
        "outputs": [value_contract(item) for item in model.graph.output],
    }
    expected_contract = {
        "inputs": [{"name": "input", "element_type": TensorProto.FLOAT, "shape": [1, 6, 224, 224]}],
        "outputs": [{"name": "logits", "element_type": TensorProto.FLOAT, "shape": [1, 4, 224, 224]}],
    }
    if before_contract != expected_contract:
        raise RuntimeError("quantized cloud external contract drift")
    source_ownership = graph_node_ownership(model)
    blocked_nodes = [
        node.name
        for node in model.graph.node
        if source_ownership.get(node.name) in INT8_BLOCKS
    ]
    if not blocked_nodes or any(not name for name in blocked_nodes):
        raise RuntimeError("INT8 block node inventory is empty or unnamed")
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=5.96e-08,
        max_finite_val=65504.0,
        node_block_list=blocked_nodes,
    )
    attention_patches = []
    for node in converted.graph.node:
        match = ATTENTION_CAST.fullmatch(node.name)
        if not match or int(match.group(1)) not in FP16_BLOCKS:
            continue
        before = cast_target(node)
        set_cast_target(node, TensorProto.FLOAT16)
        attention_patches.append(
            {"node_name": node.name, "target_before": before, "target_after": TensorProto.FLOAT16}
        )
    if len(attention_patches) != 18:
        raise RuntimeError(f"expected 18 FP16-block attention Cast patches, got {len(attention_patches)}")

    # onnxconverter-common converts cached internal value_info globally even
    # for node-blocked QDQ regions.  Those annotations are not executable graph
    # state and can contradict the preserved FP32 DequantizeLinear outputs.
    # Drop them and let ONNX infer fresh types from nodes and initializers.
    stale_value_info_removed = len(converted.graph.value_info)
    del converted.graph.value_info[:]

    layernorm_rows = lower_mixed_layernorm(converted)
    convtranspose_rows = compat_tool.rewrite_convtranspose(converted)
    resize_rows, resize_before, resize_after = resize_tool.rewrite_head(converted)
    if len(convtranspose_rows) != 3 or len(resize_rows) != 11:
        raise RuntimeError("cloud Head compatibility/FP16 rewrite count drift")
    after_contract = {
        "inputs": [value_contract(item) for item in converted.graph.input],
        "outputs": [value_contract(item) for item in converted.graph.output],
    }
    if after_contract != before_contract:
        raise RuntimeError("mixed conversion changed FP32 external I/O")
    onnx.checker.check_model(converted, full_check=True)

    types = tensor_types(converted)
    converted_ownership = graph_node_ownership(converted)
    qdq_counts = Counter()
    qdq_unmapped = []
    for node in converted.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        block = converted_ownership.get(node.name)
        if block is None:
            qdq_unmapped.append(node.name)
        else:
            qdq_counts[block] += 1
    head_nodes = [node for node in converted.graph.node if any(token in node.name for token in HEAD_TOKENS)]
    head_convs = [node for node in head_nodes if node.op_type == "Conv"]
    head_resizes = [node for node in head_nodes if node.op_type == "Resize"]
    head_qdq = [node.name for node in head_nodes if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]
    head_fp16_failures = [
        node.name
        for node in head_convs + head_resizes
        if not node.output or types.get(node.output[0]) != TensorProto.FLOAT16
    ]
    mixed_stats = graph_stats(converted)
    checks = {
        "quantization_parent_gate_pass": True,
        "frozen_flood_bit_width_map_exact": True,
        "cloud_specific_quantization_parameters_linked": True,
        "all_15_int8_blocks_retain_qdq": all(qdq_counts[block] > 0 for block in INT8_BLOCKS),
        "all_9_fp16_blocks_qdq_free": all(qdq_counts[block] == 0 for block in FP16_BLOCKS),
        "all_qdq_nodes_mapped": not qdq_unmapped,
        "int8_initializers_positive": mixed_stats["initializer_type_counts"].get("INT8", 0) > 0,
        "float16_initializers_positive": mixed_stats["initializer_type_counts"].get("FLOAT16", 0) > 0,
        "layernorm_groups_49": len(layernorm_rows) == 49,
        "int8_block_layernorm_fp32_groups_30": sum(row["mode"] == "fp32_int8_block" for row in layernorm_rows) == 30,
        "fp16_and_head_layernorm_groups_19": sum(row["mode"] == "fp16_io_fp32_statistics" for row in layernorm_rows) == 19,
        "cloud_head_conv_positive": len(head_convs) > 0,
        "cloud_head_resize_count_11": len(head_resizes) == 11,
        "cloud_head_conv_and_resize_fp16": not head_fp16_failures,
        "cloud_head_qdq_zero": not head_qdq,
        "convtranspose_rewritten_3": len(convtranspose_rows) == 3,
        "external_fp32_io_unchanged": after_contract == before_contract,
        "monolithic_onnx_full_checker_pass": True,
        "deployment_validation_payload_access_count_zero": True,
        "formal_test_payload_access_count_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cloud-MP mixed monolithic gate failed: {checks}")

    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "model"
    segment_root = args.output_root / "models"
    feeds_root = args.output_root / "compile_feeds"
    model_root.mkdir()
    segment_root.mkdir()
    feeds_root.mkdir()
    monolithic_path = model_root / "cloud_mp_transfer_mixed.onnx"
    temporary = monolithic_path.with_suffix(".onnx.tmp")
    onnx.save(converted, str(temporary), save_as_external_data=False)
    reloaded = onnx.load(str(temporary.resolve(strict=True)), load_external_data=False)
    onnx.checker.check_model(reloaded, full_check=True)
    os.replace(temporary, monolithic_path)

    boundary_rows = inject_mixed_boundaries(reloaded, topology)
    extractor = onnx.utils.Extractor(reloaded)
    sessions = []
    total_boundary_repairs = 0
    for row in topology["topology"]["sessions"]:
        ordinal = int(row["ordinal"])
        session_id = f"Cloud_RCS13_MP_Transfer_{ordinal:02d}"
        input_names = [item["name"] for item in row["inputs"]]
        output_names = [item["name"] for item in row["outputs"]]
        extracted = extractor.extract_model(input_names, output_names)
        repairs = base.enforce_fp32_boundaries(extracted, session_id)
        total_boundary_repairs += len(repairs)
        onnx.checker.check_model(extracted, full_check=True)
        model_path = segment_root / f"session_{ordinal:02d}.onnx"
        temporary_segment = model_path.with_suffix(".onnx.tmp")
        onnx.save(extracted, str(temporary_segment), save_as_external_data=False)
        verified = onnx.load(str(temporary_segment.resolve(strict=True)), load_external_data=False)
        onnx.checker.check_model(verified, full_check=True)
        os.replace(temporary_segment, model_path)
        feed_path = feeds_root / f"session_{ordinal:02d}.npz"
        np.savez(
            feed_path,
            **{item["name"]: np.zeros(item["shape"], dtype=np.float32) for item in row["inputs"]},
        )
        session_qdq = Counter(node.op_type for node in verified.graph.node)
        sessions.append(
            {
                **copy.deepcopy(row),
                "session_id": session_id,
                "model": identity(model_path),
                "compile_feeds": identity(feed_path),
                "boundary_repairs": repairs,
                "graph_stats": base.graph_stats(verified),
                "qdq_nodes": session_qdq["QuantizeLinear"] + session_qdq["DequantizeLinear"],
                "onnx_checker": "PASS_FULL_TWICE",
            }
        )
        del verified
        del extracted
    del extractor
    del reloaded

    synthetic = np.load(args.synthetic_input.resolve(strict=True), allow_pickle=False)
    if synthetic.shape != (1, 6, 224, 224) or synthetic.dtype != np.float32:
        raise RuntimeError("synthetic compile input contract drift")
    manifest = {
        "schema": "phase7f_cloud_rcs13_mp_transfer_manifest_v1",
        "status": "BUILT_ONNX_CHECKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "Cloud-RCS13-MP-Transfer",
        "mode": args.mode,
        "hardware": "海光 K100 AI 加速卡",
        "session_count": 13,
        "source_quantized_onnx": quantized_source,
        "quantization_report": identity(args.quantization_report),
        "quantization_parameters": quant_parameters,
        "frozen_flood_transfer_map": identity(args.flood_transfer_map),
        "source_topology": identity(args.cloud_topology),
        "monolithic_mixed_onnx": identity(monolithic_path),
        "external_contract": topology["external_contract"],
        "precision_map": {
            "fp16_blocks": list(FP16_BLOCKS),
            "int8_qdq_blocks": list(INT8_BLOCKS),
            "head": "new cloud FP16 Head; FP32 LayerNorm statistics; FP32 logits",
            "external_and_inter_session_io": "FP32",
        },
        "topology": {"sessions": sessions},
        "conversion": {
            "onnxconverter_common_version": onnxconverter_common.__version__,
            "int8_blocked_node_count": len(blocked_nodes),
            "stale_internal_value_info_removed_before_fresh_inference": stale_value_info_removed,
            "fp16_attention_cast_patches": attention_patches,
            "layernorm_rewrites": layernorm_rows,
            "convtranspose_rewrites": convtranspose_rows,
            "head_resize_rewrites": resize_rows,
            "head_resize_before": resize_before,
            "head_resize_after": resize_after,
            "boundary_type_injection": boundary_rows,
            "public_boundary_repairs": total_boundary_repairs,
        },
        "graph_stats": mixed_stats,
        "head_audit": {
            "head_node_count": len(head_nodes),
            "head_conv_count": len(head_convs),
            "head_resize_count": len(head_resizes),
            "head_qdq_nodes": head_qdq,
            "head_fp16_failures": head_fp16_failures,
            "conclusion": "new four-class cloud Head independently audited as FP16 Conv/Resize with FP32 LayerNorm statistics and no QDQ",
        },
        "checks": checks,
        "data_firewall": {
            "calibration_payload_access_count": 1280,
            "quantization_calibration_sample_count": int(quant_report["sample_count"]),
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
            "formal_test_used": False,
        },
    }
    manifest_path = args.output_root / "cloud_rcs13_mp_transfer_manifest.json"
    write_json(manifest_path, manifest)
    with (args.output_root / "cloud_mp_transfer_fp32_islands.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layernorm_rows[0]))
        writer.writeheader()
        writer.writerows(layernorm_rows)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "mode": args.mode,
                "sessions": len(sessions),
                "total_boundary_repairs": total_boundary_repairs,
                "qdq_counts": dict(sorted(qdq_counts.items())),
                "head_conv_count": len(head_convs),
                "head_resize_count": len(head_resizes),
                "formal_test_payload_access_count": 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
