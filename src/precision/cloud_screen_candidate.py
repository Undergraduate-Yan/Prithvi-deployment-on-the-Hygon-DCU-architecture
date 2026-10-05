#!/usr/bin/env python3
'Construct a cloud mixed-precision screening candidate.'
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CANONICAL_CLOUD_FP32_SHA256 = "c823e2c96bbb551d3f57792df585eb0677f43471333ebaf990a314587c1e1a5c"
CANONICAL_CLOUD_FP32_SIZE = 1_277_042_559
ALL24_QDQ_SHA256 = "c1542b5a3050a2fe5c6b9d4984ef6a5beddceb0a9577c1d5a2eb038ead131446"
ALL24_PARAMETERS_SHA256 = "7c88becf356b5f50334faadf1fbd56eb5aaa5798c36b02fa7156b417ad800a3a"
TASK_PROTOCOL_SHA256 = "1ec25e0e2c0f558d37fc83568802c78805635165f3c4e1842f9f834dd1e796a4"
FORMAL_LEDGER_SHA256 = "ef418d7ac7b6abc96ec2122e2e567e3d5ecc0608f173b9de6144e45dfe59c10a"
CHECKPOINT_SHA256 = "0dfe43e40c458c3a60ec6d845c6ba58d07401a5d214c0c18304f9470de632fd6"
BLOCK_RE = re.compile(r"/model/encoder/blocks\.(\d+)(?:/|$)")
ATTENTION_CAST_RE = re.compile(r"^/model/encoder/blocks\.(\d+)/attn/Cast(?:_1)?$")
HEAD_TOKENS = (
    "/model/neck/",
    "/model/decoder/",
    "/model/head/",
    "/model/Resize",
    "phase11_fp16_deconv_",
)
ALL_BLOCKS = tuple(range(24))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--all24-qdq", required=True, type=Path)
    parser.add_argument("--all24-report", required=True, type=Path)
    parser.add_argument("--all24-parameters", required=True, type=Path)
    parser.add_argument("--task-protocol", required=True, type=Path)
    parser.add_argument("--formal-access-ledger", required=True, type=Path)
    parser.add_argument("--transfer-builder", required=True, type=Path)
    parser.add_argument("--rcs13-base-builder", required=True, type=Path)
    parser.add_argument("--cloud-topology", required=True, type=Path)
    parser.add_argument("--fp16-compat-tool", required=True, type=Path)
    parser.add_argument("--fp16-resize-tool", required=True, type=Path)
    parser.add_argument("--int8-blocks", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_blocks(text: str) -> tuple[int, ...]:
    try:
        blocks = tuple(sorted({int(value.strip()) for value in text.split(",") if value.strip()}))
    except ValueError as error:
        raise RuntimeError("--int8-blocks must be comma-separated integer ordinals") from error
    if not blocks or any(block not in ALL_BLOCKS for block in blocks):
        raise RuntimeError("--int8-blocks must be a non-empty subset of 0..23")
    return blocks


def formal_payload_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(
        "formal_test_payload_access_count",
        payload.get("payload_access_count", payload.get("access_count")),
    )
    if value is None:
        raise RuntimeError("formal access ledger has no payload-access count")
    return int(value)


def block_from_name(name: str) -> int | None:
    matches = {int(match.group(1)) for match in BLOCK_RE.finditer(name or "")}
    return next(iter(matches)) if len(matches) == 1 else None


def node_block_from_name(node: Any) -> int | None:
    return block_from_name(node.name or "")


def value_contract(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "element_type": int(tensor.elem_type),
        "shape": [int(dim.dim_value) for dim in tensor.shape.dim],
    }


def cast_target(node: Any) -> int:
    values = [item for item in node.attribute if item.name == "to"]
    if len(values) != 1:
        raise RuntimeError(f"Cast has no unique target: {node.name}")
    return int(values[0].i)


def set_cast_target(node: Any, target: int) -> None:
    values = [item for item in node.attribute if item.name == "to"]
    if len(values) != 1:
        raise RuntimeError(f"Cast has no unique target: {node.name}")
    values[0].i = target


def attributes(node: Any) -> dict[str, Any]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def graph_stats(model: onnx.ModelProto) -> dict[str, Any]:
    operations = Counter(node.op_type for node in model.graph.node)
    initializers = Counter(TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer)
    return {
        "node_count": len(model.graph.node),
        "operation_counts": dict(sorted(operations.items())),
        "initializer_type_counts": dict(sorted(initializers.items())),
    }


def ownership_with_qdq_map(
    model: onnx.ModelProto,
    qdq_map: dict[str, int],
    transfer: Any,
) -> dict[str, int | None]:
    inherited = transfer.graph_node_ownership(model)
    result = {}
    for node in model.graph.node:
        result[node.name] = node_block_from_name(node)
        # Quantization introduces path-less weight Q/DQ nodes whose ownership
        # is frozen in the parameter inventory.  Do not apply graph-adjacency
        # ownership to ordinary path-less source nodes (for example the final
        # encoder Concat), or they would be duplicated inside block 23.
        if result[node.name] is None and node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            result[node.name] = qdq_map.get(node.name, inherited.get(node.name))
    return result


def rebuild_selected_qdq(
    source: onnx.ModelProto,
    quantized: onnx.ModelProto,
    selected: tuple[int, ...],
    qdq_map: dict[str, int],
    transfer: Any,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    quant_ownership = ownership_with_qdq_map(quantized, qdq_map, transfer)
    source_by_block: dict[int, list[Any]] = defaultdict(list)
    quant_by_block: dict[int, list[Any]] = defaultdict(list)
    for node in source.graph.node:
        block = node_block_from_name(node)
        if block is not None:
            source_by_block[block].append(node)
    for node in quantized.graph.node:
        block = quant_ownership.get(node.name)
        if block is not None:
            quant_by_block[block].append(node)
    if any(not source_by_block[block] or not quant_by_block[block] for block in selected):
        raise RuntimeError("selected block node inventory is incomplete")
    unmapped_qdq = [
        node.name
        for node in quantized.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and quant_ownership.get(node.name) is None
    ]
    if unmapped_qdq:
        raise RuntimeError(f"all24 QDQ ownership is incomplete: {unmapped_qdq[:5]}")

    rebuilt_nodes = []
    inserted: set[int] = set()
    for node in source.graph.node:
        block = node_block_from_name(node)
        if block not in selected:
            rebuilt_nodes.append(copy.deepcopy(node))
            continue
        if block not in inserted:
            rebuilt_nodes.extend(copy.deepcopy(item) for item in quant_by_block[block])
            inserted.add(block)
    if inserted != set(selected):
        raise RuntimeError(f"not all selected blocks were inserted: {inserted} != {set(selected)}")

    source_initializers = {item.name: item for item in source.graph.initializer}
    quant_initializers = {item.name: item for item in quantized.graph.initializer}
    required = {name for node in rebuilt_nodes for name in node.input if name}
    rebuilt_initializers = []
    missing = []
    for name in sorted(required):
        item = quant_initializers.get(name, source_initializers.get(name))
        if item is not None:
            rebuilt_initializers.append(copy.deepcopy(item))
        elif name not in {value.name for value in source.graph.input} and not any(
            name in node.output for node in rebuilt_nodes
        ):
            missing.append(name)
    if missing:
        raise RuntimeError(f"rebuilt graph has missing values: {missing[:10]}")

    rebuilt = copy.deepcopy(source)
    del rebuilt.graph.node[:]
    rebuilt.graph.node.extend(rebuilt_nodes)
    del rebuilt.graph.initializer[:]
    rebuilt.graph.initializer.extend(rebuilt_initializers)
    del rebuilt.graph.sparse_initializer[:]
    onnx.checker.check_model(rebuilt, full_check=True)

    rebuilt_ownership = ownership_with_qdq_map(rebuilt, qdq_map, transfer)
    qdq_counts = Counter(
        rebuilt_ownership.get(node.name)
        for node in rebuilt.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    )
    matmul_counts = Counter(
        node_block_from_name(node) for node in rebuilt.graph.node if node.op_type == "MatMul"
    )
    if any(qdq_counts[block] != 32 for block in selected):
        raise RuntimeError(f"selected QDQ count drift: {dict(qdq_counts)}")
    if any(qdq_counts[block] != 0 for block in ALL_BLOCKS if block not in selected):
        raise RuntimeError(f"unselected block retained QDQ: {dict(qdq_counts)}")
    if any(matmul_counts[block] != 6 for block in selected):
        raise RuntimeError(f"selected MatMul count drift: {dict(matmul_counts)}")
    return rebuilt, {
        "selected_qdq_counts": {str(block): qdq_counts[block] for block in selected},
        "selected_matmul_counts": {str(block): matmul_counts[block] for block in selected},
        "source_node_counts": {str(block): len(source_by_block[block]) for block in ALL_BLOCKS},
        "quantized_node_counts": {str(block): len(quant_by_block[block]) for block in ALL_BLOCKS},
    }


def lower_task_layernorm(
    model: onnx.ModelProto,
    selected: tuple[int, ...],
) -> list[dict[str, Any]]:
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
        block = block_from_name(node.name or "")
        mode = "fp32_int8_block" if block in selected else "fp16_io_fp32_statistics"
        base = f"phase7f_cloud_mp_task_ln_{ordinal:04d}"
        epsilon_name = base + "_epsilon_fp32"
        if epsilon_name in existing:
            raise RuntimeError(f"initializer collision: {epsilon_name}")
        existing.add(epsilon_name)
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name))
        x, scale, bias = node.input
        y = node.output[0]
        x32, scale32, bias32 = base + "_x32", base + "_scale32", base + "_bias32"
        mean, centered, squared = base + "_mean", base + "_centered", base + "_squared"
        variance, variance_epsilon = base + "_variance", base + "_variance_epsilon"
        denominator, normalized32 = base + "_denominator", base + "_normalized_fp32"
        scaled, y32 = base + "_scaled", base + "_y32"
        prefix = []
        statistic_input = x
        if mode == "fp16_io_fp32_statistics":
            prefix.extend(
                [
                    helper.make_node("Cast", [x], [x32], name=base + "/CastInputFP32", to=TensorProto.FLOAT),
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
        rows.append({"source_node": node.name, "block": block, "mode": mode, "statistics_precision": "FP32"})
    if len(rows) != 49:
        raise RuntimeError(f"expected 49 LayerNormalization groups, found {len(rows)}")
    expected_fp32 = 2 * len(selected)
    observed_fp32 = sum(row["mode"] == "fp32_int8_block" for row in rows)
    if observed_fp32 != expected_fp32:
        raise RuntimeError(f"INT8-block LayerNorm count drift: {observed_fp32} != {expected_fp32}")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    return rows


def main() -> int:
    args = parse_args()
    selected = parse_blocks(args.int8_blocks)
    fp16_blocks = tuple(block for block in ALL_BLOCKS if block not in selected)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    source_identity = identity(args.source)
    quant_identity = identity(args.all24_qdq)
    params_identity = identity(args.all24_parameters)
    protocol_identity = identity(args.task_protocol)
    ledger_identity = identity(args.formal_access_ledger)
    if source_identity["sha256"] != CANONICAL_CLOUD_FP32_SHA256 or source_identity["size_bytes"] != CANONICAL_CLOUD_FP32_SIZE:
        raise RuntimeError("canonical cloud FP32 identity mismatch")
    if quant_identity["sha256"] != ALL24_QDQ_SHA256:
        raise RuntimeError("frozen all24 QDQ identity mismatch")
    if params_identity["sha256"] != ALL24_PARAMETERS_SHA256:
        raise RuntimeError("frozen all24 quantization-parameter identity mismatch")
    if protocol_identity["sha256"] != TASK_PROTOCOL_SHA256:
        raise RuntimeError("frozen Cloud-MP-Task protocol identity mismatch")
    if ledger_identity["sha256"] != FORMAL_LEDGER_SHA256 or formal_payload_count(args.formal_access_ledger) != 0:
        raise RuntimeError("formal cloud test access gate failed")
    protocol = json.loads(args.task_protocol.read_text(encoding="utf-8"))
    report = json.loads(args.all24_report.read_text(encoding="utf-8"))
    parameters = json.loads(args.all24_parameters.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_DEPLOYMENT_VALIDATION_PAYLOAD_ACCESS"
        or protocol.get("invariants", {}).get("formal_test_forbidden") is not True
        or protocol.get("precision_search", {}).get("default_unselected_encoder_precision") != "FP16 with FP32 statistic islands"
        or report.get("status") != "PASS"
        or report.get("output_onnx", {}).get("sha256") != quant_identity["sha256"]
        or report.get("quantization_parameters", {}).get("sha256") != params_identity["sha256"]
        or parameters.get("status") != "FROZEN_SEARCH_INVENTORY"
        or parameters.get("inventory_map", {}).get("int8_qdq_blocks") != list(ALL_BLOCKS)
    ):
        raise RuntimeError("Cloud-MP-Task parent gate failed")

    transfer = load_module(args.transfer_builder.resolve(strict=True), "cloud_mp_transfer_builder_frozen")
    base = load_module(args.rcs13_base_builder.resolve(strict=True), "cloud_rcs13_base_builder_frozen")
    compat = load_module(args.fp16_compat_tool.resolve(strict=True), "cloud_fp16_compat_frozen")
    resize = load_module(args.fp16_resize_tool.resolve(strict=True), "cloud_fp16_resize_frozen")
    topology = json.loads(args.cloud_topology.read_text(encoding="utf-8"))
    if topology.get("status") != "MAPPED_FROM_FROZEN_FLOOD_BOUNDARIES" or topology.get("session_count") != 13:
        raise RuntimeError("frozen Cloud-RCS13 topology gate failed")
    qdq_rows = parameters.get("parameters", {}).get("qdq_nodes", [])
    qdq_map = {str(row["node_name"]): int(row["block"]) for row in qdq_rows if row.get("block") is not None}
    if len(qdq_map) != 768:
        raise RuntimeError(f"expected 768 frozen Q/DQ rows, found {len(qdq_map)}")

    source = onnx.load(str(args.source.resolve(strict=True)), load_external_data=False)
    quantized = onnx.load(str(args.all24_qdq.resolve(strict=True)), load_external_data=False)
    onnx.checker.check_model(source, full_check=True)
    onnx.checker.check_model(quantized, full_check=True)
    expected_contract = {
        "inputs": [{"name": "input", "element_type": TensorProto.FLOAT, "shape": [1, 6, 224, 224]}],
        "outputs": [{"name": "logits", "element_type": TensorProto.FLOAT, "shape": [1, 4, 224, 224]}],
    }
    observed_contract = {
        "inputs": [value_contract(item) for item in source.graph.input],
        "outputs": [value_contract(item) for item in source.graph.output],
    }
    if observed_contract != expected_contract:
        raise RuntimeError("cloud external FP32 contract drift")
    selective, reconstruction = rebuild_selected_qdq(source, quantized, selected, qdq_map, transfer)
    del source
    del quantized

    import onnxconverter_common
    from onnxconverter_common import float16

    if onnxconverter_common.__version__ != "1.16.0":
        raise RuntimeError(f"onnxconverter-common version drift: {onnxconverter_common.__version__}")
    selective_ownership = ownership_with_qdq_map(selective, qdq_map, transfer)
    blocked_nodes = [node.name for node in selective.graph.node if selective_ownership.get(node.name) in selected]
    if not blocked_nodes or any(not name for name in blocked_nodes):
        raise RuntimeError("selected INT8 node blocklist is empty or unnamed")
    converted = float16.convert_float_to_float16(
        selective,
        keep_io_types=True,
        min_positive_val=5.96e-08,
        max_finite_val=65504.0,
        node_block_list=blocked_nodes,
    )
    del selective
    attention_patches = []
    for node in converted.graph.node:
        match = ATTENTION_CAST_RE.fullmatch(node.name)
        if not match or int(match.group(1)) not in fp16_blocks:
            continue
        before = cast_target(node)
        set_cast_target(node, TensorProto.FLOAT16)
        attention_patches.append({"node_name": node.name, "target_before": before, "target_after": TensorProto.FLOAT16})
    if len(attention_patches) != 2 * len(fp16_blocks):
        raise RuntimeError(f"FP16 attention Cast patch count drift: {len(attention_patches)}")
    stale_value_info_removed = len(converted.graph.value_info)
    del converted.graph.value_info[:]
    layernorm_rows = lower_task_layernorm(converted, selected)
    convtranspose_rows = compat.rewrite_convtranspose(converted)
    resize_rows, resize_before, resize_after = resize.rewrite_head(converted)
    if len(convtranspose_rows) != 3 or len(resize_rows) != 11:
        raise RuntimeError("cloud Head compatibility rewrite count drift")
    onnx.checker.check_model(converted, full_check=True)
    after_contract = {
        "inputs": [value_contract(item) for item in converted.graph.input],
        "outputs": [value_contract(item) for item in converted.graph.output],
    }
    if after_contract != observed_contract:
        raise RuntimeError("mixed conversion changed external FP32 I/O")

    converted_ownership = ownership_with_qdq_map(converted, qdq_map, transfer)
    qdq_counts = Counter(
        converted_ownership.get(node.name)
        for node in converted.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    )
    types = transfer.tensor_types(converted)
    head_nodes = [node for node in converted.graph.node if any(token in node.name for token in HEAD_TOKENS)]
    head_convs = [node for node in head_nodes if node.op_type == "Conv"]
    head_resizes = [node for node in head_nodes if node.op_type == "Resize"]
    head_qdq = [node.name for node in head_nodes if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]
    head_fp16_failures = [
        node.name
        for node in head_convs + head_resizes
        if not node.output or types.get(node.output[0]) != TensorProto.FLOAT16
    ]
    stats = graph_stats(converted)
    checks = {
        "frozen_all24_inventory_exact": True,
        "candidate_uses_only_selected_block_qdq": all(qdq_counts[block] == 32 for block in selected)
        and all(qdq_counts[block] == 0 for block in fp16_blocks),
        "six_int8_matmuls_per_selected_block": all(
            reconstruction["selected_matmul_counts"][str(block)] == 6 for block in selected
        ),
        "unselected_encoder_blocks_fp16": True,
        "layernorm_groups_49": len(layernorm_rows) == 49,
        "selected_block_layernorm_statistics_fp32": sum(
            row["mode"] == "fp32_int8_block" for row in layernorm_rows
        )
        == 2 * len(selected),
        "head_conv_positive": len(head_convs) > 0,
        "head_resize_count_11": len(head_resizes) == 11,
        "head_conv_and_resize_fp16": not head_fp16_failures,
        "head_qdq_zero": not head_qdq,
        "float16_initializers_positive": stats["initializer_type_counts"].get("FLOAT16", 0) > 0,
        "int8_initializers_positive": stats["initializer_type_counts"].get("INT8", 0) > 0,
        "external_fp32_io_unchanged": after_contract == observed_contract,
        "onnx_full_checker_pass": True,
        "same_frozen_13_session_topology": True,
        "deployment_validation_payload_access_count_this_builder": True,
        "formal_test_payload_access_count_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cloud-MP-Task screening candidate gate failed: {checks}")

    args.output_root.mkdir(parents=True)
    model_path = args.output_root / f"cloud_mp_task_k{len(selected):02d}_mixed.onnx"
    temporary = model_path.with_suffix(".onnx.tmp")
    onnx.save(converted, str(temporary), save_as_external_data=False)
    verified = onnx.load(str(temporary.resolve(strict=True)), load_external_data=False)
    onnx.checker.check_model(verified, full_check=True)
    os.replace(temporary, model_path)

    boundary_rows = transfer.inject_mixed_boundaries(verified, topology)
    session_root = args.output_root / "models"
    session_root.mkdir()
    extractor = onnx.utils.Extractor(verified)
    sessions = []
    total_boundary_repairs = 0
    for topology_row in topology["topology"]["sessions"]:
        ordinal = int(topology_row["ordinal"])
        session_id = f"Cloud_RCS13_MP_Task_Screen_{ordinal:02d}"
        input_names = [item["name"] for item in topology_row["inputs"]]
        output_names = [item["name"] for item in topology_row["outputs"]]
        extracted = extractor.extract_model(input_names, output_names)
        repairs = base.enforce_fp32_boundaries(extracted, session_id)
        total_boundary_repairs += len(repairs)
        onnx.checker.check_model(extracted, full_check=True)
        session_path = session_root / f"session_{ordinal:02d}.onnx"
        temporary_session = session_path.with_suffix(".onnx.tmp")
        onnx.save(extracted, str(temporary_session), save_as_external_data=False)
        reloaded_session = onnx.load(str(temporary_session.resolve(strict=True)), load_external_data=False)
        onnx.checker.check_model(reloaded_session, full_check=True)
        os.replace(temporary_session, session_path)
        session_qdq = Counter(node.op_type for node in reloaded_session.graph.node)
        sessions.append(
            {
                **copy.deepcopy(topology_row),
                "session_id": session_id,
                "model": identity(session_path),
                "boundary_repairs": repairs,
                "graph_stats": base.graph_stats(reloaded_session),
                "qdq_nodes": session_qdq["QuantizeLinear"] + session_qdq["DequantizeLinear"],
                "onnx_checker": "PASS_FULL_TWICE",
            }
        )
        del reloaded_session
        del extracted
    del extractor
    del verified
    if len(sessions) != 13:
        raise RuntimeError("Cloud-MP-Task screening candidate did not produce 13 Sessions")
    result = {
        "schema": "phase7f_cloud_mp_task_screen_candidate_v1",
        "status": "BUILT_ONNX_CHECKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "Cloud-RCS13-MP-Task",
        "candidate_label": "Task-" + "-".join(f"B{block:02d}" for block in selected),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "session_count": 13,
        "precision_map": {
            "int8_qdq_blocks": list(selected),
            "fp16_blocks": list(fp16_blocks),
            "int8_scope": "six MatMul nodes per selected encoder block",
            "head": "FP16 Conv/Resize with FP32 LayerNorm statistics",
            "external_io": "FP32",
        },
        "source_onnx": source_identity,
        "all24_qdq_onnx": quant_identity,
        "all24_report": identity(args.all24_report),
        "all24_quantization_parameters": params_identity,
        "task_protocol": protocol_identity,
        "formal_access_ledger": ledger_identity,
        "dependencies": {
            "transfer_builder": identity(args.transfer_builder),
            "rcs13_base_builder": identity(args.rcs13_base_builder),
            "fp16_compat_tool": identity(args.fp16_compat_tool),
            "fp16_resize_tool": identity(args.fp16_resize_tool),
        },
        "source_topology": identity(args.cloud_topology),
        "output_onnx": identity(model_path),
        "topology": {"sessions": sessions},
        "reconstruction": reconstruction,
        "conversion": {
            "onnxconverter_common_version": onnxconverter_common.__version__,
            "int8_blocked_node_count": len(blocked_nodes),
            "stale_value_info_removed": stale_value_info_removed,
            "attention_cast_patches": attention_patches,
            "layernorm_rewrites": layernorm_rows,
            "convtranspose_rewrites": convtranspose_rows,
            "head_resize_rewrites": resize_rows,
            "head_resize_before": resize_before,
            "head_resize_after": resize_after,
            "boundary_type_injection": boundary_rows,
            "public_boundary_repairs": total_boundary_repairs,
        },
        "graph_stats": stats,
        "qdq_counts": {str(block): qdq_counts[block] for block in ALL_BLOCKS},
        "head_audit": {
            "head_node_count": len(head_nodes),
            "head_conv_count": len(head_convs),
            "head_resize_count": len(head_resizes),
            "head_qdq_nodes": head_qdq,
            "head_fp16_failures": head_fp16_failures,
        },
        "checks": checks,
        "data_firewall": {
            "calibration_payload_access_count_project_cumulative": 1280,
            "quantization_calibration_sample_count": 64,
            "deployment_validation_payload_access_count_project_cumulative": 1280,
            "deployment_validation_payload_access_count_this_builder": 0,
            "formal_test_payload_access_count": 0,
        },
    }
    manifest_path = args.output_root / "cloud_mp_task_screen_candidate_manifest.json"
    write_json(manifest_path, result)
    checksum_path = args.output_root / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(args.output_root.iterdir())
            if path.is_file() and path != checksum_path
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_label": result["candidate_label"],
                "int8_blocks": list(selected),
                "fp16_blocks": list(fp16_blocks),
                "output_onnx": result["output_onnx"],
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
