#!/usr/bin/env python3
"""Build the cloud-specific INT8-QDQ portion of Cloud-MP-Transfer.

Only the frozen flood encoder *bit-width map* is transferred.  Every activation
range, scale, zero point and quantized ONNX byte is recomputed from the frozen
CloudSEN12+ calibration inputs.  Deployment-validation and formal-test payloads
have no input path in this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_CLOUD_FP32_SHA256 = "c823e2c96bbb551d3f57792df585eb0677f43471333ebaf990a314587c1e1a5c"
CANONICAL_CLOUD_FP32_SIZE = 1_277_042_559
CHECKPOINT_SHA256 = "0dfe43e40c458c3a60ec6d845c6ba58d07401a5d214c0c18304f9470de632fd6"
CALIBRATION_IDS_SHA256 = "83b60f3b7f1eb03f4411e41909e1e6a11561427fa60e1c8d853aa131c63b7bfa"
CALIBRATION_INPUTS_SHA256 = "dc57e60d3bf7cd788468a675ab1d5cea36885e7ec2922000add8c69ce5cd7e06"
CALIBRATION_LEDGER_SHA256 = "05ec13e922dd96e5376c99eea7e8d85d798f33b6212bc6439999800376d69408"
EXPECTED_SAMPLE_COUNT = 1280
FP16_BLOCKS = (0, 8, 10, 14, 15, 17, 18, 19, 20)
INT8_BLOCKS = (1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 21, 22, 23)
BLOCK_RE = re.compile(r"/model/encoder/blocks\.(\d+)(?:/|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--formal-access-ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "official"), default="official")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--selection-manifest", type=Path)
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


def formal_payload_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(
        "formal_test_payload_access_count",
        payload.get("payload_access_count", payload.get("access_count")),
    )
    if value is None:
        raise RuntimeError("formal access ledger has no payload-access count")
    return int(value)


def value_contract(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "element_type": int(tensor.elem_type),
        "shape": [int(dim.dim_value) for dim in tensor.shape.dim],
    }


def block_from_text(*values: str) -> int | None:
    blocks = {
        int(match.group(1))
        for value in values
        for match in BLOCK_RE.finditer(value or "")
    }
    return next(iter(blocks)) if len(blocks) == 1 else None


def initializer_record(initializer: Any, onnx: Any) -> dict[str, Any]:
    array = onnx.numpy_helper.to_array(initializer)
    contiguous = np.ascontiguousarray(array)
    return {
        "name": initializer.name,
        "dtype": str(contiguous.dtype),
        "onnx_dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
        "shape": list(contiguous.shape),
        "value_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "values": contiguous.reshape(-1).tolist(),
    }


def attributes(node: Any, onnx: Any) -> dict[str, Any]:
    return {
        item.name: onnx.helper.get_attribute_value(item)
        for item in node.attribute
    }


def collect_quantization_parameters(model: Any, onnx: Any) -> dict[str, Any]:
    initializers = {item.name: item for item in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    rows = []
    initializer_names: set[str] = set()
    for ordinal, node in enumerate(model.graph.node):
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        attrs = attributes(node, onnx)
        scale_name = node.input[1] if len(node.input) > 1 else None
        zero_name = node.input[2] if len(node.input) > 2 else None
        if scale_name not in initializers or zero_name not in initializers:
            raise RuntimeError(f"Q/DQ parameter initializer missing at {node.name}")
        initializer_names.update((scale_name, zero_name))
        upstream = producers.get(node.input[0]) if node.input else None
        downstream = [
            item
            for output in node.output
            for item in consumers.get(output, [])
        ]
        block = block_from_text(
            node.name,
            *node.input,
            *node.output,
            upstream.name if upstream is not None else "",
            *(item.name for item in downstream),
        )
        rows.append(
            {
                "ordinal": ordinal,
                "node_name": node.name,
                "op_type": node.op_type,
                "block": block,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "axis": int(attrs["axis"]) if "axis" in attrs else None,
                "granularity": "per-channel" if "axis" in attrs else "per-tensor",
                "scale_initializer": scale_name,
                "zero_point_initializer": zero_name,
                "upstream_node": upstream.name if upstream is not None else None,
                "downstream_nodes": [item.name for item in downstream],
            }
        )
    return {
        "qdq_nodes": rows,
        "initializers": [
            initializer_record(initializers[name], onnx)
            for name in sorted(initializer_names)
        ],
    }


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    if args.sample_limit is not None and args.sample_limit < 1:
        raise RuntimeError("sample-limit must be positive")
    if args.mode == "official" and args.selection_manifest is None:
        raise RuntimeError("official mode requires a pre-frozen calibration selection manifest")
    if args.mode == "official" and args.sample_limit is not None:
        raise RuntimeError("official mode derives its sample count only from the frozen selection")

    source = identity(args.source)
    if (
        source["sha256"] != CANONICAL_CLOUD_FP32_SHA256
        or source["size_bytes"] != CANONICAL_CLOUD_FP32_SIZE
    ):
        raise RuntimeError("canonical Cloud-FP32 ONNX identity mismatch")
    calibration_root = args.calibration_root.resolve(strict=True)
    calibration_manifest_path = calibration_root / "cloud_mp_calibration_inputs_manifest.json"
    calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    if calibration_manifest.get("status") != "PASS":
        raise RuntimeError("cloud calibration input gate is not PASS")
    firewall = calibration_manifest.get("data_firewall", {})
    if firewall != {
        "payload_sets_accessed": ["calibration"],
        "calibration_payload_access_count": EXPECTED_SAMPLE_COUNT,
        "deployment_validation_payload_access_count": 0,
        "formal_test_payload_access_count": 0,
    }:
        raise RuntimeError(f"calibration data-firewall contract drift: {firewall}")
    inputs_path = calibration_root / "inputs.normalized.f32.npy"
    ids_path = calibration_root / "calibration_ids.txt"
    ledger_path = calibration_root / "calibration_payload_access.jsonl"
    observed_inputs = identity(inputs_path)
    observed_ids = identity(ids_path)
    observed_ledger = identity(ledger_path)
    if observed_inputs["sha256"] != CALIBRATION_INPUTS_SHA256:
        raise RuntimeError("cloud calibration input array identity mismatch")
    if observed_ids["sha256"] != CALIBRATION_IDS_SHA256:
        raise RuntimeError("cloud calibration ordered-ID identity mismatch")
    if observed_ledger["sha256"] != CALIBRATION_LEDGER_SHA256:
        raise RuntimeError("cloud calibration payload ledger identity mismatch")
    sample_ids = [
        line.strip()
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sample_ids) != EXPECTED_SAMPLE_COUNT or len(set(sample_ids)) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("cloud calibration ordered-ID list count/uniqueness mismatch")
    if formal_payload_count(args.formal_access_ledger.resolve(strict=True)) != 0:
        raise RuntimeError("formal cloud test payload is no longer sealed")

    selection_identity = None
    if args.mode == "official":
        selection_path = args.selection_manifest.resolve(strict=True)
        selection_identity = identity(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema") != "phase7f_cloud_mp_calibration64_selection_v1":
            raise RuntimeError("cloud MP calibration selection schema drift")
        if selection.get("status") != "FROZEN_BEFORE_OFFICIAL_QUANTIZATION":
            raise RuntimeError("cloud MP calibration selection was not frozen before quantization")
        if not selection.get("checks") or not all(selection["checks"].values()):
            raise RuntimeError("cloud MP calibration selection gate failed")
        chosen = selection.get("selection", {})
        selected_indices = [int(value) for value in chosen.get("indices", [])]
        selected_ids = [str(value) for value in chosen.get("sample_ids", [])]
        if len(selected_indices) != 64 or len(set(selected_indices)) != 64:
            raise RuntimeError("official calibration selection must contain 64 unique indices")
        if selected_ids != [sample_ids[index] for index in selected_indices]:
            raise RuntimeError("official calibration selection ID/index mapping mismatch")
        selected_ids_identity = chosen.get("ids", {})
        selected_ids_path = selection_path.parent / "cloud_mp_calibration64_ids.txt"
        observed_selected_ids = identity(selected_ids_path)
        if (
            observed_selected_ids["sha256"] != selected_ids_identity.get("sha256")
            or observed_selected_ids["size_bytes"] != selected_ids_identity.get("size_bytes")
        ):
            raise RuntimeError("official calibration64 ID file identity mismatch")
    else:
        selected_indices = list(range(min(args.sample_limit or 1, EXPECTED_SAMPLE_COUNT)))
        selected_ids = [sample_ids[index] for index in selected_indices]
        observed_selected_ids = None

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    if ort.__version__ != "1.19.2":
        raise RuntimeError(f"onnxruntime version drift: {ort.__version__}")
    model = onnx.load(str(args.source.resolve(strict=True)), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    opsets = {item.domain: int(item.version) for item in model.opset_import}
    expected_contract = {
        "inputs": [{"name": "input", "element_type": onnx.TensorProto.FLOAT, "shape": [1, 6, 224, 224]}],
        "outputs": [{"name": "logits", "element_type": onnx.TensorProto.FLOAT, "shape": [1, 4, 224, 224]}],
    }
    contract = {
        "inputs": [value_contract(item) for item in model.graph.input],
        "outputs": [value_contract(item) for item in model.graph.output],
    }
    if opsets.get("") != 17 or contract != expected_contract:
        raise RuntimeError(f"canonical cloud graph contract drift: opset={opsets}, io={contract}")
    targets_by_block: dict[int, list[str]] = {}
    for block in INT8_BLOCKS:
        prefix = f"/model/encoder/blocks.{block}/"
        rows = [
            node.name
            for node in model.graph.node
            if node.op_type == "MatMul" and node.name.startswith(prefix)
        ]
        if len(rows) != 6 or len(set(rows)) != 6:
            raise RuntimeError(f"expected six MatMul targets in cloud block {block}, got {rows}")
        targets_by_block[block] = rows
    nodes_to_quantize = [
        name for block in INT8_BLOCKS for name in targets_by_block[block]
    ]
    if len(nodes_to_quantize) != 90:
        raise RuntimeError("Cloud-MP-Transfer target count must be 90")
    del model

    inputs = np.load(inputs_path, mmap_mode="r")
    if inputs.shape != (EXPECTED_SAMPLE_COUNT, 6, 224, 224) or inputs.dtype != np.float32:
        raise RuntimeError("cloud calibration input tensor contract mismatch")
    sample_count = len(selected_indices)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.index >= sample_count:
                return None
            source_index = selected_indices[self.index]
            row = np.ascontiguousarray(inputs[source_index : source_index + 1])
            self.index += 1
            if self.index % 16 == 0 or self.index == sample_count:
                print(
                    json.dumps(
                        {
                            "event": "cloud_mp_calibration_progress",
                            "finished": self.index,
                            "total": sample_count,
                            "time_utc": datetime.now(timezone.utc).isoformat(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return {"input": row}

        def rewind(self) -> None:
            self.index = 0

    args.output_root.mkdir(parents=True)
    output_model = args.output_root / "cloud_mp_transfer_int8_qdq_fp32.onnx"
    quantize_static(
        model_input=str(args.source.resolve(strict=True)),
        model_output=str(output_model),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        # ORT 1.19 range adjustment needs Softmax ranges even though only the
        # explicitly registered MatMul nodes are quantized.
        op_types_to_quantize=["MatMul", "Softmax"],
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=nodes_to_quantize,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "DedicatedQDQPair": True,
            "QDQKeepRemovableActivations": True,
        },
        use_external_data_format=False,
    )
    quantized = onnx.load(str(output_model.resolve(strict=True)), load_external_data=False)
    onnx.checker.check_model(quantized, full_check=True)
    op_counts = Counter(node.op_type for node in quantized.graph.node)
    by_name = {node.name: node for node in quantized.graph.node}
    missing_targets = [
        name for name in nodes_to_quantize if name not in by_name and f"{name}_quant" not in by_name
    ]
    parameters = collect_quantization_parameters(quantized, onnx)
    qdq_block_counts = Counter(
        row["block"] for row in parameters["qdq_nodes"] if row["block"] is not None
    )
    unexpected_blocks = sorted(
        block for block in qdq_block_counts if block not in INT8_BLOCKS
    )
    unmapped_qdq_count = sum(row["block"] is None for row in parameters["qdq_nodes"])
    initializer_type_counts = Counter(
        onnx.TensorProto.DataType.Name(item.data_type)
        for item in quantized.graph.initializer
    )
    checks = {
        "canonical_cloud_fp32_identity_match": True,
        "checkpoint_lineage_frozen": True,
        "calibration_manifest_pass": True,
        "calibration_inputs_identity_match": True,
        "calibration_ids_identity_match": True,
        "calibration_payload_ledger_identity_match": True,
        "official_selection_frozen_before_quantization": args.mode != "official" or selection_identity is not None,
        "official_uses_balanced_roi_unique_64": args.mode != "official" or sample_count == 64,
        "flood_bit_width_map_transferred_exactly": set(FP16_BLOCKS).isdisjoint(INT8_BLOCKS)
        and set(FP16_BLOCKS) | set(INT8_BLOCKS) == set(range(24)),
        "target_matmul_count_90": len(nodes_to_quantize) == 90,
        "all_target_matmul_nodes_retained": not missing_targets,
        "quantize_linear_positive": op_counts["QuantizeLinear"] > 0,
        "dequantize_linear_positive": op_counts["DequantizeLinear"] > 0,
        "int8_initializers_positive": initializer_type_counts["INT8"] > 0,
        "every_transferred_int8_block_has_qdq": all(qdq_block_counts[block] > 0 for block in INT8_BLOCKS),
        "qdq_only_in_transferred_int8_blocks": not unexpected_blocks,
        "every_qdq_node_mapped_to_one_encoder_block": unmapped_qdq_count == 0,
        "no_flood_scale_or_zero_point_reused": True,
        "onnx_full_checker_pass": True,
        "deployment_validation_payload_access_count_zero": True,
        "formal_test_payload_access_count_zero": formal_payload_count(args.formal_access_ledger) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cloud-MP-Transfer quantization gate failed: {checks}")
    output_identity = identity(output_model)
    parameter_path = args.output_root / "cloud_mp_transfer_quantization_parameters.json"
    write_json(
        parameter_path,
        {
            "schema": "phase7f_cloud_mp_transfer_quantization_parameters_v1",
            "status": "SMOKE_ONLY" if args.mode == "smoke" else "FROZEN_OFFICIAL",
            "candidate": "Cloud-RCS13-MP-Transfer",
            "method": {
                "tool": "ONNX Runtime static PTQ",
                "format": "QDQ",
                "activation_type": "QInt8",
                "weight_type": "QInt8",
                "calibration_method": "MinMax",
                "activation_symmetric": True,
                "weight_symmetric": True,
                "per_channel_weights": True,
                "sample_count": sample_count,
            },
            "transferred_map": {
                "fp16_blocks": list(FP16_BLOCKS),
                "int8_qdq_blocks": list(INT8_BLOCKS),
                "targets_by_block": {str(key): value for key, value in targets_by_block.items()},
            },
            "calibration": {
                "inputs": observed_inputs,
                "ordered_ids": observed_ids,
                "payload_access_ledger": observed_ledger,
                "ordered_ids_sha256": CALIBRATION_IDS_SHA256,
                "official_selection_manifest": selection_identity,
                "selected_ids_file": observed_selected_ids,
                "selected_indices": selected_indices,
                "sample_ids": selected_ids,
                "sample_count": sample_count,
            },
            "parameters": parameters,
            "flood_quantization_parameters_reused": False,
        },
    )
    report = {
        "schema": "phase7f_cloud_mp_transfer_int8_qdq_build_v1",
        "status": "SMOKE_PASS" if args.mode == "smoke" else "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "candidate": "Cloud-RCS13-MP-Transfer",
        "hardware": "海光 K100 AI 加速卡",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_onnx": source,
        "output_onnx": output_identity,
        "quantization_parameters": identity(parameter_path),
        "runtime": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "sample_count": sample_count,
        "operator_counts": dict(sorted(op_counts.items())),
        "initializer_type_counts": dict(sorted(initializer_type_counts.items())),
        "qdq_block_counts": {str(key): value for key, value in sorted(qdq_block_counts.items())},
        "missing_targets": missing_targets,
        "unexpected_qdq_blocks": unexpected_blocks,
        "unmapped_qdq_count": unmapped_qdq_count,
        "checks": checks,
        "data_firewall": {
            "calibration_payload_access_count": EXPECTED_SAMPLE_COUNT,
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
            "formal_test_used_for_quantization": False,
        },
    }
    write_json(args.output_root / "cloud_mp_transfer_int8_qdq_build.json", report)
    checksum_path = args.output_root / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(args.output_root.iterdir())
            if path.is_file() and path != checksum_path
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
