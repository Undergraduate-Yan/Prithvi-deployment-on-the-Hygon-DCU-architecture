#!/usr/bin/env python3
'Research implementation: quantize cloud.'
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BASE_QUANTIZER_SHA256 = "5b64611ff3f4c6bf64102759d54edf93411859085748a17645fd4c274c86f1ff"
CANONICAL_CLOUD_FP32_SHA256 = "c823e2c96bbb551d3f57792df585eb0677f43471333ebaf990a314587c1e1a5c"
CANONICAL_CLOUD_FP32_SIZE = 1_277_042_559
CHECKPOINT_SHA256 = "0dfe43e40c458c3a60ec6d845c6ba58d07401a5d214c0c18304f9470de632fd6"
CALIBRATION_IDS_SHA256 = "83b60f3b7f1eb03f4411e41909e1e6a11561427fa60e1c8d853aa131c63b7bfa"
CALIBRATION_INPUTS_SHA256 = "dc57e60d3bf7cd788468a675ab1d5cea36885e7ec2922000add8c69ce5cd7e06"
CALIBRATION_LEDGER_SHA256 = "05ec13e922dd96e5376c99eea7e8d85d798f33b6212bc6439999800376d69408"
EXPECTED_SAMPLE_COUNT = 1280
INT8_INVENTORY_BLOCKS = tuple(range(24))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--task-protocol", required=True, type=Path)
    parser.add_argument("--deployment-validation-manifest", required=True, type=Path)
    parser.add_argument("--formal-access-ledger", required=True, type=Path)
    parser.add_argument("--base-quantizer", required=True, type=Path)
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


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_base(path: Path) -> Any:
    if sha256_file(path) != BASE_QUANTIZER_SHA256:
        raise RuntimeError("frozen Cloud-MP quantizer implementation identity mismatch")
    spec = importlib.util.spec_from_file_location("cloud_mp_transfer_quantizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen Cloud-MP quantizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    base = load_base(args.base_quantizer.resolve(strict=True))
    source = identity(args.source)
    if source["sha256"] != CANONICAL_CLOUD_FP32_SHA256 or source["size_bytes"] != CANONICAL_CLOUD_FP32_SIZE:
        raise RuntimeError("canonical Cloud-FP32 ONNX identity mismatch")
    protocol = json.loads(args.task_protocol.read_text(encoding="utf-8"))
    dv_manifest = json.loads(args.deployment_validation_manifest.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_DEPLOYMENT_VALIDATION_PAYLOAD_ACCESS"
        or protocol.get("allowed_next", {}).get("quantize_all_24_encoder_blocks_once") is not True
        or dv_manifest.get("status") != "PASS"
        or dv_manifest.get("data_firewall", {}).get("deployment_validation_payload_access_count") != 1280
        or dv_manifest.get("data_firewall", {}).get("formal_test_payload_access_count") != 0
        or base.formal_payload_count(args.formal_access_ledger.resolve(strict=True)) != 0
    ):
        raise RuntimeError("Cloud-MP-Task all24 parent/data gate failed")
    calibration_root = args.calibration_root.resolve(strict=True)
    calibration_manifest_path = calibration_root / "cloud_mp_calibration_inputs_manifest.json"
    calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    if calibration_manifest.get("status") != "PASS":
        raise RuntimeError("cloud calibration manifest failed")
    inputs_path = calibration_root / "inputs.normalized.f32.npy"
    ids_path = calibration_root / "calibration_ids.txt"
    ledger_path = calibration_root / "calibration_payload_access.jsonl"
    observed_inputs, observed_ids, observed_ledger = identity(inputs_path), identity(ids_path), identity(ledger_path)
    if (
        observed_inputs["sha256"] != CALIBRATION_INPUTS_SHA256
        or observed_ids["sha256"] != CALIBRATION_IDS_SHA256
        or observed_ledger["sha256"] != CALIBRATION_LEDGER_SHA256
    ):
        raise RuntimeError("cloud calibration identity drift")
    all_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selection_path = args.selection_manifest.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("selection", {})
    selected_indices = [int(value) for value in chosen.get("indices", [])]
    selected_ids = [str(value) for value in chosen.get("sample_ids", [])]
    selected_ids_path = selection_path.parent / "cloud_mp_calibration64_ids.txt"
    if (
        selection.get("status") != "FROZEN_BEFORE_OFFICIAL_QUANTIZATION"
        or len(selected_indices) != 64
        or len(set(selected_indices)) != 64
        or selected_ids != [all_ids[index] for index in selected_indices]
        or identity(selected_ids_path)["sha256"] != chosen.get("ids", {}).get("sha256")
    ):
        raise RuntimeError("calibration64 selection identity/order drift")

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
    targets_by_block: dict[int, list[str]] = {}
    for block in INT8_INVENTORY_BLOCKS:
        prefix = f"/model/encoder/blocks.{block}/"
        targets = [node.name for node in model.graph.node if node.op_type == "MatMul" and node.name.startswith(prefix)]
        if len(targets) != 6 or len(set(targets)) != 6:
            raise RuntimeError(f"expected six MatMul targets in cloud block {block}: {targets}")
        targets_by_block[block] = targets
    nodes_to_quantize = [name for block in INT8_INVENTORY_BLOCKS for name in targets_by_block[block]]
    if len(nodes_to_quantize) != 144:
        raise RuntimeError("Cloud-MP-Task all24 target count must be 144")
    del model
    inputs = np.load(inputs_path, mmap_mode="r")
    if inputs.shape != (EXPECTED_SAMPLE_COUNT, 6, 224, 224) or inputs.dtype != np.float32:
        raise RuntimeError("cloud calibration input tensor contract mismatch")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.index >= len(selected_indices):
                return None
            source_index = selected_indices[self.index]
            row = np.ascontiguousarray(inputs[source_index : source_index + 1])
            self.index += 1
            if self.index % 16 == 0:
                print(json.dumps({"event": "cloud_mp_task_all24_calibration", "finished": self.index, "total": 64, "time_utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True), flush=True)
            return {"input": row}

        def rewind(self) -> None:
            self.index = 0

    args.output_root.mkdir(parents=True)
    output_model = args.output_root / "cloud_mp_task_all24_qdq_fp32.onnx"
    quantize_static(
        model_input=str(args.source.resolve(strict=True)),
        model_output=str(output_model),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
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
    parameters = base.collect_quantization_parameters(quantized, onnx)
    op_counts = Counter(node.op_type for node in quantized.graph.node)
    qdq_blocks = Counter(row["block"] for row in parameters["qdq_nodes"] if row["block"] is not None)
    unmapped = sum(row["block"] is None for row in parameters["qdq_nodes"])
    initializer_counts = Counter(onnx.TensorProto.DataType.Name(item.data_type) for item in quantized.graph.initializer)
    checks = {
        "canonical_cloud_fp32_identity_match": True,
        "task_protocol_frozen_before_dv_payload_access": True,
        "deployment_validation_manifest_pass_1280": True,
        "calibration64_frozen": True,
        "target_matmul_count_144": len(nodes_to_quantize) == 144,
        "all_24_blocks_have_qdq": all(qdq_blocks[block] > 0 for block in INT8_INVENTORY_BLOCKS),
        "qdq_only_in_encoder_blocks_0_to_23": not any(block not in INT8_INVENTORY_BLOCKS for block in qdq_blocks),
        "all_qdq_mapped": unmapped == 0,
        "int8_initializers_positive": initializer_counts["INT8"] > 0,
        "onnx_full_checker_pass": True,
        "no_flood_quantization_parameter_reuse": True,
        "deployment_validation_not_used_for_quantization": True,
        "formal_test_payload_access_count_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cloud-MP-Task all24 quantization gate failed: {checks}")
    parameter_path = args.output_root / "cloud_mp_task_all24_quantization_parameters.json"
    write_json(
        parameter_path,
        {
            "schema": "phase7f_cloud_mp_task_all24_quantization_parameters_v1",
            "status": "FROZEN_SEARCH_INVENTORY",
            "candidate": "Cloud-RCS13-MP-Task-All24-Inventory",
            "method": {
                "tool": "ONNX Runtime static PTQ",
                "format": "QDQ",
                "activation_type": "QInt8",
                "weight_type": "QInt8",
                "calibration_method": "MinMax",
                "activation_symmetric": True,
                "weight_symmetric": True,
                "per_channel_weights": True,
                "sample_count": 64,
            },
            "inventory_map": {"int8_qdq_blocks": list(INT8_INVENTORY_BLOCKS), "targets_by_block": {str(k): v for k, v in targets_by_block.items()}},
            "calibration": {
                "inputs": observed_inputs,
                "ordered_ids": observed_ids,
                "payload_access_ledger": observed_ledger,
                "selection_manifest": identity(selection_path),
                "selected_ids_file": identity(selected_ids_path),
                "selected_indices": selected_indices,
                "sample_ids": selected_ids,
            },
            "parameters": parameters,
            "flood_quantization_parameters_reused": False,
        },
    )
    report = {
        "schema": "phase7f_cloud_mp_task_all24_qdq_inventory_v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "Cloud-RCS13-MP-Task-All24-Inventory",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_onnx": source,
        "output_onnx": identity(output_model),
        "quantization_parameters": identity(parameter_path),
        "task_protocol": identity(args.task_protocol),
        "deployment_validation_manifest": identity(args.deployment_validation_manifest),
        "base_quantizer": identity(args.base_quantizer),
        "runtime": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "operator_counts": dict(sorted(op_counts.items())),
        "initializer_type_counts": dict(sorted(initializer_counts.items())),
        "qdq_block_counts": {str(key): value for key, value in sorted(qdq_blocks.items())},
        "unmapped_qdq_count": unmapped,
        "checks": checks,
        "data_firewall": {
            "calibration_payload_access_count": 1280,
            "quantization_calibration_sample_count": 64,
            "deployment_validation_payload_access_count_project_cumulative": 1280,
            "deployment_validation_payload_access_count_this_quantizer": 0,
            "formal_test_payload_access_count": 0,
        },
    }
    write_json(args.output_root / "cloud_mp_task_all24_qdq_inventory.json", report)
    checksum_path = args.output_root / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(args.output_root.iterdir()) if path.is_file() and path != checksum_path),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
