#!/usr/bin/env python3
"""Build identity-locked M0-M4 FP16/INT8 25-segment candidate manifests.

This builder deliberately does not splice the earlier FP32 block-0 artifact.  It
extracts the selected transformer blocks from the frozen FP16 full model,
rewrites each LayerNormalization into an FP32 statistic island, and exposes
only FP32 segment I/O.  All remaining backbone segments and their caches are
reused byte-for-byte from the frozen INT8 evidence bundle.  Segment 24 is the
validated FP32-compatible fpn4-MaxPool-barrier head.

The output is a construction artifact, not an admission result.  New FP16 MXR
caches are intentionally absent and must be compiled and admitted on K100.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
BUILD_SCHEMA = "phase11_sensitivity_mixed_precision_build_v1"

FP16_SOURCE = (
    638_894_819,
    "10df534d4dbaabf8336e4195a60d2ebd719b14c7a3e34362d48a0609425e6dbd",
)
FP16_REPORT = (
    4_914,
    "2b559ae9ca6c6fc5bfeac67d3df996bda7581f5f4d111736869ecb2b05108c3d",
)
INT8_QUANT_REPORT = (
    10_043,
    "c1a7689c716560f1fb7eb96ffb23db3555e32aa5aafeb2cbff8e7385b84f357a",
)
INT8_COMPAT_SOURCE = (
    368_047_490,
    "edd2c3e69b64986dc6d0b78fe32230f9a9f7eb0396c2d77fdcb6bf4c59506a7a",
)
INT8_SEGMENT_REPORT = (
    21_342,
    "e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e",
)
REPAIRED_HEAD = (
    60_551_018,
    "a5ecda6c168d187c8f8ff6b0712ddd46d899759878772dedcdff119165876e48",
)
REPAIRED_HEAD_CACHE = (
    61_093_619,
    "8d7a9aaca042d2c1e5ece0b8fdf32039550eee15f0128e3fbf9c915af41f8654",
)
SOURCE_FP32_SHA256 = "d7828912240f61ba3cfb9d27c787d90abb4a1b3428bfce8e9488aa0c1e6225d4"
SOURCE_CHECKPOINT_SHA256 = "59f031f2eaa60219452c175825162ad30be158560716ccf01497019522cd8256"

BLOCK = {index: f"/task/model/encoder/blocks.{index}/Add_1_output_0" for index in range(24)}
LABELS = [f"encoder_block_{index:02d}" for index in range(24)] + ["upernet_decoder_head"]
SPECS = [
    (
        f"encoder_block_{index:02d}",
        ["image" if index == 0 else BLOCK[index - 1]],
        [BLOCK[index]],
    )
    for index in range(24)
]
HEAD_INPUTS = [BLOCK[5], BLOCK[11], BLOCK[17], BLOCK[23]]
BARRIER_TENSOR = "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0"

CANDIDATES: dict[str, frozenset[int]] = {
    "M0": frozenset(),
    "M1": frozenset({0}),
    "M2": frozenset({0, 14}),
    "M3": frozenset({0, 14, 17}),
    "M4": frozenset({0, 14, 17, 18}),
}
FP16_BLOCKS = tuple(sorted(set().union(*CANDIDATES.values())))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    row = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (row["size_bytes"], row["sha256"]) != expected:
        raise RuntimeError(f"identity drift for {path}: {row}; expected={expected}")
    return row


def load_json(path: Path, expected: tuple[int, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    ident = identity(path, expected)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload, ident


def tensor_contract(value: Any) -> dict[str, Any]:
    tensor = value.type.tensor_type
    shape: list[int | str | None] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return {"name": value.name, "elem_type": int(tensor.elem_type), "shape": shape}


def graph_contract(model: onnx.ModelProto) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [tensor_contract(item) for item in model.graph.input],
        [tensor_contract(item) for item in model.graph.output],
    )


def require_fp32_static_contract(rows: list[dict[str, Any]], context: str) -> None:
    for row in rows:
        if int(row.get("elem_type", -1)) != TensorProto.FLOAT:
            raise RuntimeError(f"{context} is not externally FP32: {row}")
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape or shape[0] != 1:
            raise RuntimeError(f"{context} batch is not static 1: {row}")
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise RuntimeError(f"{context} shape is not fully static: {row}")


def normalized_contract(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid contract rows: {rows!r}")
    normalized = []
    for row in rows:
        normalized.append(
            {
                "name": str(row["name"]),
                "elem_type": int(row["elem_type"]),
                "shape": list(row["shape"]),
            }
        )
    return normalized


def resolve_manifest_path(manifest: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(strict=True)


def relative_posix(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(strict=True), parent.resolve())).as_posix()


def materialize_file(source: Path, target: Path, mode: str) -> None:
    source = source.resolve(strict=True)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise RuntimeError(f"unsupported payload materialization mode: {mode}")


def materialize_frozen_payload(
    shared_root: Path,
    int8_rows: list[dict[str, Any]],
    head: dict[str, Any],
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Place all runtime payload beneath the candidate bundle authorization root."""
    models_root = shared_root / "models"
    caches_root = shared_root / "caches"
    materialized_rows: list[dict[str, Any]] = []
    files = []
    for index, row in enumerate(int8_rows[:24]):
        model_target = models_root / f"{index:02d}_{row['label']}_int8_qdq.onnx"
        cache_target = caches_root / f"segment_{index:02d}_int8_qdq.mxr"
        materialize_file(row["model_path"], model_target, mode)
        materialize_file(row["cache_path"], cache_target, mode)
        model_ident = identity(model_target, (row["model_identity"]["size_bytes"], row["model_identity"]["sha256"]))
        cache_ident = identity(cache_target, (row["cache_identity"]["size_bytes"], row["cache_identity"]["sha256"]))
        copied = dict(row)
        copied["model_path"] = model_target.resolve(strict=True)
        copied["cache_path"] = cache_target.resolve(strict=True)
        materialized_rows.append(copied)
        files.extend(
            [
                {"role": f"int8_model_{index:02d}", "path": model_target, "identity": model_ident},
                {"role": f"int8_cache_{index:02d}", "path": cache_target, "identity": cache_ident},
            ]
        )
    # Preserve the unused original head row only as contract metadata.  Candidate
    # segment 24 always uses the separately admitted repaired head below.
    materialized_rows.append(int8_rows[24])

    head_model_target = models_root / "24_upernet_decoder_head_fpn4barrier.onnx"
    head_cache_target = caches_root / "segment_24_head_fpn4barrier.mxr"
    materialize_file(head["model_path"], head_model_target, mode)
    materialize_file(head["cache_path"], head_cache_target, mode)
    materialized_head = dict(head)
    materialized_head["model_path"] = head_model_target.resolve(strict=True)
    materialized_head["cache_path"] = head_cache_target.resolve(strict=True)
    files.extend(
        [
            {"role": "repaired_head_model", "path": head_model_target, "identity": identity(head_model_target, REPAIRED_HEAD)},
            {"role": "repaired_head_cache", "path": head_cache_target, "identity": identity(head_cache_target, REPAIRED_HEAD_CACHE)},
        ]
    )
    report = {
        "mode": mode,
        "file_count": len(files),
        "all_runtime_payload_under_shared_root": True,
        "files": [
            {
                "role": row["role"],
                "path": str(row["path"].relative_to(shared_root)).replace("\\", "/"),
                "identity": row["identity"],
            }
            for row in files
        ],
    }
    return materialized_rows, materialized_head, report


def audit_provenance(fp16_report_path: Path, int8_report_path: Path) -> dict[str, Any]:
    fp16, fp16_ident = load_json(fp16_report_path, FP16_REPORT)
    int8, int8_ident = load_json(int8_report_path, INT8_QUANT_REPORT)
    required = {
        "source_fp32_sha256": SOURCE_FP32_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    }
    for label, payload in (("FP16", fp16), ("INT8", int8)):
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"{label} provenance drift for {key}")
    if fp16.get("fp16_sha256") != FP16_SOURCE[1] or fp16.get("precision") != "fp16_internal_fp32_io":
        raise RuntimeError("FP16 report does not describe the frozen source")
    if int8.get("variant") != "int8_backbone" or int8.get("quant_format") != "QDQ":
        raise RuntimeError("INT8 report is not the frozen backbone QDQ report")
    if int8.get("activation_type") != "QInt8" or int8.get("weight_type") != "QInt8":
        raise RuntimeError("INT8 quantization type drift")
    return {
        "same_fp32_source": True,
        "same_checkpoint": True,
        "source_fp32_sha256": SOURCE_FP32_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "fp16_report_identity": fp16_ident,
        "int8_quantization_report_identity": int8_ident,
        "int8_calibration_sha256": int8.get("calibration_sha256"),
        "int8_calibration_sample_count": int8.get("calibration_sample_count"),
    }


def audit_onnx(path: Path) -> tuple[onnx.ModelProto, dict[str, int]]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    return model, dict(Counter(node.op_type for node in model.graph.node))


def audit_int8_bundle(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest, manifest_ident = load_json(manifest_path)
    if manifest.get("status") != "assembled_and_identity_locked":
        raise RuntimeError("frozen INT8 bundle manifest is not identity locked")
    if manifest.get("bundle_format") != "phase11_int8_backbone_segment25_v1":
        raise RuntimeError("frozen INT8 bundle format drift")
    source = manifest.get("source_provenance_not_bundled", {})
    if (int(source.get("size_bytes", -1)), str(source.get("sha256", ""))) != INT8_COMPAT_SOURCE:
        raise RuntimeError("INT8 compatibility source identity drift")
    prereq_report = manifest.get("prerequisites", {}).get("segment_report", {})
    if (
        int(prereq_report.get("size_bytes", -1)),
        str(prereq_report.get("sha256", "")),
    ) != INT8_SEGMENT_REPORT:
        raise RuntimeError("frozen INT8 segment report identity drift")

    rows = manifest.get("segments")
    if not isinstance(rows, list) or len(rows) != 25:
        raise RuntimeError("frozen INT8 manifest must contain exactly 25 segments")
    audited: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected_label = LABELS[index]
        if row.get("index") != index or row.get("label") != expected_label:
            raise RuntimeError(f"INT8 segment ordering drift at {index}: {row.get('label')}")
        model_path = resolve_manifest_path(manifest_path, str(row["model"]))
        cache_path = resolve_manifest_path(manifest_path, str(row["cache"]))
        model_ident = identity(
            model_path,
            (int(row["model_identity"]["size_bytes"]), str(row["model_identity"]["sha256"])),
        )
        cache_ident = identity(
            cache_path,
            (int(row["cache_identity"]["size_bytes"]), str(row["cache_identity"]["sha256"])),
        )
        model, node_types = audit_onnx(model_path)
        inputs, outputs = graph_contract(model)
        expected_inputs = normalized_contract(row.get("inputs"))
        expected_outputs = normalized_contract(row.get("outputs"))
        if inputs != expected_inputs or outputs != expected_outputs:
            raise RuntimeError(f"frozen INT8 segment contract drift at {index}")
        require_fp32_static_contract(inputs + outputs, f"INT8 segment {index}")
        if index < 24 and (
            node_types.get("QuantizeLinear", 0) == 0 or node_types.get("DequantizeLinear", 0) == 0
        ):
            raise RuntimeError(f"INT8 backbone segment {index} no longer contains QDQ nodes")
        audited.append(
            {
                "index": index,
                "label": expected_label,
                "model_path": model_path,
                "model_identity": model_ident,
                "cache_path": cache_path,
                "cache_identity": cache_ident,
                "inputs": inputs,
                "outputs": outputs,
                "node_types": node_types,
            }
        )
        del model
    return manifest, manifest_ident, audited


def audit_repaired_head(
    model_path: Path,
    cache_path: Path,
    frozen_head_contract: dict[str, Any],
) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    cache_path = cache_path.resolve(strict=True)
    model_ident = identity(model_path, REPAIRED_HEAD)
    cache_ident = identity(cache_path, REPAIRED_HEAD_CACHE)
    model, node_types = audit_onnx(model_path)
    inputs, outputs = graph_contract(model)
    expected_inputs = normalized_contract(frozen_head_contract["inputs"])
    if inputs != expected_inputs:
        raise RuntimeError("repaired head input contract drift")
    if [item["name"] for item in outputs] != ["logits", BARRIER_TENSOR]:
        raise RuntimeError("repaired head output/barrier contract drift")
    require_fp32_static_contract(inputs + outputs, "repaired FP32-compatible head")
    if node_types.get("QuantizeLinear", 0) or node_types.get("DequantizeLinear", 0):
        raise RuntimeError("repaired head unexpectedly contains QDQ nodes")
    del model
    return {
        "model_path": model_path,
        "model_identity": model_ident,
        "cache_path": cache_path,
        "cache_identity": cache_ident,
        "inputs": inputs,
        "outputs": outputs,
        "node_types": node_types,
    }


def node_attributes(node: Any) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def rewrite_layernorm_statistic_islands(model: onnx.ModelProto, block_index: int) -> list[dict[str, Any]]:
    rewritten = []
    replacement = []
    initializer_names = {item.name for item in model.graph.initializer}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization":
            replacement.append(node)
            continue
        attrs = node_attributes(node)
        axis = int(attrs.get("axis", -1))
        epsilon = float(attrs.get("epsilon", 1e-5))
        stash_type = int(attrs.get("stash_type", TensorProto.FLOAT))
        if axis != -1 or stash_type != TensorProto.FLOAT or len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(f"unsupported LayerNormalization at {node.name}: {attrs}")
        base = f"phase11_mixed_b{block_index:02d}_ln_{node_index}"
        epsilon_name = base + "_epsilon_fp32"
        if epsilon_name in initializer_names:
            raise RuntimeError(f"initializer collision: {epsilon_name}")
        initializer_names.add(epsilon_name)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), epsilon_name)
        )
        x, scale, bias = node.input
        y = node.output[0]
        x32, scale32, bias32 = base + "_x32", base + "_scale32", base + "_bias32"
        mean, centered, squared = base + "_mean", base + "_centered", base + "_squared"
        variance, variance_epsilon, stddev = (
            base + "_variance",
            base + "_variance_epsilon",
            base + "_stddev",
        )
        normalized, scaled, y32 = base + "_normalized", base + "_scaled", base + "_y32"
        replacement.extend(
            [
                helper.make_node("Cast", [x], [x32], name=base + "/CastInputFP32", to=TensorProto.FLOAT),
                helper.make_node("Cast", [scale], [scale32], name=base + "/CastScaleFP32", to=TensorProto.FLOAT),
                helper.make_node("Cast", [bias], [bias32], name=base + "/CastBiasFP32", to=TensorProto.FLOAT),
                helper.make_node("ReduceMean", [x32], [mean], name=base + "/ReduceMean", axes=[-1], keepdims=1),
                helper.make_node("Sub", [x32, mean], [centered], name=base + "/Center"),
                helper.make_node("Mul", [centered, centered], [squared], name=base + "/Square"),
                helper.make_node("ReduceMean", [squared], [variance], name=base + "/Variance", axes=[-1], keepdims=1),
                helper.make_node("Add", [variance, epsilon_name], [variance_epsilon], name=base + "/AddEpsilon"),
                helper.make_node("Sqrt", [variance_epsilon], [stddev], name=base + "/Sqrt"),
                helper.make_node("Div", [centered, stddev], [normalized], name=base + "/Normalize"),
                helper.make_node("Mul", [normalized, scale32], [scaled], name=base + "/Scale"),
                helper.make_node("Add", [scaled, bias32], [y32], name=base + "/Bias"),
                helper.make_node("Cast", [y32], [y], name=base + "/CastOutputFP16", to=TensorProto.FLOAT16),
            ]
        )
        rewritten.append(
            {
                "original_node": node.name,
                "original_node_index": node_index,
                "axis": axis,
                "epsilon": epsilon,
                "mode": "fp16_io_fp32_statistic_island",
            }
        )
    if len(rewritten) != 2:
        raise RuntimeError(
            f"FP16 transformer block {block_index} must contain exactly 2 LayerNormalization nodes; "
            f"found {len(rewritten)}"
        )
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    return rewritten


def set_static_batch_one(value: Any) -> None:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape") or not tensor.shape.dim:
        raise RuntimeError(f"missing tensor shape for {value.name}")
    batch = tensor.shape.dim[0]
    batch.ClearField("dim_param")
    batch.dim_value = 1


def externalize_fp32_io(model: onnx.ModelProto, block_index: int) -> dict[str, Any]:
    input_casts = []
    output_casts = []

    prepended = []
    for value in model.graph.input:
        set_static_batch_one(value)
        elem_type = int(value.type.tensor_type.elem_type)
        if elem_type == TensorProto.FLOAT:
            continue
        if elem_type != TensorProto.FLOAT16:
            raise RuntimeError(f"unsupported FP16 block input type for {value.name}: {elem_type}")
        external_name = value.name
        internal_name = external_name + f"__phase11_mixed_b{block_index:02d}_fp16"
        for node in model.graph.node:
            for offset, name in enumerate(node.input):
                if name == external_name:
                    node.input[offset] = internal_name
        value.type.tensor_type.elem_type = TensorProto.FLOAT
        prepended.append(
            helper.make_node(
                "Cast",
                [external_name],
                [internal_name],
                name=f"phase11_mixed_b{block_index:02d}/ExternalInputToFP16",
                to=TensorProto.FLOAT16,
            )
        )
        input_casts.append({"external": external_name, "internal": internal_name})

    appended = []
    for value in model.graph.output:
        set_static_batch_one(value)
        elem_type = int(value.type.tensor_type.elem_type)
        if elem_type == TensorProto.FLOAT:
            continue
        if elem_type != TensorProto.FLOAT16:
            raise RuntimeError(f"unsupported FP16 block output type for {value.name}: {elem_type}")
        external_name = value.name
        internal_name = external_name + f"__phase11_mixed_b{block_index:02d}_fp16"
        producer_count = 0
        for node in model.graph.node:
            for offset, name in enumerate(node.output):
                if name == external_name:
                    node.output[offset] = internal_name
                    producer_count += 1
        if producer_count != 1:
            raise RuntimeError(f"expected one producer for block output {external_name}; found {producer_count}")
        for node in model.graph.node:
            for offset, name in enumerate(node.input):
                if name == external_name:
                    node.input[offset] = internal_name
        value.type.tensor_type.elem_type = TensorProto.FLOAT
        appended.append(
            helper.make_node(
                "Cast",
                [internal_name],
                [external_name],
                name=f"phase11_mixed_b{block_index:02d}/InternalOutputToFP32",
                to=TensorProto.FLOAT,
            )
        )
        output_casts.append({"internal": internal_name, "external": external_name})

    if block_index == 0:
        if input_casts:
            raise RuntimeError("block 0 image input must already be FP32 in the frozen FP16 source")
    elif len(input_casts) != 1:
        raise RuntimeError(f"block {block_index} must have one embedded FP32-to-FP16 input cast")
    if len(output_casts) != 1:
        raise RuntimeError(f"block {block_index} must have one embedded FP16-to-FP32 output cast")

    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(prepended + original_nodes + appended)
    return {"input_casts": input_casts, "output_casts": output_casts}


def add_metadata(model: onnx.ModelProto, block_index: int) -> None:
    entries = {
        "phase11_candidate_role": "sensitivity_selected_fp16_transformer_block",
        "phase11_block_index": str(block_index),
        "phase11_external_io": "float32",
        "phase11_internal_precision": "float16_with_fp32_layernorm_statistic_islands",
        "phase11_fp16_source_sha256": FP16_SOURCE[1],
        "phase11_source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    }
    existing = {item.key for item in model.metadata_props}
    overlap = existing & entries.keys()
    if overlap:
        raise RuntimeError(f"metadata collision: {sorted(overlap)}")
    for key, value in entries.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value


def prepare_extraction_source(fp16_source: Path, shared_root: Path) -> tuple[Path, dict[str, Any]]:
    source_ident = identity(fp16_source, FP16_SOURCE)
    model, node_types = audit_onnx(fp16_source)
    inputs, outputs = graph_contract(model)
    if [item["name"] for item in inputs] != ["image"] or [item["name"] for item in outputs] != ["logits"]:
        raise RuntimeError("frozen FP16 full-model I/O drift")
    require_fp32_static_contract(inputs + outputs, "frozen FP16 full model")
    if node_types.get("LayerNormalization") != 49:
        raise RuntimeError("frozen FP16 source must contain 49 LayerNormalization nodes")
    if node_types.get("QuantizeLinear", 0) or node_types.get("DequantizeLinear", 0):
        raise RuntimeError("frozen FP16 source unexpectedly contains QDQ nodes")
    known = {
        item.name
        for item in [*model.graph.input, *model.graph.output, *model.graph.value_info]
    }
    required_boundaries = {
        name for block_index in FP16_BLOCKS for name in [*SPECS[block_index][1], *SPECS[block_index][2]]
    }
    del model
    if required_boundaries <= known:
        return fp16_source.resolve(strict=True), {
            "identity": source_ident,
            "shape_inference_copy_created": False,
            "node_types": node_types,
        }

    inferred_path = shared_root / ".fp16_source_shape_inferred.onnx"
    onnx.shape_inference.infer_shapes_path(
        str(fp16_source.resolve(strict=True)),
        str(inferred_path),
        data_prop=False,
    )
    inferred, _ = audit_onnx(inferred_path)
    inferred_known = {
        item.name
        for item in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]
    }
    del inferred
    missing = sorted(required_boundaries - inferred_known)
    if missing:
        raise RuntimeError(f"FP16 shape inference did not expose split boundaries: {missing}")
    return inferred_path, {
        "identity": source_ident,
        "shape_inference_copy_created": True,
        "node_types": node_types,
    }


def build_fp16_segments(
    extraction_source: Path,
    output_root: Path,
    int8_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=False)
    built: dict[int, dict[str, Any]] = {}
    for block_index in FP16_BLOCKS:
        label, input_names, output_names = SPECS[block_index]
        raw_path = output_root / f".{block_index:02d}_{label}.raw.onnx"
        output_path = output_root / f"{block_index:02d}_{label}_fp16_ln32io32.onnx"
        onnx.utils.extract_model(
            str(extraction_source),
            str(raw_path),
            input_names,
            output_names,
            check_model=True,
            infer_shapes=False,
        )
        model = onnx.load(str(raw_path), load_external_data=False)
        onnx.checker.check_model(model)
        if [item.name for item in model.graph.input] != input_names:
            raise RuntimeError(f"FP16 extracted input drift for block {block_index}")
        if [item.name for item in model.graph.output] != output_names:
            raise RuntimeError(f"FP16 extracted output drift for block {block_index}")
        layernorm = rewrite_layernorm_statistic_islands(model, block_index)
        boundary_casts = externalize_fp32_io(model, block_index)
        add_metadata(model, block_index)
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path), save_as_external_data=False)
        raw_path.unlink()

        reloaded, node_types = audit_onnx(output_path)
        inputs, outputs = graph_contract(reloaded)
        if inputs != int8_rows[block_index]["inputs"] or outputs != int8_rows[block_index]["outputs"]:
            raise RuntimeError(
                f"FP16/INT8 segment boundary contract mismatch for block {block_index}: "
                f"FP16={inputs, outputs}; INT8={int8_rows[block_index]['inputs'], int8_rows[block_index]['outputs']}"
            )
        require_fp32_static_contract(inputs + outputs, f"generated FP16 block {block_index}")
        if node_types.get("LayerNormalization", 0):
            raise RuntimeError(f"generated FP16 block {block_index} retains LayerNormalization")
        if node_types.get("QuantizeLinear", 0) or node_types.get("DequantizeLinear", 0):
            raise RuntimeError(f"generated FP16 block {block_index} unexpectedly contains QDQ")
        initializer_types = Counter(int(item.data_type) for item in reloaded.graph.initializer)
        if initializer_types.get(TensorProto.FLOAT16, 0) == 0:
            raise RuntimeError(f"generated FP16 block {block_index} has no FP16 initializers")
        statistic_casts = [
            node
            for node in reloaded.graph.node
            if node.name.startswith(f"phase11_mixed_b{block_index:02d}_ln_")
            and node.op_type == "Cast"
        ]
        if len(statistic_casts) != 8:
            raise RuntimeError(
                f"generated FP16 block {block_index} must contain 8 LayerNorm-island Cast nodes; "
                f"found {len(statistic_casts)}"
            )
        built[block_index] = {
            "index": block_index,
            "label": label,
            "path": output_path.resolve(strict=True),
            "identity": identity(output_path),
            "inputs": inputs,
            "outputs": outputs,
            "node_types": node_types,
            "initializer_data_types": {str(key): value for key, value in sorted(initializer_types.items())},
            "layernorm": {
                "rewritten_count": len(layernorm),
                "remaining_count": node_types.get("LayerNormalization", 0),
                "changes": layernorm,
            },
            "boundary_casts": boundary_casts,
        }
        del reloaded
    return built


def candidate_segment(
    candidate_dir: Path,
    index: int,
    fp16_blocks: frozenset[int],
    int8_rows: list[dict[str, Any]],
    fp16_rows: dict[int, dict[str, Any]],
    head: dict[str, Any],
) -> dict[str, Any]:
    if index == 24:
        return {
            "index": 24,
            "label": "upernet_decoder_head_fpn4barrier",
            "precision": "fp32_compat_barrier",
            "model": relative_posix(head["model_path"], candidate_dir),
            "model_identity": head["model_identity"],
            "cache": relative_posix(head["cache_path"], candidate_dir),
            "cache_identity": head["cache_identity"],
            "cache_action": "reuse_validated_head",
            "inputs": head["inputs"],
            "outputs": head["outputs"],
            "source_lineage": "validated_int8_backbone_fp32_head_with_fpn4_maxpool_graph_output_barrier",
        }
    if index in fp16_blocks:
        row = fp16_rows[index]
        return {
            "index": index,
            "label": row["label"],
            "precision": "fp16",
            "model": relative_posix(row["path"], candidate_dir),
            "model_identity": row["identity"],
            "cache": None,
            "cache_identity": None,
            "cache_action": "compile_new_fp16",
            "inputs": row["inputs"],
            "outputs": row["outputs"],
            "source_lineage": "frozen_fp16_full_same_checkpoint_extracted_ln_fp32_island_io_fp32",
        }
    row = int8_rows[index]
    return {
        "index": index,
        "label": row["label"],
        "precision": "int8_qdq",
        "model": relative_posix(row["model_path"], candidate_dir),
        "model_identity": row["model_identity"],
        "cache": relative_posix(row["cache_path"], candidate_dir),
        "cache_identity": row["cache_identity"],
        "cache_action": "reuse_frozen_int8",
        "inputs": row["inputs"],
        "outputs": row["outputs"],
        "source_lineage": "frozen_int8_backbone_qdq_compat_segment",
    }


def build_candidate_manifests(
    staging_root: Path,
    selected: list[str],
    fp16_source_audit: dict[str, Any],
    provenance: dict[str, Any],
    int8_manifest_identity: dict[str, Any],
    int8_rows: list[dict[str, Any]],
    fp16_rows: dict[int, dict[str, Any]],
    head: dict[str, Any],
) -> list[dict[str, Any]]:
    manifests = []
    for candidate_id in selected:
        fp16_blocks = CANDIDATES[candidate_id]
        candidate_dir = staging_root / candidate_id
        candidate_dir.mkdir(parents=False, exist_ok=False)
        segments = [
            candidate_segment(candidate_dir, index, fp16_blocks, int8_rows, fp16_rows, head)
            for index in range(25)
        ]
        if [row["index"] for row in segments] != list(range(25)):
            raise RuntimeError(f"candidate {candidate_id} ordering failure")
        for previous, following in zip(segments, segments[1:24]):
            if previous["outputs"] != following["inputs"]:
                raise RuntimeError(
                    f"candidate {candidate_id} adjacent contract mismatch: "
                    f"{previous['index']} -> {following['index']}"
                )
        if [row["name"] for row in segments[24]["inputs"]] != HEAD_INPUTS:
            raise RuntimeError(f"candidate {candidate_id} head inputs drift")
        authorized_root = staging_root.resolve()
        for row in segments:
            runtime_paths = [row["model"]]
            if row["cache"] is not None:
                runtime_paths.append(row["cache"])
            for raw in runtime_paths:
                resolved = (candidate_dir / raw).resolve(strict=True)
                common = Path(os.path.commonpath((str(authorized_root), str(resolved))))
                if common != authorized_root:
                    raise RuntimeError(
                        f"candidate {candidate_id} runtime payload escapes bundle root: {resolved}"
                    )

        unique_models = {
            (row["model_identity"]["size_bytes"], row["model_identity"]["sha256"])
            for row in segments
        }
        unique_caches = {
            (row["cache_identity"]["size_bytes"], row["cache_identity"]["sha256"])
            for row in segments
            if row["cache_identity"] is not None
        }
        payload = {
            "schema": SCHEMA,
            "status": "created_static_pass",
            "candidate_id": candidate_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fp16_backbone_blocks": sorted(fp16_blocks),
            "int8_backbone_blocks": sorted(set(range(24)) - fp16_blocks),
            "head_precision": "fp32_compat_barrier",
            "segment_count": 25,
            "retain_encoder_outputs_after_segments": [5, 11, 17, 23],
            "external_io_contract": {
                "dtype": "float32",
                "batch": 1,
                "input": {"name": "image", "shape": [1, 6, 224, 224]},
                "output": {"name": "logits", "shape": [1, 2, 224, 224]},
                "intersegment_transport": "device_OrtValue_IOBinding",
            },
            "source_identities": {
                "fp16_full_model": {
                    **fp16_source_audit["identity"],
                    "runtime_payload": False,
                },
                "frozen_int8_bundle_manifest": {
                    **int8_manifest_identity,
                    "runtime_payload": False,
                },
                "int8_compat_full_model_not_bundled": {
                    "size_bytes": INT8_COMPAT_SOURCE[0],
                    "sha256": INT8_COMPAT_SOURCE[1],
                },
                "repaired_head_model": head["model_identity"],
                "repaired_head_cache": head["cache_identity"],
                "common_lineage": provenance,
            },
            "segments": segments,
            "unique_model_bytes": sum(size for size, _ in unique_models),
            "present_unique_cache_bytes": sum(size for size, _ in unique_caches),
            "new_fp16_cache_count_required": len(fp16_blocks),
            "claims": {
                "static_onnx_valid": True,
                "all_segment_identities_locked": True,
                "all_external_segment_io_fp32_static_batch1": True,
                "fp16_blocks_derived_from_frozen_same_checkpoint_source": True,
                "fp16_layernorm_fp32_statistic_islands_present": bool(fp16_blocks),
                "int8_segments_reused_byte_for_byte": True,
                "repaired_head_reused_byte_for_byte": True,
                "all_required_mxr_caches_present": not fp16_blocks,
                "cpu_sequential_finite": False,
                "single_sample_migraphx_admitted": False,
                "task_accuracy_90": False,
                "three_run_task_repeatability": False,
                "performance": False,
                "native_int8_kernel_verified": False,
                "deployment_ready": False,
            },
            "claim_boundary": (
                "Construction and static contracts only. New FP16 segments require fresh K100 MXR "
                "compilation, strict provider-placement checks, single-sample diagnostics, and fixed-90 "
                "task admission. Existing INT8 strict-logit failure remains frozen and is not overridden."
            ),
        }
        manifest_path = candidate_dir / "mixed_precision_manifest.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifests.append(
            {
                "candidate_id": candidate_id,
                "path": manifest_path,
                "identity": identity(manifest_path),
                "fp16_backbone_blocks": sorted(fp16_blocks),
            }
        )
    return manifests


def parse_candidates(raw: str) -> list[str]:
    selected = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("candidate list must be nonempty and contain no duplicates")
    unknown = sorted(set(selected) - CANDIDATES.keys())
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown candidates: {unknown}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-source", type=Path, required=True)
    parser.add_argument("--fp16-source-report", type=Path, required=True)
    parser.add_argument("--int8-quantization-report", type=Path, required=True)
    parser.add_argument("--frozen-int8-manifest", type=Path, required=True)
    parser.add_argument("--repaired-head-model", type=Path, required=True)
    parser.add_argument("--repaired-head-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidates", type=parse_candidates, default=list(CANDIDATES))
    parser.add_argument(
        "--payload-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Materialize immutable INT8/head runtime payload under output-root; hardlink is zero-copy and requires one filesystem.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(output_root.name + f".building-{os.getpid()}")
    if staging_root.exists():
        raise FileExistsError(f"staging root already exists: {staging_root}")
    staging_root.mkdir()

    try:
        provenance = audit_provenance(
            args.fp16_source_report.resolve(strict=True),
            args.int8_quantization_report.resolve(strict=True),
        )
        _, int8_manifest_identity, int8_rows = audit_int8_bundle(
            args.frozen_int8_manifest.resolve(strict=True)
        )
        head = audit_repaired_head(
            args.repaired_head_model,
            args.repaired_head_cache,
            int8_rows[24],
        )
        shared_root = staging_root / "_shared"
        shared_root.mkdir()
        int8_rows, head, materialization = materialize_frozen_payload(
            shared_root,
            int8_rows,
            head,
            args.payload_mode,
        )
        extraction_source, fp16_source_audit = prepare_extraction_source(
            args.fp16_source.resolve(strict=True), shared_root
        )
        fp16_rows = build_fp16_segments(
            extraction_source,
            shared_root / "fp16_segments",
            int8_rows,
        )
        if extraction_source.parent == shared_root and extraction_source.name.startswith(".fp16_source"):
            extraction_source.unlink()
        manifests = build_candidate_manifests(
            staging_root=staging_root,
            selected=args.candidates,
            fp16_source_audit=fp16_source_audit,
            provenance=provenance,
            int8_manifest_identity=int8_manifest_identity,
            int8_rows=int8_rows,
            fp16_rows=fp16_rows,
            head=head,
        )
        summary = {
            "schema": BUILD_SCHEMA,
            "status": "created_static_pass",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "candidates": [
                {
                    **{key: value for key, value in row.items() if key != "path"},
                    "path": str(row["path"].relative_to(staging_root)).replace("\\", "/"),
                }
                for row in manifests
            ],
            "shared_fp16_segments": [
                {
                    "index": index,
                    "label": row["label"],
                    "path": str(row["path"].relative_to(staging_root)).replace("\\", "/"),
                    "identity": row["identity"],
                    "layernorm": row["layernorm"],
                    "boundary_casts": row["boundary_casts"],
                }
                for index, row in sorted(fp16_rows.items())
            ],
            "runtime_payload_materialization": materialization,
            "claims": {
                "m0_m4_manifest_generation_complete": args.candidates == list(CANDIDATES),
                "source_lineage_audited": True,
                "frozen_int8_models_and_caches_audited": True,
                "repaired_head_model_and_cache_audited": True,
                "runtime_payload_materialized_beneath_output_root": True,
                "generated_fp16_segments_static_valid": True,
                "runtime_admission_complete": False,
            },
        }
        summary_path = staging_root / "build_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary_identity = identity(summary_path)
        staging_root.rename(output_root)
        print(
            json.dumps(
                {
                    "status": "created_static_pass",
                    "output_root": str(output_root),
                    "build_summary": {
                        "path": str(output_root / "build_summary.json"),
                        **summary_identity,
                    },
                    "candidate_manifests": [
                        {
                            "candidate_id": row["candidate_id"],
                            "path": str(output_root / row["candidate_id"] / "mixed_precision_manifest.json"),
                            **row["identity"],
                        }
                        for row in manifests
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        failure = {
            "schema": BUILD_SCHEMA,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_boundary": "No candidate from this partial staging directory is admitted.",
        }
        (staging_root / "BUILD_FAILED.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
