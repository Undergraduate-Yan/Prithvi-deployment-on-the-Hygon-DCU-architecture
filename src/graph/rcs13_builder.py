#!/usr/bin/env python3
"""Extract one frozen 13-Session topology from a monolithic FP32/FP16 ONNX."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def tensor_elem_type(value_info: onnx.ValueInfoProto) -> int:
    return value_info.type.tensor_type.elem_type


def set_tensor_elem_type(value_info: onnx.ValueInfoProto, elem_type: int) -> None:
    value_info.type.tensor_type.elem_type = elem_type


def rename_tensor(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new
    for collection in (model.graph.value_info,):
        for value in collection:
            if value.name == old:
                value.name = new


def enforce_fp32_boundaries(model: onnx.ModelProto, session_id: str) -> list[dict[str, Any]]:
    repairs = []
    prefix_nodes = []
    suffix_nodes = []
    for value in model.graph.input:
        original = tensor_elem_type(value)
        if original == TensorProto.FLOAT16:
            public = value.name
            internal = public + "__rcs13_fp16_internal"
            rename_tensor(model, public, internal)
            value.name = public
            set_tensor_elem_type(value, TensorProto.FLOAT)
            prefix_nodes.append(
                helper.make_node(
                    "Cast",
                    [public],
                    [internal],
                    name=f"{session_id}_boundary_input_{len(prefix_nodes)}",
                    to=TensorProto.FLOAT16,
                )
            )
            repairs.append(
                {
                    "tensor": public,
                    "direction": "FP32_public_to_FP16_internal",
                    "reason": "common C++ Runner FP32 inter-Session OrtValue contract",
                }
            )
        elif original != TensorProto.FLOAT:
            raise RuntimeError(f"unsupported public input element type {original}: {value.name}")
    for value in model.graph.output:
        original = tensor_elem_type(value)
        if original == TensorProto.FLOAT16:
            public = value.name
            internal = public + "__rcs13_fp16_internal_output"
            rename_tensor(model, public, internal)
            value.name = public
            set_tensor_elem_type(value, TensorProto.FLOAT)
            suffix_nodes.append(
                helper.make_node(
                    "Cast",
                    [internal],
                    [public],
                    name=f"{session_id}_boundary_output_{len(suffix_nodes)}",
                    to=TensorProto.FLOAT,
                )
            )
            repairs.append(
                {
                    "tensor": public,
                    "direction": "FP16_internal_to_FP32_public",
                    "reason": "common C++ Runner FP32 inter-Session OrtValue contract",
                }
            )
        elif original != TensorProto.FLOAT:
            raise RuntimeError(f"unsupported public output element type {original}: {value.name}")
    if prefix_nodes:
        original_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(prefix_nodes + original_nodes + suffix_nodes)
    elif suffix_nodes:
        model.graph.node.extend(suffix_nodes)
    return repairs


def graph_stats(model: onnx.ModelProto) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    initializer_types: dict[str, int] = {}
    for item in model.graph.initializer:
        key = onnx.TensorProto.DataType.Name(item.data_type)
        initializer_types[key] = initializer_types.get(key, 0) + 1
    return {
        "node_count": len(model.graph.node),
        "op_type_counts": counts,
        "QuantizeLinear_count": counts.get("QuantizeLinear", 0),
        "DequantizeLinear_count": counts.get("DequantizeLinear", 0),
        "Cast_count": counts.get("Cast", 0),
        "Transpose_count": counts.get("Transpose", 0),
        "graph_input_count": len(model.graph.input),
        "graph_output_count": len(model.graph.output),
        "initializer_count": len(model.graph.initializer),
        "initializer_types": initializer_types,
    }


def inject_required_value_info(
    model: onnx.ModelProto, topology: dict[str, Any], variant: str
) -> list[dict[str, Any]]:
    """Register frozen internal tensors that ONNX shape inference leaves unlisted."""
    known = {
        value.name
        for collection in (model.graph.input, model.graph.output, model.graph.value_info)
        for value in collection
    }
    produced = {name for node in model.graph.node for name in node.output}
    consumed = {name for node in model.graph.node for name in node.input}
    required: dict[str, list[int]] = {}
    for row in topology["topology"]["sessions"]:
        for item in row["inputs"] + row["outputs"]:
            required[item["name"]] = item["shape"]
    added = []
    for name, shape in required.items():
        if name in known:
            continue
        if name not in produced and name not in consumed:
            raise RuntimeError(f"frozen logical tensor is absent from monolithic source: {name}")
        elem_type = TensorProto.FLOAT
        if variant == "RCS13-FP16-A" and name != "image" and name != "logits":
            elem_type = TensorProto.FLOAT16
        model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))
        added.append(
            {
                "name": name,
                "shape": shape,
                "elem_type": onnx.TensorProto.DataType.Name(elem_type),
                "reason": "required frozen RCS13 internal boundary omitted by source shape metadata",
            }
        )
    return added


def fp32_islands(model: onnx.ModelProto, session_id: str) -> list[dict[str, Any]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    types: dict[str, int] = {}
    for collection in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info):
        for value in collection:
            if value.type.HasField("tensor_type"):
                types[value.name] = value.type.tensor_type.elem_type
    rows = []
    for ordinal, node in enumerate(inferred.graph.node):
        fp32_outputs = [name for name in node.output if types.get(name) == TensorProto.FLOAT]
        if not fp32_outputs:
            continue
        text = " ".join([node.name, *node.input, *node.output]).lower()
        layernorm = "norm" in text or node.op_type in {"ReduceMean", "Sqrt"}
        boundary = node.name.startswith(session_id + "_boundary_")
        if boundary:
            reason = "FP32 public inter-Session compatibility boundary"
        elif layernorm:
            reason = "retained FP32 LayerNorm/statistical stability path"
        else:
            reason = "retained FP32 compatibility island from validated Mono-FP16 conversion"
        rows.append(
            {
                "session_id": session_id,
                "node_ordinal": ordinal,
                "node_name": node.name or f"unnamed_{ordinal}",
                "op_type": node.op_type,
                "fp32_outputs": ";".join(fp32_outputs),
                "reason": reason,
                "layernorm_statistics": layernorm,
                "compatibility_repair": boundary or not layernorm,
                "phase3_optimization_candidate": not boundary,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("RCS13-FP32", "RCS13-FP16-A"))
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    source = args.source_model.resolve(strict=True)
    if sha256(source) != args.source_sha256:
        raise RuntimeError("monolithic source identity drift")
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    if topology.get("status") != "passed" or topology.get("session_count") != 13:
        raise RuntimeError("frozen RCS13 topology gate failed")
    args.output_root.mkdir(parents=True)
    models_root = args.output_root / "models"
    feeds_root = args.output_root / "compile_feeds"
    models_root.mkdir()
    feeds_root.mkdir()
    print(f"loading {source}", flush=True)
    source_model = onnx.load_model(str(source), load_external_data=True)
    onnx.checker.check_model(source_model)
    injected_value_info = inject_required_value_info(source_model, topology, args.variant)
    extractor = onnx.utils.Extractor(source_model)
    model_rows = []
    all_islands = []
    for row in topology["topology"]["sessions"]:
        ordinal = int(row["ordinal"])
        session_id = f"{args.variant.replace('-', '_')}_{ordinal:02d}"
        input_names = [item["name"] for item in row["inputs"]]
        output_names = [item["name"] for item in row["outputs"]]
        print(f"extracting {session_id}: {input_names} -> {output_names}", flush=True)
        extracted = extractor.extract_model(input_names, output_names)
        repairs = enforce_fp32_boundaries(extracted, session_id)
        onnx.checker.check_model(extracted, full_check=True)
        model_path = models_root / f"session_{ordinal:02d}.onnx"
        onnx.save_model(extracted, str(model_path))
        onnx.checker.check_model(str(model_path), full_check=True)
        stats = graph_stats(extracted)
        islands = fp32_islands(extracted, session_id) if args.variant == "RCS13-FP16-A" else []
        all_islands.extend(islands)
        feed_path = feeds_root / f"session_{ordinal:02d}.npz"
        feeds = {
            item["name"]: np.zeros(item["shape"], dtype=np.float32)
            for item in row["inputs"]
        }
        np.savez(feed_path, **feeds)
        model_rows.append(
            {
                "ordinal": ordinal,
                "session_id": session_id,
                "start_block": row["start_block"],
                "end_block": row["end_block"],
                "logical_role": row["logical_role"],
                "inputs": row["inputs"],
                "outputs": row["outputs"],
                "multiscale_tap_outputs": row["multiscale_tap_outputs"],
                "model": identity(model_path),
                "compile_feeds": identity(feed_path),
                "boundary_repairs": repairs,
                "graph_stats": stats,
                "onnx_checker": "passed",
            }
        )
        del extracted
    del extractor
    del source_model
    manifest = {
        "schema": "journal_pre_phase3_rcs13_variant_manifest_v1",
        "status": "built_onnx_checked",
        "variant": args.variant,
        "hardware": "海光 K100 AI 加速卡",
        "precision_role": "all_fp32" if args.variant == "RCS13-FP32" else "compatibility_fp16_baseline",
        "source_model": identity(source),
        "source_topology": identity(args.topology),
        "injected_source_value_info": injected_value_info,
        "session_count": len(model_rows),
        "external_contract": topology["external_contract"],
        "topology": {"sessions": model_rows},
        "formal_90_image_test_used": False,
    }
    manifest_path = args.output_root / (
        "rcs13_fp32_manifest.json" if args.variant == "RCS13-FP32" else "rcs13_fp16_a_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    precision_path = args.output_root / (
        "rcs13_fp32_precision_map.json"
        if args.variant == "RCS13-FP32"
        else "rcs13_fp16_a_precision_map.json"
    )
    precision_path.write_text(
        json.dumps(
            {
                "schema": "journal_pre_phase3_rcs13_precision_map_v1",
                "status": "frozen_pre_phase3_baseline",
                "variant": args.variant,
                "blocks": [
                    {"block": block, "precision": "fp32" if args.variant == "RCS13-FP32" else "fp16_compatible"}
                    for block in range(24)
                ],
                "head": "fp32" if args.variant == "RCS13-FP32" else "validated_compatibility_fp16_with_fp32_islands",
                "external_and_inter_session_io": "fp32",
                "formal_90_image_test_used": False,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.variant == "RCS13-FP16-A":
        island_path = args.output_root / "rcs13_fp16_a_fp32_islands.csv"
        fields = (
            "session_id",
            "node_ordinal",
            "node_name",
            "op_type",
            "fp32_outputs",
            "reason",
            "layernorm_statistics",
            "compatibility_repair",
            "phase3_optimization_candidate",
        )
        with island_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_islands)
    report_name = "rcs13_fp32_build_report.md" if args.variant == "RCS13-FP32" else "rcs13_fp16_a_build_report.md"
    report = [
        f"# {args.variant} ONNX build report",
        "",
        "Status: **BUILT / ONNX CHECKER PASSED**",
        "",
        "The frozen B8 logical 13-Session boundaries were reused without search or modification.",
        f"Source monolithic model: `{source}` (`{args.source_sha256}`).",
        f"All {len(model_rows)} extracted ONNX Sessions passed full ONNX checker validation.",
        f"Boundary compatibility repairs inserted: {sum(len(row['boundary_repairs']) for row in model_rows)}.",
        "Formal 90-image test content was not accessed.",
    ]
    (args.output_root / report_name).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "variant": args.variant,
                "sessions": len(model_rows),
                "FP32_island_rows": len(all_islands),
                "formal_90_image_test_used": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
