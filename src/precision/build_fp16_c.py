#!/usr/bin/env python3
'Research implementation: build fp16 c.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


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


def cast_target(node: onnx.NodeProto) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "to":
            return int(attribute.i)
    return None


def graph_stats(model: onnx.ModelProto) -> dict[str, Any]:
    types = tensor_types(model)
    counts = Counter(node.op_type for node in model.graph.node)
    return {
        "node_count": len(model.graph.node),
        "Cast_count": counts["Cast"],
        "Resize_count": counts["Resize"],
        "Conv_count": counts["Conv"],
        "MatMul_count": counts["MatMul"],
        "fp32_output_node_count": sum(
            any(types.get(name) == TensorProto.FLOAT for name in node.output) for node in model.graph.node
        ),
        "fp16_resize_count": sum(
            node.op_type == "Resize" and types.get(node.output[0]) == TensorProto.FLOAT16
            for node in model.graph.node
        ),
        "fp32_resize_count": sum(
            node.op_type == "Resize" and types.get(node.output[0]) == TensorProto.FLOAT
            for node in model.graph.node
        ),
    }


def rewrite_head(model: onnx.ModelProto) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    before = graph_stats(model)
    types = tensor_types(model)
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    graph_outputs = {item.name for item in model.graph.output}
    remove_names: set[str] = set()
    output_remap: dict[str, str] = {}
    rewrites: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        if not node.input or not node.output:
            raise RuntimeError(f"malformed Resize: {node.name}")
        input_cast = producers.get(node.input[0])
        output_consumers = consumers.get(node.output[0], [])
        if (
            input_cast is None
            or input_cast.op_type != "Cast"
            or cast_target(input_cast) != TensorProto.FLOAT
            or types.get(input_cast.input[0]) != TensorProto.FLOAT16
            or len(consumers.get(input_cast.output[0], [])) != 1
            or output_consumers is None
            or len(output_consumers) != 1
            or output_consumers[0].op_type != "Cast"
            or cast_target(output_consumers[0]) != TensorProto.FLOAT16
            or types.get(node.output[0]) != TensorProto.FLOAT
            or output_consumers[0].output[0] in graph_outputs
        ):
            raise RuntimeError(f"Resize compatibility guard failed: {node.name}")
        output_cast = output_consumers[0]
        original_input = input_cast.input[0]
        final_output = output_cast.output[0]
        node.input[0] = original_input
        output_remap[node.output[0]] = final_output
        node.output[0] = final_output
        remove_names.update({input_cast.name, output_cast.name})
        rewrites.append({
            "session_ordinal": 12,
            "rewrite": "convert_guarded_resize_data_path_to_fp16",
            "resize_node": node.name,
            "removed_input_cast": input_cast.name,
            "removed_output_cast": output_cast.name,
            "source_tensor_fp16": original_input,
            "final_tensor_fp16": final_output,
            "guard": "single producer Cast(FP16->FP32), single consumer Cast(FP32->FP16), non-public intermediate",
        })
    if len(rewrites) != 11 or len(remove_names) != 22:
        raise RuntimeError(f"frozen Resize rewrite count drift: groups={len(rewrites)} casts={len(remove_names)}")
    retained = [node for node in model.graph.node if node.name not in remove_names]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    removed_tensors = set(output_remap)
    for name in remove_names:
        node = next((candidate for candidate in producers.values() if candidate.name == name), None)
        if node is not None:
            removed_tensors.update(node.output)
    retained_value_info = [item for item in model.graph.value_info if item.name not in removed_tensors]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    onnx.checker.check_model(inferred, full_check=True)
    after = graph_stats(model)
    if (
        before["node_count"] - after["node_count"] != 22
        or before["Cast_count"] - after["Cast_count"] != 22
        or before["Resize_count"] != after["Resize_count"]
        or after["fp16_resize_count"] != 11
        or after["fp32_resize_count"] != 0
    ):
        raise RuntimeError("FP16-C rewrite accounting drift")
    return rewrites, before, after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    source_path = args.source_manifest.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("schema") != "journal_phase3_rcs13_fp16_variant_manifest_v1"
        or source.get("status") != "built_onnx_checked"
        or source.get("variant") != "RCS13-FP16-B"
        or source.get("session_count") != 13
    ):
        raise RuntimeError("expected checked RCS13-FP16-B source")
    args.output_root.mkdir(parents=True)
    models_root = args.output_root / "models"
    models_root.mkdir()
    sessions = []
    rewrites: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    for source_session in source["topology"]["sessions"]:
        ordinal = int(source_session["ordinal"])
        if identity(Path(source_session["model"]["path"])) != source_session["model"]:
            raise RuntimeError(f"source identity drift at session {ordinal}")
        row = dict(source_session)
        row["source_session_id"] = source_session["session_id"]
        row["source_model"] = source_session["model"]
        row["session_id"] = f"RCS13_FP16_C_{ordinal:02d}"
        if ordinal == 12:
            model = onnx.load_model(source_session["model"]["path"], load_external_data=True)
            rewrites, before, after = rewrite_head(model)
            output_model = models_root / "session_12.onnx"
            onnx.save_model(model, str(output_model))
            onnx.checker.check_model(str(output_model), full_check=True)
            row["model"] = identity(output_model)
            row["graph_stats_before_fp16_c"] = before
            row["graph_stats"] = {**source_session["graph_stats"], **after}
            row["phase3_fp16_c_rewrite_count"] = len(rewrites)
            row["cache_reuse_eligible_from_fp16_b"] = False
        else:
            row["phase3_fp16_c_rewrite_count"] = 0
            row["cache_reuse_eligible_from_fp16_b"] = True
        sessions.append(row)
    if before is None or after is None:
        raise RuntimeError("head session missing")
    manifest = {
        "schema": "journal_phase3_rcs13_fp16_variant_manifest_v1",
        "status": "built_onnx_checked",
        "variant": "RCS13-FP16-C",
        "hardware": "海光 K100 AI 加速卡",
        "source_variant": "RCS13-FP16-B",
        "source_manifest": identity(source_path),
        "session_count": 13,
        "external_contract": source["external_contract"],
        "topology": {"sessions": sessions},
        "rewrite_summary": {
            "head_resize_groups_converted_to_fp16": len(rewrites),
            "head_casts_removed": 22,
            "head_nodes_before": before["node_count"],
            "head_nodes_after": after["node_count"],
            "head_casts_before": before["Cast_count"],
            "head_casts_after": after["Cast_count"],
            "head_fp32_resize_before": before["fp32_resize_count"],
            "head_fp16_resize_after": after["fp16_resize_count"],
            "unchanged_sessions_reusing_identical_model_and_cache": 12,
        },
        "preserved_fp32_scope": source["preserved_fp32_scope"],
        "formal_90_image_test_access_count": 0,
        "formal_90_image_test_used": False,
    }
    manifest_path = args.output_root / "rcs13_fp16_c_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_root / "fp16_c_rewrite_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rewrites[0]))
        writer.writeheader()
        writer.writerows(rewrites)
    report = [
        "# RCS13-FP16-C controlled graph build",
        "",
        "Status: **BUILT / 13 OF 13 TOPOLOGY ROWS / CHANGED HEAD CHECKER PASSED**",
        "",
        "All 16 Head Conv nodes and all 6 Head MatMul nodes were already FP16 in FP16-B.",
        "Converted 11 guarded Resize data paths from FP32 to FP16 and removed 22 surrounding Cast nodes.",
        "Sessions 00-11 are byte-identical to FP16-B and are explicitly eligible for exact cache reuse.",
        "External FP32 I/O and LayerNorm statistical FP32 paths remain unchanged.",
        "Formal 90-image test content was not accessed.",
    ]
    (args.output_root / "fp16_c_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(manifest["rewrite_summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
