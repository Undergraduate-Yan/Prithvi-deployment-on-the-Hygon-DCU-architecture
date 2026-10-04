#!/usr/bin/env python3
'Research implementation: build fp16 b.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper


LN_SUFFIXES = {
    "CastInputFP32",
    "CastScaleFP32",
    "CastBiasFP32",
    "ReduceMean",
    "Sub",
    "Square",
    "Variance",
    "AddEpsilon",
    "Sqrt",
    "Div",
    "Scale",
    "Bias",
    "CastOutputFP16",
}


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
    for initializer in inferred.graph.initializer:
        result[initializer.name] = initializer.data_type
    return result


def cast_target(node: onnx.NodeProto) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "to":
            return int(attribute.i)
    return None


def graph_stats(model: onnx.ModelProto) -> dict[str, Any]:
    ops = Counter(node.op_type for node in model.graph.node)
    types = tensor_types(model)
    fp32_output_nodes = sum(
        any(types.get(name) == TensorProto.FLOAT for name in node.output) for node in model.graph.node
    )
    return {
        "node_count": len(model.graph.node),
        "Cast_count": ops["Cast"],
        "Resize_count": ops["Resize"],
        "fp32_output_node_count": fp32_output_nodes,
        "op_type_counts": dict(sorted(ops.items())),
        "initializer_types": dict(sorted(Counter(
            TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer
        ).items())),
    }


def validate_ln_group(prefix: str, nodes: dict[str, onnx.NodeProto]) -> None:
    if set(nodes) != LN_SUFFIXES:
        raise RuntimeError(f"LayerNorm group drift at {prefix}: {sorted(nodes)}")
    expected_ops = {
        "CastInputFP32": "Cast", "CastScaleFP32": "Cast", "CastBiasFP32": "Cast",
        "ReduceMean": "ReduceMean", "Sub": "Sub", "Square": "Mul", "Variance": "ReduceMean",
        "AddEpsilon": "Add", "Sqrt": "Sqrt", "Div": "Div", "Scale": "Mul", "Bias": "Add",
        "CastOutputFP16": "Cast",
    }
    for suffix, op in expected_ops.items():
        if nodes[suffix].op_type != op:
            raise RuntimeError(f"LayerNorm op drift at {prefix}/{suffix}")
    if cast_target(nodes["CastScaleFP32"]) != TensorProto.FLOAT:
        raise RuntimeError(f"LayerNorm scale cast drift at {prefix}")
    if cast_target(nodes["CastBiasFP32"]) != TensorProto.FLOAT:
        raise RuntimeError(f"LayerNorm bias cast drift at {prefix}")
    if cast_target(nodes["CastOutputFP16"]) != TensorProto.FLOAT16:
        raise RuntimeError(f"LayerNorm output cast drift at {prefix}")
    if list(nodes["Scale"].input) != [nodes["Div"].output[0], nodes["CastScaleFP32"].output[0]]:
        raise RuntimeError(f"LayerNorm scale wiring drift at {prefix}")
    if list(nodes["Bias"].input) != [nodes["Scale"].output[0], nodes["CastBiasFP32"].output[0]]:
        raise RuntimeError(f"LayerNorm bias wiring drift at {prefix}")
    if list(nodes["CastOutputFP16"].input) != [nodes["Bias"].output[0]]:
        raise RuntimeError(f"LayerNorm output wiring drift at {prefix}")


def rewrite_model(model: onnx.ModelProto, session_ordinal: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    before = graph_stats(model)
    types = tensor_types(model)
    graph_outputs = {value.name for value in model.graph.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    identity_map: dict[str, str] = {}
    identity_nodes: set[str] = set()
    rewrites: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "Cast" or "_boundary_" in node.name or not node.input or not node.output:
            continue
        source, target = node.input[0], node.output[0]
        if types.get(source) == cast_target(node) and target not in graph_outputs:
            identity_map[target] = source
            identity_nodes.add(node.name)
            rewrites.append({
                "session_ordinal": session_ordinal,
                "rewrite": "remove_provable_identity_cast",
                "node_name": node.name,
                "source_tensor": source,
                "removed_output_tensor": target,
                "reason": "inferred input dtype equals Cast target dtype; output is not a public graph output",
            })

    def canonical(name: str) -> str:
        seen = set()
        while name in identity_map:
            if name in seen:
                raise RuntimeError("identity Cast rewrite cycle")
            seen.add(name)
            name = identity_map[name]
        return name

    ln_groups: dict[str, dict[str, onnx.NodeProto]] = defaultdict(dict)
    for node in model.graph.node:
        if node.name.startswith("phase11_fp16_ln_") and "/" in node.name:
            prefix, suffix = node.name.rsplit("/", 1)
            if suffix in LN_SUFFIXES:
                ln_groups[prefix][suffix] = node
    for prefix, nodes in ln_groups.items():
        validate_ln_group(prefix, nodes)
    removed_ln_names = {
        node.name
        for nodes in ln_groups.values()
        for suffix, node in nodes.items()
        if suffix in {"CastScaleFP32", "CastBiasFP32", "Scale", "Bias", "CastOutputFP16"}
    }
    insertion_by_name: dict[str, list[onnx.NodeProto]] = {}
    for prefix, nodes in sorted(ln_groups.items()):
        norm32 = nodes["Div"].output[0]
        norm16 = prefix + "_normalized16"
        scaled16 = prefix + "_scaled16"
        scale16 = canonical(nodes["CastScaleFP32"].input[0])
        bias16 = canonical(nodes["CastBiasFP32"].input[0])
        final_output = nodes["CastOutputFP16"].output[0]
        insertion_by_name[nodes["Scale"].name] = [
            helper.make_node("Cast", [norm32], [norm16], name=prefix + "/CastNormalizedFP16", to=TensorProto.FLOAT16),
            helper.make_node("Mul", [norm16, scale16], [scaled16], name=prefix + "/ScaleFP16"),
            helper.make_node("Add", [scaled16, bias16], [final_output], name=prefix + "/BiasFP16"),
        ]
        rewrites.append({
            "session_ordinal": session_ordinal,
            "rewrite": "move_layernorm_affine_to_fp16",
            "node_name": prefix,
            "source_tensor": norm32,
            "removed_output_tensor": nodes["Bias"].output[0],
            "reason": "retain FP32 mean/variance/epsilon/sqrt/division; execute only affine scale and bias in FP16",
        })

    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in identity_nodes:
            continue
        if node.name in insertion_by_name:
            new_nodes.extend(insertion_by_name[node.name])
        if node.name in removed_ln_names:
            continue
        clone = onnx.NodeProto()
        clone.CopyFrom(node)
        for index, name in enumerate(clone.input):
            clone.input[index] = canonical(name)
        new_nodes.append(clone)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    removed_tensors = set(identity_map)
    for nodes in ln_groups.values():
        for suffix in ("CastScaleFP32", "CastBiasFP32", "Scale", "Bias"):
            removed_tensors.update(nodes[suffix].output)
    retained_value_info = [value for value in model.graph.value_info if value.name not in removed_tensors]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    onnx.checker.check_model(inferred, full_check=True)
    after = graph_stats(model)
    expected_identity = sum(row["rewrite"] == "remove_provable_identity_cast" for row in rewrites)
    expected_ln = len(ln_groups)
    if before["Cast_count"] - after["Cast_count"] != expected_identity + 2 * expected_ln:
        raise RuntimeError(f"Cast reduction accounting drift at session {session_ordinal}")
    if before["node_count"] - after["node_count"] != expected_identity + 2 * expected_ln:
        raise RuntimeError(f"node reduction accounting drift at session {session_ordinal}")
    return rewrites, before, after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    source_path = args.source_manifest.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("variant") != "RCS13-FP16-A" or source.get("session_count") != 13:
        raise RuntimeError("expected frozen RCS13-FP16-A source")
    args.output_root.mkdir(parents=True)
    models_root = args.output_root / "models"
    models_root.mkdir()

    sessions = []
    all_rewrites: list[dict[str, Any]] = []
    totals_before = Counter()
    totals_after = Counter()
    for source_session in source["topology"]["sessions"]:
        ordinal = int(source_session["ordinal"])
        model_path = Path(source_session["model"]["path"])
        if identity(model_path) != source_session["model"]:
            raise RuntimeError(f"source identity drift at session {ordinal}")
        model = onnx.load_model(str(model_path), load_external_data=True)
        rewrites, before, after = rewrite_model(model, ordinal)
        output_model = models_root / f"session_{ordinal:02d}.onnx"
        onnx.save_model(model, str(output_model))
        onnx.checker.check_model(str(output_model), full_check=True)
        all_rewrites.extend(rewrites)
        totals_before.update({key: before[key] for key in ("node_count", "Cast_count", "fp32_output_node_count")})
        totals_after.update({key: after[key] for key in ("node_count", "Cast_count", "fp32_output_node_count")})
        row = dict(source_session)
        row["source_session_id"] = source_session["session_id"]
        row["session_id"] = f"RCS13_FP16_B_{ordinal:02d}"
        row["source_model"] = source_session["model"]
        row["model"] = identity(output_model)
        row["graph_stats_before"] = before
        row["graph_stats"] = after
        row["phase3_rewrite_count"] = len(rewrites)
        sessions.append(row)
        print(f"RCS13-FP16-B session={ordinal:02d} checker passed rewrites={len(rewrites)}", flush=True)

    identity_count = sum(row["rewrite"] == "remove_provable_identity_cast" for row in all_rewrites)
    ln_count = sum(row["rewrite"] == "move_layernorm_affine_to_fp16" for row in all_rewrites)
    if identity_count != 31 or ln_count != 49:
        raise RuntimeError(f"frozen rewrite count drift: identity={identity_count}, layernorm={ln_count}")
    manifest = {
        "schema": "journal_phase3_rcs13_fp16_variant_manifest_v1",
        "status": "built_onnx_checked",
        "variant": "RCS13-FP16-B",
        "hardware": "海光 K100 AI 加速卡",
        "source_variant": "RCS13-FP16-A",
        "source_manifest": identity(source_path),
        "session_count": 13,
        "external_contract": source["external_contract"],
        "topology": {"sessions": sessions},
        "rewrite_summary": {
            "provable_identity_casts_removed": identity_count,
            "layernorm_groups_affine_moved_to_fp16": ln_count,
            "nodes_before": totals_before["node_count"],
            "nodes_after": totals_after["node_count"],
            "casts_before": totals_before["Cast_count"],
            "casts_after": totals_after["Cast_count"],
            "fp32_output_nodes_before": totals_before["fp32_output_node_count"],
            "fp32_output_nodes_after": totals_after["fp32_output_node_count"],
        },
        "preserved_fp32_scope": "LayerNorm mean, centered difference, square, variance, epsilon add, sqrt and division; common FP32 external/inter-Session I/O",
        "deferred_to_fp16_c": "UPerNet Resize compatibility barriers and remaining Head FP32 islands",
        "formal_90_image_test_used": False,
        "formal_90_image_test_access_count": 0,
    }
    manifest_path = args.output_root / "rcs13_fp16_b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_root / "fp16_b_rewrite_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rewrites[0]))
        writer.writeheader()
        writer.writerows(all_rewrites)
    report = [
        "# RCS13-FP16-B controlled graph build",
        "",
        "Status: **BUILT / 13 OF 13 ONNX CHECKER PASSED**",
        "",
        f"Provable identity Casts removed: {identity_count}.",
        f"LayerNorm groups with FP16 affine scale/bias: {ln_count}.",
        f"Cast count: {totals_before['Cast_count']} → {totals_after['Cast_count']}.",
        f"FP32-output node count: {totals_before['fp32_output_node_count']} → {totals_after['fp32_output_node_count']}.",
        "LayerNorm statistical computation remains FP32. Resize/Head compatibility barriers are unchanged and deferred to FP16-C.",
        "Formal 90-image test content was not accessed.",
    ]
    (args.output_root / "fp16_b_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(manifest["rewrite_summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
