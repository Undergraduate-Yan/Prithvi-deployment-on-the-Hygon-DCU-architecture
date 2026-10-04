#!/usr/bin/env python3
'Research implementation: clean merged onnx.'

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def audit(model: onnx.ModelProto) -> dict[str, object]:
    counts = Counter(node.op_type for node in model.graph.node)
    initializer_bytes = sum(len(item.raw_data) for item in model.graph.initializer)
    dynamic_dims = 0
    static_dims = 0
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                static_dims += 1
            else:
                dynamic_dims += 1
    return {
        "nodes": len(model.graph.node),
        "QuantizeLinear": counts["QuantizeLinear"],
        "DequantizeLinear": counts["DequantizeLinear"],
        "Cast": counts["Cast"],
        "Identity": counts["Identity"],
        "MatMul": counts["MatMul"],
        "Gemm": counts["Gemm"],
        "Conv": counts["Conv"],
        "initializers": len(model.graph.initializer),
        "initializer_raw_bytes": initializer_bytes,
        "graph_inputs": len(model.graph.input),
        "graph_outputs": len(model.graph.output),
        "value_info": len(model.graph.value_info),
        "static_dimensions": static_dims,
        "dynamic_dimensions": dynamic_dims,
    }


def replace_inputs(model: onnx.ModelProto, old: str, new: str) -> int:
    changed = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
                changed += 1
    return changed


def remove_safe_identities(model: onnx.ModelProto) -> int:
    graph_outputs = {item.name for item in model.graph.output}
    removed = 0
    kept = []
    for node in model.graph.node:
        if (
            node.op_type == "Identity"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.output[0] not in graph_outputs
        ):
            replace_inputs(model, node.output[0], node.input[0])
            removed += 1
        else:
            kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def remove_unreachable_nodes(model: onnx.ModelProto) -> int:
    producers = {output: index for index, node in enumerate(model.graph.node) for output in node.output}
    needed_nodes: set[int] = set()
    pending = [item.name for item in model.graph.output]
    seen_values: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen_values:
            continue
        seen_values.add(name)
        index = producers.get(name)
        if index is None or index in needed_nodes:
            continue
        needed_nodes.add(index)
        pending.extend(model.graph.node[index].input)
    old_count = len(model.graph.node)
    kept = [node for index, node in enumerate(model.graph.node) if index in needed_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return old_count - len(kept)


def remove_unused_initializers(model: onnx.ModelProto) -> int:
    used = {name for node in model.graph.node for name in node.input}
    used.update(item.name for item in model.graph.output)
    old_count = len(model.graph.initializer)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return old_count - len(kept)


def deduplicate_value_info(model: onnx.ModelProto) -> int:
    seen: set[str] = set()
    kept = []
    for value in model.graph.value_info:
        if value.name in seen:
            continue
        seen.add(value.name)
        kept.append(value)
    removed = len(model.graph.value_info) - len(kept)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    return removed


def compare_cpu(raw: Path, clean: Path, feeds_path: Path, shim: Path) -> dict[str, object]:
    resolved_shim = shim.resolve(strict=True)
    os.environ["PATH"] = os.pathsep.join(
        [str(resolved_shim.parent), os.environ.get("PATH", "")]
    )
    if Path(shutil.which("lsmod") or "").resolve(strict=False) != resolved_shim:
        raise RuntimeError("static lsmod shim is not first on PATH")
    import onnxruntime as ort

    with np.load(feeds_path, allow_pickle=False) as packed:
        feeds = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    raw_session = ort.InferenceSession(str(raw), sess_options=options, providers=["CPUExecutionProvider"])
    clean_session = ort.InferenceSession(str(clean), sess_options=options, providers=["CPUExecutionProvider"])
    raw_names = [item.name for item in raw_session.get_outputs()]
    clean_names = [item.name for item in clean_session.get_outputs()]
    if raw_names != clean_names:
        raise RuntimeError(f"output contract changed: {raw_names} != {clean_names}")
    raw_outputs = raw_session.run(raw_names, feeds)
    clean_outputs = clean_session.run(clean_names, feeds)
    max_abs = 0.0
    absolute_sum = 0.0
    squared_diff = 0.0
    squared_ref = 0.0
    count = 0
    finite = True
    exact = True
    agreements = []
    for left, right in zip(raw_outputs, clean_outputs):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError("shape or dtype contract changed")
        finite = finite and bool(np.isfinite(left).all() and np.isfinite(right).all())
        exact = exact and bool(np.array_equal(left, right))
        diff = left.astype(np.float64) - right.astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(diff), initial=0.0)))
        absolute_sum += float(np.abs(diff).sum())
        squared_diff += float(np.square(diff).sum())
        squared_ref += float(np.square(left.astype(np.float64)).sum())
        count += diff.size
        if left.ndim >= 2 and left.shape[1] > 1:
            agreements.append(float(np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1))))
    return {
        "all_finite": finite,
        "bitwise_equal": exact,
        "mae": absolute_sum / max(count, 1),
        "max_abs": max_abs,
        "relative_l2": (squared_diff / max(squared_ref, np.finfo(np.float64).tiny)) ** 0.5,
        "class_agreement": min(agreements) if agreements else None,
        "output_names": raw_names,
        "passed": finite and exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feeds", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("clean output/report already exists")
    model = onnx.load(str(args.input.resolve(strict=True)), load_external_data=True)
    onnx.checker.check_model(model)
    raw_audit = audit(model)
    transformations = {
        "safe_identity_nodes_removed": remove_safe_identities(model),
        "unreachable_nodes_removed": remove_unreachable_nodes(model),
        "unused_initializers_removed": remove_unused_initializers(model),
        "duplicate_value_info_removed": deduplicate_value_info(model),
        "dq_q_pairs_folded": 0,
        "cast_pairs_folded": 0,
    }
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    check = compare_cpu(
        args.input.resolve(),
        args.output.resolve(),
        args.feeds.resolve(strict=True),
        args.shim,
    )
    report = {
        "schema": "journal_phase2_clean_merged_onnx_v1",
        "status": "passed" if check["passed"] else "failed",
        "hardware_target": "海光 K100 AI 加速卡",
        "formal_90_image_test_used": False,
        "input": tensor_identity(args.input),
        "output": tensor_identity(args.output),
        "raw_audit": raw_audit,
        "clean_audit": audit(model),
        "transformations": transformations,
        "cpu_raw_vs_clean": check,
        "scope_note": "Phase2B representative feed only; full configuration-validation audit remains Phase2C.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
