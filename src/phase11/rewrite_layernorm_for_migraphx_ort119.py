#!/usr/bin/env python3
"""Replace ONNX LayerNormalization with an equivalent primitive-op subgraph.

This creates a new compatibility artifact.  The source model is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise SystemExit("source and output must differ")
    model = onnx.load(str(args.source), load_external_data=True)
    before = len(model.graph.node)
    existing = {x.name for x in model.graph.initializer}
    replacement = []
    changed = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization":
            replacement.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        axis = int(attrs.get("axis", -1))
        eps = float(attrs.get("epsilon", 1e-5))
        stash_type = int(attrs.get("stash_type", TensorProto.FLOAT))
        if axis != -1 or stash_type != TensorProto.FLOAT or len(node.input) != 3 or len(node.output) != 1:
            raise RuntimeError(f"unsupported LayerNormalization contract at {node.name}: axis={axis}, stash={stash_type}")
        base = f"phase11_ln_{index}"
        eps_name = base + "_epsilon"
        if eps_name in existing:
            raise RuntimeError(f"name collision: {eps_name}")
        existing.add(eps_name)
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(eps, dtype=np.float32), eps_name))
        x, scale, bias = node.input
        y = node.output[0]
        mean, centered, squared = base+"_mean", base+"_centered", base+"_squared"
        variance, var_eps, std = base+"_variance", base+"_var_eps", base+"_std"
        norm, scaled = base+"_norm", base+"_scaled"
        replacement.extend([
            helper.make_node("ReduceMean", [x], [mean], name=base+"/ReduceMean", axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, mean], [centered], name=base+"/Sub"),
            helper.make_node("Mul", [centered, centered], [squared], name=base+"/Square"),
            helper.make_node("ReduceMean", [squared], [variance], name=base+"/Variance", axes=[-1], keepdims=1),
            helper.make_node("Add", [variance, eps_name], [var_eps], name=base+"/AddEpsilon"),
            helper.make_node("Sqrt", [var_eps], [std], name=base+"/Sqrt"),
            helper.make_node("Div", [centered, std], [norm], name=base+"/Div"),
            helper.make_node("Mul", [norm, scale], [scaled], name=base+"/Scale"),
            helper.make_node("Add", [scaled, bias], [y], name=base+"/Bias"),
        ])
        changed.append({"index": index, "name": node.name, "axis": axis, "epsilon": eps})
    if len(changed) != 49:
        raise RuntimeError(f"expected 49 LayerNormalization nodes, found {len(changed)}")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output), save_as_external_data=False)
    reloaded = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(reloaded)
    remaining = sum(n.op_type == "LayerNormalization" for n in reloaded.graph.node)
    if remaining:
        raise RuntimeError(f"LayerNormalization nodes remain: {remaining}")
    report = {
        "status": "created_static_pass",
        "transformation": "LayerNormalization(axis=-1) -> ReduceMean/Sub/Mul/ReduceMean/Add/Sqrt/Div/Mul/Add",
        "source": {"path": str(args.source), "size_bytes": args.source.stat().st_size, "sha256": sha256(args.source)},
        "output": {"path": str(args.output), "size_bytes": args.output.stat().st_size, "sha256": sha256(args.output)},
        "nodes_before": before,
        "nodes_after": len(reloaded.graph.node),
        "layernorm_replaced": len(changed),
        "layernorm_remaining": remaining,
        "changed_nodes": changed,
        "claims": {"numeric_equivalence": False, "migraphx_admission": False, "deployment": False},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
