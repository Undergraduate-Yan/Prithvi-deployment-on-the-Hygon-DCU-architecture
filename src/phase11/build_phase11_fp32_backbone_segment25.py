#!/usr/bin/env python3
"""Split the locked FP32 graph into 24 one-block models plus UPerNet head."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx

SOURCE = (1_277_071_980, "d7828912240f61ba3cfb9d27c787d90abb4a1b3428bfce8e9488aa0c1e6225d4")
BLOCK = {i: f"/task/model/encoder/blocks.{i}/Add_1_output_0" for i in range(24)}
SPECS = [(f"encoder_block_{i:02d}", ["image" if i == 0 else BLOCK[i - 1]], [BLOCK[i]]) for i in range(24)]
SPECS.append(("upernet_decoder_head", [BLOCK[5], BLOCK[11], BLOCK[17], BLOCK[23]], ["logits"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract(value) -> dict:
    tensor = value.type.tensor_type
    shape = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return {"name": value.name, "elem_type": int(tensor.elem_type), "shape": shape}


def staticize_batch(model) -> list[str]:
    changed = []
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        if tensor.HasField("shape") and tensor.shape.dim:
            dim = tensor.shape.dim[0]
            if not (dim.HasField("dim_value") and dim.dim_value == 1):
                dim.ClearField("dim_param")
                dim.dim_value = 1
                changed.append(value.name)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inferred-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    if (source.stat().st_size, sha256(source)) != SOURCE:
        raise RuntimeError("source identity drift")
    if args.output_dir.exists() or args.inferred_source.exists():
        raise FileExistsError("output directory or inferred source already exists")
    args.output_dir.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    source_model = onnx.load(str(source), load_external_data=False)
    onnx.checker.check_model(source_model)
    if [item.name for item in source_model.graph.input] != ["image"]:
        raise RuntimeError("source input drift")
    if [item.name for item in source_model.graph.output] != ["logits"]:
        raise RuntimeError("source output drift")
    node_outputs = {name for node in source_model.graph.node for name in node.output}
    missing = sorted({BLOCK[i] for i in range(24)} - node_outputs)
    if missing:
        raise RuntimeError(f"split tensors missing from graph: {missing}")
    del source_model

    # The exported FP32 graph does not carry intermediate value_info. Infer once,
    # then extract all 25 segments from the same frozen inferred graph.
    onnx.shape_inference.infer_shapes_path(str(source), str(args.inferred_source), data_prop=False)
    inferred = onnx.load(str(args.inferred_source), load_external_data=False)
    onnx.checker.check_model(inferred)
    known = {item.name for item in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    needed = {name for _, inputs, outputs in SPECS for name in [*inputs, *outputs]}
    if needed - known:
        raise RuntimeError(f"shape inference did not expose split tensors: {sorted(needed-known)}")
    inferred_identity = {
        "path": str(args.inferred_source.resolve()),
        "size_bytes": args.inferred_source.stat().st_size,
        "sha256": sha256(args.inferred_source),
    }
    del inferred

    rows = []
    for order, (label, inputs, outputs) in enumerate(SPECS):
        output = args.output_dir / f"{order:02d}_{label}.onnx"
        onnx.utils.extract_model(
            str(args.inferred_source), str(output), inputs, outputs,
            check_model=True, infer_shapes=False,
        )
        extracted = onnx.load(str(output), load_external_data=False)
        changed = staticize_batch(extracted)
        onnx.save(extracted, str(output), save_as_external_data=False)
        extracted = onnx.load(str(output), load_external_data=False)
        onnx.checker.check_model(extracted)
        if [item.name for item in extracted.graph.input] != inputs:
            raise RuntimeError(f"{label} input drift")
        if [item.name for item in extracted.graph.output] != outputs:
            raise RuntimeError(f"{label} output drift")
        if any(item.type.tensor_type.shape.dim[0].dim_value != 1 for item in [*extracted.graph.input, *extracted.graph.output]):
            raise RuntimeError(f"{label} batch is not static 1")
        rows.append({
            "order": order,
            "label": label,
            "path": str(output.resolve()),
            "size_bytes": output.stat().st_size,
            "sha256": sha256(output),
            "node_count": len(extracted.graph.node),
            "initializer_count": len(extracted.graph.initializer),
            "staticized_batch_values": changed,
            "inputs": [contract(item) for item in extracted.graph.input],
            "outputs": [contract(item) for item in extracted.graph.output],
        })

    report = {
        "status": "created_static_pass",
        "variant": "fp32_25_static_batch1_onnx_segments",
        "operation": "split_into_24x1_transformer_blocks_plus_upernet_decoder_head",
        "source": {"path": str(source), "size_bytes": SOURCE[0], "sha256": SOURCE[1]},
        "shape_inferred_source": inferred_identity,
        "segments": rows,
        "execution_order": [row["label"] for row in rows],
        "semantic_change_expected": False,
        "claims": {
            "cpu_sequential_parity": False,
            "strict_migraphx_admission": False,
            "device_resident_intersegment_io": False,
            "task_accuracy_90": False,
            "performance": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
