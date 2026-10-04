#!/usr/bin/env python3
'Research implementation: partition flood int8.'
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx

SOURCE = (368_047_490, "edd2c3e69b64986dc6d0b78fe32230f9a9f7eb0396c2d77fdcb6bf4c59506a7a")
BLOCK = {i: f"/task/model/encoder/blocks.{i}/Add_1_output_0" for i in range(24)}
SPECS = [(f"encoder_block_{i:02d}", ["image" if i == 0 else BLOCK[i - 1]], [BLOCK[i]]) for i in range(24)]
SPECS.append(("upernet_decoder_head", [BLOCK[5], BLOCK[11], BLOCK[17], BLOCK[23]], ["logits"]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contract(value) -> dict:
    tensor = value.type.tensor_type; shape = []
    for dim in tensor.shape.dim:
        shape.append(int(dim.dim_value) if dim.HasField("dim_value") else str(dim.dim_param) if dim.HasField("dim_param") else None)
    return {"name": value.name, "elem_type": int(tensor.elem_type), "shape": shape}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--report", type=Path, required=True); args = ap.parse_args()
    source = args.source.resolve(strict=True)
    if (source.stat().st_size, sha256(source)) != SOURCE: raise RuntimeError("source identity drift")
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source), load_external_data=False); onnx.checker.check_model(model)
    if [x.name for x in model.graph.input] != ["image"] or [x.name for x in model.graph.output] != ["logits", "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0"]: raise RuntimeError("source I/O drift")
    known = {x.name for x in [*model.graph.input, *model.graph.output, *model.graph.value_info]}; needed = {n for _, ins, outs in SPECS for n in [*ins, *outs]}
    if needed - known: raise RuntimeError(f"split tensors missing: {sorted(needed-known)}")
    rows = []
    for order, (label, inputs, outputs) in enumerate(SPECS):
        output = args.output_dir / f"{order:02d}_{label}.onnx"
        onnx.utils.extract_model(str(source), str(output), inputs, outputs, check_model=True, infer_shapes=False)
        extracted = onnx.load(str(output), load_external_data=False); staticized = []
        for value in [*extracted.graph.input, *extracted.graph.output, *extracted.graph.value_info]:
            tensor = value.type.tensor_type
            if tensor.HasField("shape") and tensor.shape.dim:
                dim = tensor.shape.dim[0]
                if dim.HasField("dim_param") and dim.dim_param == "unk__3": dim.ClearField("dim_param"); dim.dim_value = 1; staticized.append(value.name)
        onnx.save(extracted, str(output), save_as_external_data=False); extracted = onnx.load(str(output), load_external_data=False); onnx.checker.check_model(extracted)
        if [x.name for x in extracted.graph.input] != inputs or [x.name for x in extracted.graph.output] != outputs: raise RuntimeError(f"{label} I/O drift")
        rows.append({"order": order, "label": label, "path": str(output.resolve()), "size_bytes": output.stat().st_size, "sha256": sha256(output), "node_count": len(extracted.graph.node), "initializer_count": len(extracted.graph.initializer), "staticized_batch_value_count": len(staticized), "inputs": [contract(x) for x in extracted.graph.input], "outputs": [contract(x) for x in extracted.graph.output]})
    report = {"status": "created_static_pass", "variant": "int8_backbone_compat_25_static_batch1_onnx_segments", "operation": "split_into_24x1_transformer_blocks_plus_upernet_decoder_head", "source": {"path": str(source), "size_bytes": SOURCE[0], "sha256": SOURCE[1]}, "segments": rows, "execution_order": [r["label"] for r in rows], "shape_specialization": "all extracted unk__3 batch dimensions fixed to 1", "semantic_change_expected": False, "claims": {"cpu_sequential_parity": False, "strict_migraphx_admission": False, "device_resident_intersegment_io": False, "task_accuracy_90": False, "performance": False, "native_int8_kernel_verified": False}}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
