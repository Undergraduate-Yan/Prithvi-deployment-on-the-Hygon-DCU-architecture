#!/usr/bin/env python3
'Research implementation: prepare cloud system.'
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx


HARDWARE = "海光 K100 AI 加速卡"
IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def tensor_meta(value: onnx.ValueInfoProto) -> tuple[tuple[int, ...], str]:
    tensor = value.type.tensor_type
    dtype = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
    }.get(tensor.elem_type)
    if dtype is None:
        raise RuntimeError(f"unsupported pipeline boundary dtype: {value.name}={tensor.elem_type}")
    dimensions = []
    for index, item in enumerate(tensor.shape.dim):
        if item.HasField("dim_value") and item.dim_value > 0:
            dimensions.append(int(item.dim_value))
        elif index == 0:
            dimensions.append(1)
        else:
            raise RuntimeError(f"unresolved static dimension: {value.name}")
    return tuple(dimensions), dtype


def clean(value: str) -> str:
    if not value or "\t" in value or "\n" in value or "\r" in value:
        raise RuntimeError(f"invalid plan field: {value!r}")
    return value


def make_plan(role: str, models: list[Path], caches: list[Path], output: Path) -> dict:
    if len(models) != 14 or len(caches) != 14:
        raise RuntimeError(f"{role}: exactly 14 models/caches required")
    produced = {"input": ((1, 6, 224, 224), "float32")}
    lines = [
        "K100_PIPELINE_PLAN_V2", f"variant\t{clean(role)}", f"hardware\t{HARDWARE}",
        "runtime\t1.19.2", f"image\t{IMAGE}", "session_count\t14",
        "input\tinput\t1,6,224,224", "final_logits\tlogits\t1,4,224,224",
    ]
    assets = []
    for ordinal, (model_path, cache_path) in enumerate(zip(models, caches)):
        model_path, cache_path = model_path.resolve(strict=True), cache_path.resolve(strict=True)
        graph = onnx.load(str(model_path), load_external_data=False).graph
        initializer_names = {item.name for item in graph.initializer}
        inputs = [item for item in graph.input if item.name not in initializer_names]
        outputs = list(graph.output)
        input_names = [clean(item.name) for item in inputs]
        if not input_names or any(name not in produced for name in input_names):
            missing = [name for name in input_names if name not in produced]
            raise RuntimeError(f"{role} session {ordinal}: unresolved inputs {missing}")
        for item in inputs:
            if tensor_meta(item) != produced[item.name]:
                raise RuntimeError(f"{role} session {ordinal}: boundary shape/dtype mismatch for {item.name}")
        fields = [
            "S", str(ordinal), f"session_{ordinal:02d}", str(model_path), str(model_path.stat().st_size),
            str(cache_path), str(cache_path.stat().st_size), str(len(input_names)), *input_names, str(len(outputs)),
        ]
        output_rows = []
        for item in outputs:
            name = clean(item.name)
            dimensions, dtype = tensor_meta(item)
            if name in produced:
                raise RuntimeError(f"{role} session {ordinal}: duplicate output {name}")
            produced[name] = (dimensions, dtype)
            fields.extend((name, ",".join(map(str, dimensions)), dtype))
            output_rows.append({"name": name, "shape": list(dimensions), "dtype": dtype})
        lines.append("\t".join(fields))
        assets.append({
            "ordinal": ordinal, "model": identity(model_path), "cache": identity(cache_path),
            "inputs": [
                {"name": item.name, "shape": list(tensor_meta(item)[0]), "dtype": tensor_meta(item)[1]}
                for item in inputs
            ],
            "outputs": output_rows,
        })
    if produced.get("logits") != ((1, 4, 224, 224), "float32"):
        raise RuntimeError(f"{role}: final logits contract drift")
    lines.append("END")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"role": role, "session_count": 14, "plan": identity(output), "assets": assets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dv-inputs", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    plans = args.output_root / "plans"
    plans.mkdir()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = config.get("candidates", [])
    if not candidates:
        raise RuntimeError("system candidate config is empty")
    values = np.load(args.dv_inputs, mmap_mode="r")
    sample = np.ascontiguousarray(values[0:1], dtype=np.float32)
    if sample.shape != (1, 6, 224, 224) or not np.isfinite(sample).all():
        raise RuntimeError("benchmark input contract drift")
    input_path = args.output_root / "deployment_validation_input_000.float32.raw"
    sample.tofile(input_path)
    rows = []
    for candidate in candidates:
        role = candidate["role"]
        rows.append(make_plan(role, [Path(x) for x in candidate["models"]], [Path(x) for x in candidate["caches"]], plans / f"{role}.tsv"))
    manifest = {
        "schema": "cloud_phase7r_system_entry_v1", "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": HARDWARE, "deployment_image": IMAGE, "runner": identity(args.runner),
        "runner_name": "Runner-A-Cloud-Contract-v2",
        "topology": "14 identical session boundaries for every candidate",
        "input_scope": "first tile of the supplied development input pack; verify against its retained input identity", "benchmark_input": identity(input_path),
        "protocol": {"fresh_processes": 5, "warmups": 50, "measured_calls": 200, "scope": "pipeline plus FP32 logits D2H"},
        "candidates": rows,
    }
    output = args.output_root / "system_entry_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate_count": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
