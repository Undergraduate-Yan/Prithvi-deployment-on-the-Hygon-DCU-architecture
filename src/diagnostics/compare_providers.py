#!/usr/bin/env python3
'Research implementation: compare providers.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MGX = "MIGraphXExecutionProvider"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-pack", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--save-arrays", type=Path)
    return parser.parse_args()


def session(path: Path, provider: str) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if provider == MGX else ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    if provider == MGX:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    started = time.monotonic()
    value = ort.InferenceSession(str(path), sess_options=options, providers=[(provider, {"device_id": 0})] if provider == MGX else [provider])
    if provider == MGX:
        value.disable_fallback()
        if value.get_providers()[0] != MGX:
            raise RuntimeError("MIGraphX is not primary provider")
    return value, time.monotonic() - started


def delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool | list[int]]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    denom = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return {
        "shape": list(reference.shape),
        "cpu_finite": bool(np.isfinite(reference).all()),
        "k100_finite": bool(np.isfinite(candidate).all()),
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(denom, 1e-12)),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    with np.load(args.input_pack, allow_pickle=False) as packed:
        arrays = {row["name"]: np.ascontiguousarray(packed[row["pack_key"]], dtype=np.float32) for row in manifest["inputs"]}
    sample_count = min(args.samples, min(value.shape[0] for value in arrays.values()))
    cpu, cpu_create = session(args.model, "CPUExecutionProvider")
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
    mgx, mgx_create = session(args.model, MGX)
    if [item.name for item in cpu.get_inputs()] != [item.name for item in mgx.get_inputs()]:
        raise RuntimeError("provider input contract drift")
    output_names = [item.name for item in cpu.get_outputs()]
    if output_names != [item.name for item in mgx.get_outputs()]:
        raise RuntimeError("provider output contract drift")
    aggregate = {name: [] for name in output_names}
    saved = {}
    for index in range(sample_count):
        feed = {name: value[index : index + 1] for name, value in arrays.items()}
        cpu_outputs = cpu.run(output_names, feed)
        mgx_outputs = mgx.run(output_names, feed)
        for name, reference, candidate in zip(output_names, cpu_outputs, mgx_outputs):
            aggregate[name].append(delta(reference, candidate))
            if index == 0 and args.save_arrays:
                key = hashlib.sha256(name.encode()).hexdigest()[:12]
                saved[f"cpu_{key}"] = reference
                saved[f"k100_{key}"] = candidate
    rows = []
    for name in output_names:
        metrics = aggregate[name]
        rows.append({
            "output_name": name,
            "samples": sample_count,
            "worst_mae": max(float(row["mae"]) for row in metrics),
            "worst_max_abs": max(float(row["max_abs"]) for row in metrics),
            "worst_relative_l2": max(float(row["relative_l2"]) for row in metrics),
            "non_finite_samples": sum(not row["cpu_finite"] or not row["k100_finite"] for row in metrics),
            "gate_1e4": "PASS" if max(float(row["max_abs"]) for row in metrics) <= 1e-4 else "FAIL",
            "per_sample": metrics,
        })
    result = {
        "schema": "phase7r_provider_intermediate_comparison_v1",
        "status": "PASS_EXECUTED",
        "scope": "deployment-validation only",
        "formal_payload_accessed": False,
        "model": str(args.model.resolve()),
        "providers": {"cpu": cpu.get_providers(), "k100": mgx.get_providers()},
        "cpu_session_create_seconds": cpu_create,
        "k100_session_create_seconds": mgx_create,
        "samples": sample_count,
        "outputs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.save_arrays:
        np.savez_compressed(args.save_arrays, **saved)
    print(json.dumps({"status": result["status"], "k100_session_create_seconds": mgx_create, "outputs": [{k: row[k] for k in ("output_name", "worst_mae", "worst_max_abs", "gate_1e4")} for row in rows]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
