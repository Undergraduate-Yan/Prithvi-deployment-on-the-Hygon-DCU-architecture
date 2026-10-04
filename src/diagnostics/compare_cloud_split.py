#!/usr/bin/env python3
'Research implementation: compare cloud split.'
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MGX = "MIGraphXExecutionProvider"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--prefix-model", required=True, type=Path)
    parser.add_argument("--suffix-model", required=True, type=Path)
    parser.add_argument("--input-pack", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args()


def create(path: Path, provider: str) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4 if provider == "CPUExecutionProvider" else 1
    options.inter_op_num_threads = 1
    if provider == MGX:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers: list[object] = [(MGX, {"device_id": 0})]
    else:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = [provider]
    started = time.monotonic()
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if provider == MGX:
        session.disable_fallback()
        if session.get_providers()[0] != MGX:
            raise RuntimeError("MIGraphX is not primary provider")
    return session, time.monotonic() - started


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "shape": list(reference.shape),
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-12)),
        "finite": bool(np.isfinite(candidate).all()),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    worst_max = max(float(row["max_abs"]) for row in rows)
    return {
        "samples": len(rows),
        "worst_mae": max(float(row["mae"]) for row in rows),
        "worst_max_abs": worst_max,
        "worst_relative_l2": max(float(row["relative_l2"]) for row in rows),
        "non_finite_samples": sum(not bool(row["finite"]) for row in rows),
        "gate_1e4": "PASS" if worst_max <= 1e-4 else "FAIL",
        "per_sample": rows,
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    with np.load(args.input_pack, allow_pickle=False) as packed:
        arrays = {row["name"]: np.ascontiguousarray(packed[row["pack_key"]], dtype=np.float32) for row in manifest["inputs"]}
    count = min(args.samples, min(value.shape[0] for value in arrays.values()))
    reference, reference_create = create(args.reference_model, "CPUExecutionProvider")
    prefix_cpu, prefix_cpu_create = create(args.prefix_model, "CPUExecutionProvider")
    suffix_cpu, suffix_cpu_create = create(args.suffix_model, "CPUExecutionProvider")
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
    prefix_k100, prefix_k100_create = create(args.prefix_model, MGX)
    suffix_k100, suffix_k100_create = create(args.suffix_model, MGX)
    output_names = [item.name for item in reference.get_outputs()]
    cpu_rows = {name: [] for name in output_names}
    k100_rows = {name: [] for name in output_names}
    prediction_agreement = []
    for index in range(count):
        feed = {name: value[index : index + 1] for name, value in arrays.items()}
        expected = reference.run(output_names, feed)
        cpu_boundary = prefix_cpu.run(None, feed)
        cpu_feed = {item.name: value for item, value in zip(suffix_cpu.get_inputs(), cpu_boundary)}
        cpu_actual = suffix_cpu.run(output_names, cpu_feed)
        k100_boundary = prefix_k100.run(None, feed)
        k100_feed = {item.name: value for item, value in zip(suffix_k100.get_inputs(), k100_boundary)}
        k100_actual = suffix_k100.run(output_names, k100_feed)
        for name, ref, cpu_value, k100_value in zip(output_names, expected, cpu_actual, k100_actual):
            cpu_rows[name].append(metrics(ref, cpu_value))
            k100_rows[name].append(metrics(ref, k100_value))
        prediction_agreement.append(float(np.mean(np.argmax(expected[0], axis=1) == np.argmax(k100_actual[0], axis=1))))
    cpu_summary = {name: aggregate(rows) for name, rows in cpu_rows.items()}
    k100_summary = {name: aggregate(rows) for name, rows in k100_rows.items()}
    result = {
        "schema": "phase7r_session12_split_numeric_gate_v1",
        "status": "PASS" if all(row["gate_1e4"] == "PASS" for row in k100_summary.values()) else "FAIL",
        "scope": "deployment-validation only",
        "formal_payload_accessed": False,
        "samples": count,
        "providers": {"reference": reference.get_providers(), "candidate": [prefix_k100.get_providers(), suffix_k100.get_providers()]},
        "session_create_seconds": {
            "reference_cpu": reference_create,
            "prefix_cpu": prefix_cpu_create,
            "suffix_cpu": suffix_cpu_create,
            "prefix_k100": prefix_k100_create,
            "suffix_k100": suffix_k100_create,
        },
        "cpu_split_semantic_equivalence": cpu_summary,
        "k100_split_vs_cpu_reference": k100_summary,
        "prediction_agreement_mean": float(np.mean(prediction_agreement)),
        "prediction_agreement_min": float(np.min(prediction_agreement)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "session_create_seconds": result["session_create_seconds"],
        "k100_split_vs_cpu_reference": {name: {key: row[key] for key in ("worst_mae", "worst_max_abs", "gate_1e4")} for name, row in k100_summary.items()},
        "prediction_agreement_mean": result["prediction_agreement_mean"],
        "prediction_agreement_min": result["prediction_agreement_min"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
