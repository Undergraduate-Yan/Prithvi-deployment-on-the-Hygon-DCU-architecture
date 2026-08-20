#!/usr/bin/env python3
"""Attribute FP16 compatibility-graph drift to each static transform.

This is a diagnostic only.  It never changes the frozen source or the formal
combined candidate and does not turn a failed gate into a pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from build_phase11_fp16_single_barrier_candidate import (
    SOURCE_SHA256,
    SOURCE_SIZE,
    rewrite_convtranspose,
    rewrite_layernorm,
    sha256,
)
from phase11_checkpoint_diagnostic import load_raw_image, statistics


def cpu_session(model_or_path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_or_path, sess_options=options, providers=["CPUExecutionProvider"]
    )


def evaluate(model: onnx.ModelProto, image: np.ndarray) -> tuple[np.ndarray, float, int]:
    onnx.checker.check_model(model)
    payload = model.SerializeToString()
    started = time.perf_counter()
    sess = cpu_session(payload)
    result = np.asarray(sess.run(None, {sess.get_inputs()[0].name: image})[0])
    return result, time.perf_counter() - started, len(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    source = args.source.resolve(strict=True)
    if source.stat().st_size != SOURCE_SIZE or sha256(source) != SOURCE_SHA256:
        raise RuntimeError("frozen FP16 source identity mismatch")

    image = load_raw_image(args.sample)
    base = onnx.load(str(source), load_external_data=False)
    ref_session = cpu_session(str(source))
    started = time.perf_counter()
    reference = np.asarray(
        ref_session.run(None, {ref_session.get_inputs()[0].name: image})[0]
    )
    reference_seconds = time.perf_counter() - started

    rows = []
    for name, use_layernorm, use_convtranspose in (
        ("layernorm_only", True, False),
        ("convtranspose_only", False, True),
        ("layernorm_and_convtranspose", True, True),
    ):
        model = copy.deepcopy(base)
        ln_changes = rewrite_layernorm(model) if use_layernorm else []
        ct_changes = rewrite_convtranspose(model) if use_convtranspose else []
        output, seconds, serialized_size = evaluate(model, image)
        rows.append(
            {
                "variant": name,
                "layernorm_rewrites": len(ln_changes),
                "convtranspose_rewrites": len(ct_changes),
                "serialized_size_bytes": serialized_size,
                "session_and_inference_seconds": seconds,
                "comparison_vs_frozen_source": statistics(reference, output),
            }
        )

    result = {
        "status": "diagnostic_completed",
        "source": {
            "size_bytes": source.stat().st_size,
            "sha256": SOURCE_SHA256,
        },
        "reference_inference_seconds": reference_seconds,
        "variants": rows,
        "claims": {
            "formal_cpu_numeric_gate_passed": False,
            "migraphx_placement": False,
            "task_accuracy_90": False,
            "performance": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
