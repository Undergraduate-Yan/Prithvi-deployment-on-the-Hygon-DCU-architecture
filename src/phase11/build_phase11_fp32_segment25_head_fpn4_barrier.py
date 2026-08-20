#!/usr/bin/env python3
"""Build the FP32 segment-24 head with only the validated fpn4 MaxPool output barrier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


SEGMENT_REPORT = (
    24_959,
    "c94692d130a0eb924290b3d56f790e166535fa3049fd9f3e72ef1fe16085c90f",
)
SOURCE_HEAD = (
    60_551_311,
    "2644c864768639151a062efcd0c69e657cdb0b7455c9b3561f08411397af2f6e",
)
PAIRED_INPUTS = (
    3_653_541,
    "98fe1142953a05944244f8934141b2500d787bfe9d549a2f3c34d7c643b49ab9",
)
BARRIER_NODE = "/task/model/decoder/fpn4/fpn4.0/MaxPool"
BARRIER_TENSOR = "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0"
BARRIER_SHAPE = (1, 1024, 7, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def array_record(value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segment-report", type=Path, required=True)
    parser.add_argument("--paired-inputs", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate.exists() or args.report.exists():
        raise RuntimeError("refusing to overwrite candidate or build report")

    identities = {
        "segment_report": lock(args.segment_report, SEGMENT_REPORT),
        "source_head": lock(args.source, SOURCE_HEAD),
        "paired_head_inputs": lock(args.paired_inputs, PAIRED_INPUTS),
    }
    segment_report = json.loads(args.segment_report.read_text(encoding="utf-8"))
    rows = segment_report.get("segments", [])
    if segment_report.get("variant") != "fp32_layernorm_decomposed_deconv_rewritten_25_static_batch1_onnx_segments":
        raise RuntimeError("segment report variant drift")
    if len(rows) != 25 or rows[-1].get("label") != "upernet_decoder_head":
        raise RuntimeError("segment report topology drift")
    if (int(rows[-1]["size_bytes"]), str(rows[-1]["sha256"])) != SOURCE_HEAD:
        raise RuntimeError("source head identity disagrees with segment report")

    model = onnx.load(str(args.source), load_external_data=False)
    if [item.name for item in model.graph.output] != ["logits"]:
        raise RuntimeError("source head output contract drift")
    hits = [node for node in model.graph.node if node.name == BARRIER_NODE and BARRIER_TENSOR in node.output]
    if len(hits) != 1 or any(item.name == BARRIER_TENSOR for item in model.graph.output):
        raise RuntimeError("barrier identity drift")
    model.graph.output.append(
        helper.make_tensor_value_info(BARRIER_TENSOR, TensorProto.FLOAT, list(BARRIER_SHAPE))
    )
    onnx.checker.check_model(model)
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.candidate))
    reloaded = onnx.load(str(args.candidate), load_external_data=False)
    onnx.checker.check_model(reloaded)
    if [item.name for item in reloaded.graph.output] != ["logits", BARRIER_TENSOR]:
        raise RuntimeError("candidate output contract drift")

    import onnxruntime as ort

    if ort.__version__ != "1.19.2":
        raise RuntimeError(f"ONNX Runtime identity drift: {ort.__version__}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    source_session = ort.InferenceSession(
        str(args.source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        str(args.candidate), sess_options=options, providers=["CPUExecutionProvider"]
    )
    packed = np.load(args.paired_inputs, allow_pickle=False)
    feeds = {
        item.name: np.ascontiguousarray(packed[f"input_{index}"], dtype=np.float32)
        for index, item in enumerate(source_session.get_inputs())
    }
    if set(feeds) != {item.name for item in candidate_session.get_inputs()}:
        raise RuntimeError("candidate input contract drift")
    source_logits = np.ascontiguousarray(source_session.run(["logits"], feeds)[0], dtype=np.float32)
    candidate_outputs = candidate_session.run(["logits", BARRIER_TENSOR], feeds)
    candidate_logits = np.ascontiguousarray(candidate_outputs[0], dtype=np.float32)
    barrier = np.ascontiguousarray(candidate_outputs[1], dtype=np.float32)
    if source_logits.shape != (1, 2, 224, 224) or barrier.shape != BARRIER_SHAPE:
        raise RuntimeError("CPU output shape drift")
    reference_logits = np.ascontiguousarray(packed["cpu_output"], dtype=np.float32)
    gates = {
        "onnx_checker_passed": True,
        "only_one_graph_output_added": len(reloaded.graph.output) == 2,
        "source_cpu_reproduces_frozen_cpu_output_exactly": bool(np.array_equal(source_logits, reference_logits)),
        "candidate_cpu_logits_exactly_equal_source": bool(np.array_equal(candidate_logits, source_logits)),
        "candidate_cpu_logits_finite": bool(np.isfinite(candidate_logits).all()),
        "candidate_cpu_barrier_finite": bool(np.isfinite(barrier).all()),
        "candidate_cpu_barrier_shape_exact": tuple(barrier.shape) == BARRIER_SHAPE,
    }
    report = {
        "schema": "phase11_fp32_segment25_head_fpn4barrier_build_v1",
        "status": "passed" if all(gates.values()) else "failed",
        "operation": "append_single_fpn4_maxpool_graph_output_barrier",
        "identities": identities,
        "candidate": lock(args.candidate),
        "barrier": {
            "node": BARRIER_NODE,
            "tensor": BARRIER_TENSOR,
            "shape": list(BARRIER_SHAPE),
            "dtype": "float32",
        },
        "source_cpu_logits": array_record(source_logits),
        "candidate_cpu_logits": array_record(candidate_logits),
        "candidate_cpu_barrier": array_record(barrier),
        "gates": gates,
        "evidence_boundary": {
            "source_head_is_convtranspose_rewritten_compatibility_artifact": True,
            "no_computational_node_or_initializer_changed": True,
            "only_graph_output_visibility_changed": True,
            "performance_not_measured": True,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
