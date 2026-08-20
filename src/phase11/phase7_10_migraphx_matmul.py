#!/usr/bin/env python3
"""Strict MIGraphX Phase 7-10 MatMul admission, benchmark and profile workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

M, K, N = 196, 1024, 1024
PROVIDER = "MIGraphXExecutionProvider"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_models(root: Path) -> dict[str, tuple[Path, np.ndarray]]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260813)
    x32 = rng.normal(0, 0.25, (M, K)).astype(np.float32)
    w32 = rng.normal(0, 0.05, (K, N)).astype(np.float32)
    out: dict[str, tuple[Path, np.ndarray]] = {}

    for name, dtype, x, w in (
        ("fp32", TensorProto.FLOAT, x32, w32),
        ("fp16", TensorProto.FLOAT16, x32.astype(np.float16), w32.astype(np.float16)),
    ):
        path = root / f"matmul_{name}.onnx"
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name=f"MatMul_{name}")],
            f"matmul_{name}",
            [helper.make_tensor_value_info("x", dtype, [M, K])],
            [helper.make_tensor_value_info("y", dtype, [M, N])],
            [numpy_helper.from_array(w, "w")],
        )
        save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        out[name] = (path, x)

    # Static QDQ S8S8 MatMul. Inputs and outputs remain float32; the Q/DQ pair
    # presents a fixed, auditable INT8 candidate to the compiler.
    xs = np.array(0.01, np.float32)
    ws = np.array(max(float(np.max(np.abs(w32))) / 127.0, 1e-8), np.float32)
    zero = np.array(0, np.int8)
    wq = np.clip(np.rint(w32 / ws), -127, 127).astype(np.int8)
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"], name="X_Q"),
        helper.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xdq"], name="X_DQ"),
        helper.make_node("DequantizeLinear", ["wq", "ws", "wz"], ["wdq"], name="W_DQ"),
        helper.make_node("MatMul", ["xdq", "wdq"], ["y"], name="QDQ_MatMul"),
    ]
    initializers = [
        numpy_helper.from_array(xs, "xs"), numpy_helper.from_array(zero, "xz"),
        numpy_helper.from_array(ws, "ws"), numpy_helper.from_array(zero, "wz"),
        numpy_helper.from_array(wq, "wq"),
    ]
    path = root / "matmul_int8_qdq.onnx"
    graph = helper.make_graph(
        nodes, "matmul_int8_qdq",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])], initializers,
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    out["int8_qdq"] = (path, x32)
    return out


def session(model: Path, profile_prefix: Path | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = str(profile_prefix)
    s = ort.InferenceSession(
        str(model), sess_options=so,
        providers=[(PROVIDER, {"device_id": 0})],
    )
    if hasattr(s, "disable_fallback"):
        s.disable_fallback()
    if s.get_providers()[0] != PROVIDER:
        raise RuntimeError(f"unexpected providers: {s.get_providers()}")
    return s


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q, method="linear"))


def profile_counts(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    nodes = [e for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
    return {
        "provider_event_counts": dict(Counter(e["args"]["provider"] for e in nodes)),
        "operations": dict(Counter(e["args"].get("op_name", "") for e in nodes)),
        "node_events": len(nodes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=("admission", "benchmark", "profile"), required=True)
    ap.add_argument("--order", default="fp32,fp16,int8_qdq")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    models = make_models(args.output_dir / "models")
    order = args.order.split(",")
    if sorted(order) != sorted(models):
        raise ValueError(order)
    result: dict[str, object] = {
        "mode": args.mode, "shape": [M, K, N], "order": order,
        "software": {"python": platform.python_version(), "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "providers": ort.get_available_providers()},
        "models": {k: {"path": str(v[0]), "sha256": sha256(v[0])} for k, v in models.items()},
        "tests": {},
    }
    for name in order:
        model, x = models[name]
        prefix = args.output_dir / f"{name}_ort_profile" if args.mode in ("admission", "profile") else None
        s = session(model, prefix)
        if args.mode == "admission":
            y = np.asarray(s.run(None, {"x": x})[0])
            profile = Path(s.end_profiling())
            pc = profile_counts(profile)
            gates = {
                "finite": bool(np.isfinite(y).all()),
                "migraphx_nodes_present": pc["provider_event_counts"].get(PROVIDER, 0) > 0,
                "cpu_nodes_zero": pc["provider_event_counts"].get("CPUExecutionProvider", 0) == 0,
            }
            result["tests"][name] = {"status": "passed" if all(gates.values()) else "failed",
                "session_providers": s.get_providers(), "output_dtype": str(y.dtype),
                "output_shape": list(y.shape), "output_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "profile": str(profile), "profile_sha256": sha256(profile), "profile_counts": pc, "gates": gates}
        elif args.mode == "benchmark":
            for _ in range(30):
                s.run(None, {"x": x})
            times: list[float] = []
            for _ in range(100):
                t0 = time.perf_counter_ns(); s.run(None, {"x": x}); t1 = time.perf_counter_ns()
                times.append((t1 - t0) / 1e6)
            result["tests"][name] = {"latency_ms": times, "median_ms": float(np.median(times)),
                "p95_ms": percentile(times, 95), "mean_ms": float(np.mean(times)),
                "throughput_matmuls_per_s": 1000.0 / float(np.mean(times))}
        else:
            for _ in range(5):
                s.run(None, {"x": x})
            profile = Path(s.end_profiling())
            result["tests"][name] = {"profile": str(profile), "profile_sha256": sha256(profile),
                "profile_counts": profile_counts(profile)}
    if args.mode == "benchmark":
        base = result["tests"]["fp32"]["median_ms"]
        for row in result["tests"].values():
            row["speedup_vs_fp32_median"] = base / row["median_ms"]
    result["status"] = "passed" if all(x.get("status", "passed") == "passed" for x in result["tests"].values()) else "failed"
    result["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    p = args.output_dir / "result.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "mode": args.mode, "result": str(p),
                      "summary": {k: {x: v for x, v in row.items() if x not in ("latency_ms",)} for k, row in result["tests"].items()}},
                     ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
