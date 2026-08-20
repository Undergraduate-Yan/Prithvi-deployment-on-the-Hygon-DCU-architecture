#!/usr/bin/env python3
"""Verify MIGraphX 5.1 compiled-model save/load on a frozen FP32 toy graph.

The orchestrator launches two fresh Python workers.  The first worker compiles
and saves a .mxr program; the second worker must load that exact cache.  Both
workers compare against CPU EP and produce a provider profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_MODEL_SIZE = 4_194_429
EXPECTED_MODEL_SHA256 = "445f55bb17d8c494de70e6999b21262e6c42db5c781427c3e7a20b9a56abb111"
MIGRAPHX_EP = "MIGraphXExecutionProvider"
CPU_EP = "CPUExecutionProvider"
SHAPE = (196, 1024)
SEED = 20260813


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def cache_record(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "size": stat.st_size,
        "sha256": sha256_file(path),
        "atime_ns": stat.st_atime_ns,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def profile_summary(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    nodes = [
        event
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    ]
    return {
        "node_events": len(nodes),
        "provider_event_counts": dict(
            Counter(event["args"]["provider"] for event in nodes)
        ),
        "operation_event_counts": dict(
            Counter(event["args"].get("op_name", "") for event in nodes)
        ),
    }


def cache_environment(mode: str, cache: Path, flavor: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ORT_MIGRAPHX_") and ("SAVE" in key or "LOAD" in key):
            env.pop(key)

    if flavor == "v119-source":
        save_path_key = "ORT_MIGRAPHX_SAVE_COMPILE_PATH"
        load_path_key = "ORT_MIGRAPHX_LOAD_COMPILE_PATH"
    elif flavor == "documentation":
        save_path_key = "ORT_MIGRAPHX_SAVE_COMPILED_PATH"
        load_path_key = "ORT_MIGRAPHX_LOAD_COMPILED_PATH"
    else:
        raise ValueError(flavor)

    env["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1" if mode == "save" else "0"
    env["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1" if mode == "load" else "0"
    env[save_path_key] = str(cache)
    env[load_path_key] = str(cache)
    return env


def worker(model: Path, cache: Path, output_dir: Path, mode: str, flavor: str) -> None:
    import onnxruntime as ort

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    if model.stat().st_size != EXPECTED_MODEL_SIZE or sha256_file(model) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("frozen MatMul model identity mismatch")
    if mode == "load" and not cache.is_file():
        raise FileNotFoundError(cache)

    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 0.25, SHAPE).astype(np.float32)

    cpu_start = time.perf_counter_ns()
    cpu_session = ort.InferenceSession(str(model), providers=[CPU_EP])
    cpu_create_s = (time.perf_counter_ns() - cpu_start) / 1e9
    cpu_output = np.asarray(cpu_session.run(["y"], {"x": x})[0])

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(output_dir / f"{mode}_profile")
    so.log_severity_level = 0
    so.log_verbosity_level = 1

    cache_before = cache_record(cache)
    create_start = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(model),
        sess_options=so,
        providers=[(MIGRAPHX_EP, {"device_id": 0})],
    )
    create_s = (time.perf_counter_ns() - create_start) / 1e9
    if hasattr(session, "disable_fallback"):
        session.disable_fallback()
    run_start = time.perf_counter_ns()
    output = np.asarray(session.run(["y"], {"x": x})[0])
    run_ms = (time.perf_counter_ns() - run_start) / 1e6
    profile_path = Path(session.end_profiling())
    profile = profile_summary(profile_path)
    cache_after = cache_record(cache)

    diff = np.abs(output.astype(np.float64) - cpu_output.astype(np.float64))
    counts = profile["provider_event_counts"]
    gates = {
        "output_finite": bool(np.isfinite(output).all()),
        "output_shape": list(output.shape) == [196, 1024],
        "output_dtype_fp32": output.dtype == np.float32,
        "migraphx_nodes_present": counts.get(MIGRAPHX_EP, 0) > 0,
        "cpu_nodes_zero": counts.get(CPU_EP, 0) == 0,
        "cpu_parity_mae": float(diff.mean()) <= 1e-5,
        "cpu_parity_max": float(diff.max()) <= 1e-4,
        "cache_exists_after": cache_after is not None and cache_after["size"] > 0,
        "load_cache_content_unchanged": mode != "load" or (
            cache_before is not None
            and cache_after is not None
            and cache_before["size"] == cache_after["size"]
            and cache_before["sha256"] == cache_after["sha256"]
        ),
    }
    result = {
        "status": "passed" if all(gates.values()) else "failed",
        "mode": mode,
        "env_flavor": flavor,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "software": {
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "registered_providers": session.get_providers(),
            "registered_provider_options": session.get_provider_options(),
        },
        "model": {
            "path": str(model),
            "size": model.stat().st_size,
            "sha256": sha256_file(model),
        },
        "input": {"shape": list(x.shape), "dtype": str(x.dtype), "sha256": sha256_array(x)},
        "cache_before": cache_before,
        "cache_after": cache_after,
        "timing_diagnostic": {
            "cpu_session_create_seconds": cpu_create_s,
            "migraphx_session_create_seconds": create_s,
            "single_host_to_host_run_ms": run_ms,
            "performance_claim_allowed": False,
        },
        "numeric": {
            "mae_vs_cpu": float(diff.mean()),
            "max_abs_vs_cpu": float(diff.max()),
            "output_sha256": sha256_array(output),
            "cpu_output_sha256": sha256_array(cpu_output),
        },
        "profile": {
            "path": str(profile_path),
            "size": profile_path.stat().st_size,
            "sha256": sha256_file(profile_path),
            **profile,
        },
        "gates": gates,
    }
    np.save(output_dir / "output.npy", output, allow_pickle=False)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path), "gates": gates}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


def run_worker(
    script: Path,
    model: Path,
    cache: Path,
    root: Path,
    mode: str,
    flavor: str,
) -> dict[str, object]:
    output_dir = root / mode
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--mode",
        mode,
        "--model",
        str(model),
        "--cache",
        str(cache),
        "--output-dir",
        str(output_dir),
        "--env-flavor",
        flavor,
    ]
    trace_path = root / f"{mode}.strace"
    traced_command = (
        ["strace", "-f", "-e", "trace=openat,read", "-o", str(trace_path), *command]
        if mode == "load"
        else command
    )
    completed = subprocess.run(
        traced_command,
        env=cache_environment(mode, cache, flavor),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (root / f"{mode}.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} worker failed with exit {completed.returncode}")
    return json.loads((output_dir / "result.json").read_text(encoding="utf-8"))


def orchestrate(model: Path, cache: Path, output_dir: Path, flavor: str) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raise FileExistsError(cache)

    script = Path(__file__).resolve()
    summary: dict[str, object] = {
        "status": "failed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": {"path": str(script), "size": script.stat().st_size, "sha256": sha256_file(script)},
        "env_flavor": flavor,
        "claims": {"compiled_cache_supported": False, "performance_claim_allowed": False},
    }
    summary_path = output_dir / "summary.json"
    try:
        save_result = run_worker(script, model, cache, output_dir, "save", flavor)
        saved_record = cache_record(cache)
        if saved_record is None:
            raise RuntimeError("cache was not created")
        saved_identity = {"size": saved_record["size"], "sha256": saved_record["sha256"]}
        load_result = run_worker(script, model, cache, output_dir, "load", flavor)
        loaded_record = cache_record(cache)
        if loaded_record is None:
            raise RuntimeError("cache disappeared after load")
        loaded_identity = {"size": loaded_record["size"], "sha256": loaded_record["sha256"]}
        save_output = np.load(output_dir / "save" / "output.npy", allow_pickle=False)
        load_output = np.load(output_dir / "load" / "output.npy", allow_pickle=False)
        cross_diff = np.abs(save_output.astype(np.float64) - load_output.astype(np.float64))
        save_log = (output_dir / "save.log").read_text(encoding="utf-8", errors="replace")
        load_log = (output_dir / "load.log").read_text(encoding="utf-8", errors="replace")
        load_trace_path = output_dir / "load.strace"
        load_trace = load_trace_path.read_text(encoding="utf-8", errors="replace")
        cache_open_trace_lines = [
            line for line in load_trace.splitlines() if str(cache) in line and "openat(" in line
        ]
        gates = {
            "save_worker_passed": save_result["status"] == "passed",
            "load_worker_passed": load_result["status"] == "passed",
            "cache_identity_unchanged": saved_identity == loaded_identity,
            "load_trace_confirms_cache_open": bool(cache_open_trace_lines),
            "save_load_mae": float(cross_diff.mean()) <= 1e-7,
            "save_load_max": float(cross_diff.max()) <= 1e-6,
        }
        summary.update(
            {
                "status": "passed" if all(gates.values()) else "failed",
                "model": save_result["model"],
                "cache": {"path": str(cache), **loaded_identity},
                "save": save_result,
                "load": load_result,
                "save_load_numeric": {
                    "mae": float(cross_diff.mean()),
                    "max_abs": float(cross_diff.max()),
                    "exact_array_equal": bool(np.array_equal(save_output, load_output)),
                },
                "session_create_speedup_diagnostic": (
                    save_result["timing_diagnostic"]["migraphx_session_create_seconds"]
                    / load_result["timing_diagnostic"]["migraphx_session_create_seconds"]
                ),
                "non_gating_log_markers": {
                    "save_complete_message_present": "Model Save: Complete" in save_log,
                    "load_success_message_present": "load model : Success" in load_log,
                },
                "load_cache_trace": {
                    "path": str(load_trace_path),
                    "size": load_trace_path.stat().st_size,
                    "sha256": sha256_file(load_trace_path),
                    "matching_openat_lines": cache_open_trace_lines,
                },
                "gates": gates,
            }
        )
        summary["claims"]["compiled_cache_supported"] = summary["status"] == "passed"
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path), "gates": summary.get("gates")}, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--env-flavor", choices=("v119-source", "documentation"), default="v119-source")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=("save", "load"), help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.mode is None:
            raise ValueError("--mode is required with --worker")
        worker(args.model, args.cache, args.output_dir, args.mode, args.env_flavor)
    else:
        orchestrate(args.model, args.cache, args.output_dir, args.env_flavor)


if __name__ == "__main__":
    main()
