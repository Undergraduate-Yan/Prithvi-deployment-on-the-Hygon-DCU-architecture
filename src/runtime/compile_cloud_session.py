#!/usr/bin/env python3
'Research implementation: compile cloud session.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort


MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--feeds", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--shim", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() or args.cache.exists():
        raise FileExistsError("output directory or final cache already exists")
    args.output_dir.mkdir(parents=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    partial = args.cache.with_suffix(args.cache.suffix + ".partial")
    shim = args.shim.resolve(strict=True)
    os.environ["PATH"] = os.pathsep.join((str(shim.parent), os.environ.get("PATH", "")))
    if Path(shutil.which("lsmod") or "").resolve(strict=False) != shim:
        raise RuntimeError("lsmod compatibility shim is not first on PATH")
    os.environ.update({
        "ORT_MIGRAPHX_SAVE_COMPILED_MODEL": "1", "ORT_MIGRAPHX_LOAD_COMPILED_MODEL": "0",
        "ORT_MIGRAPHX_SAVE_COMPILE_PATH": str(partial), "ORT_MIGRAPHX_LOAD_COMPILE_PATH": str(partial),
        "MIGRAPHX_GPU_COMPILE_PARALLEL": "1", "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2",
    })
    started = time.perf_counter()
    attempt = {
        "schema": "phase7r_compile_attempt_v1", "status": "STARTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_id": args.config_id,
        "hardware": "海光 K100 AI 加速卡", "formal_payload_accessed": False,
        "model": identity(args.model), "feeds": identity(args.feeds),
    }
    write_json(args.output_dir / "attempt_started.json", attempt)
    try:
        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError(f"runtime/provider drift: {ort.__version__} {ort.get_available_providers()}")
        with np.load(args.feeds.resolve(strict=True), allow_pickle=False) as packed:
            feeds = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = args.intra_op_threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output_dir / f"{args.config_id}_profile")
        compile_started = time.perf_counter()
        session = ort.InferenceSession(str(args.model.resolve(strict=True)), sess_options=options, providers=[(MGX, {"device_id": 0})])
        session.disable_fallback()
        creation_seconds = time.perf_counter() - compile_started
        expected = {item.name: item.type for item in session.get_inputs()}
        if set(expected) != set(feeds):
            raise RuntimeError(f"feed names drift: expected {sorted(expected)}, got {sorted(feeds)}")
        output_names = [item.name for item in session.get_outputs()]
        outputs = session.run(output_names, feeds)
        profile_path = Path(session.end_profiling()).resolve(strict=True)
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        counts = Counter(str(row["args"]["provider"]) for row in events if row.get("args", {}).get("provider"))
        checks = {
            "cache_nonempty": partial.is_file() and partial.stat().st_size > 0,
            "migraphx_events_positive": counts[MGX] > 0,
            "cpu_events_zero": counts[CPU] == 0,
            "strict_provider_first": session.get_providers()[0] == MGX,
            "outputs_finite": bool(outputs) and all(np.isfinite(value).all() for value in outputs),
        }
        if not all(checks.values()):
            raise RuntimeError(f"compile placement/output gate failed: {checks}")
        os.replace(partial, args.cache)
        result = {
            "schema": "phase7r_compile_session_result_v1", "status": "PASS", "config_id": args.config_id,
            "hardware": "海光 K100 AI 加速卡", "formal_payload_accessed": False,
            "model": identity(args.model), "feeds": identity(args.feeds), "cache": identity(args.cache),
            "profile": identity(profile_path), "provider_event_counts": dict(counts), "checks": checks,
            "session_creation_seconds": creation_seconds, "wall_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "feed_dtypes": {name: str(value.dtype) for name, value in feeds.items()},
        }
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        quarantined = None
        if partial.exists():
            quarantined = args.output_dir / "quarantined_partial_cache.mxr"
            os.replace(partial, quarantined)
        result = {
            "schema": "phase7r_compile_session_result_v1", "status": "FAIL", "config_id": args.config_id,
            "hardware": "海光 K100 AI 加速卡", "formal_payload_accessed": False,
            "model": identity(args.model), "feeds": identity(args.feeds),
            "quarantined_partial_cache": identity(quarantined) if quarantined else None,
            "wall_seconds": time.perf_counter() - started, "exception_type": type(error).__name__,
            "exception": str(error), "traceback": traceback.format_exc(),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
