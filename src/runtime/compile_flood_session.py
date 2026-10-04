#!/usr/bin/env python3
'Research implementation: compile flood session.'

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
ENV_KEYS = (
    "MIGRAPHX_GPU_COMPILE_PARALLEL",
    "ORT_MIGRAPHX_EXHAUSTIVE_TUNE",
    "MIGRAPHX_TIME_PASSES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MALLOC_ARENA_MAX",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def quarantine_partial(partial: Path, output_dir: Path) -> dict[str, object] | None:
    if not partial.exists():
        return None
    destination = output_dir / "quarantined_partial_cache.mxr"
    partial.replace(destination)
    return identity(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--feeds", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--shim", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir.exists() or args.cache.exists():
        raise FileExistsError("output directory or final cache already exists")
    args.output_dir.mkdir(parents=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    partial = args.cache.with_suffix(args.cache.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial cache already exists: {partial}")

    shim = args.shim.resolve(strict=True)
    os.environ["PATH"] = os.pathsep.join([str(shim.parent), os.environ.get("PATH", "")])
    if Path(shutil.which("lsmod") or "").resolve(strict=False) != shim:
        raise RuntimeError("static lsmod shim is not first on PATH")
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(partial.resolve(strict=False))
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(partial.resolve(strict=False))

    started = {
        "schema": "journal_phase2b_compile_attempt_started_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": args.config_id,
        "pid": os.getpid(),
        "hardware": "海光 K100 AI 加速卡",
        "formal_90_image_test_used": False,
        "model": identity(args.model),
        "feeds": identity(args.feeds),
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "intra_op_threads": args.intra_op_threads,
    }
    write_json(args.output_dir / "attempt_started.json", started)

    wall_started = time.perf_counter()
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError(
                f"vendor runtime drift: {ort.__version__}, {ort.get_available_providers()}"
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = args.intra_op_threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output_dir / f"{args.config_id}_profile")
        with np.load(args.feeds.resolve(strict=True), allow_pickle=False) as packed:
            feeds = {
                name: np.ascontiguousarray(packed[name], dtype=np.float32)
                for name in packed.files
            }
        compile_started = time.perf_counter()
        session = ort.InferenceSession(
            str(args.model.resolve(strict=True)),
            sess_options=options,
            providers=[(MGX, {"device_id": 0})],
        )
        session.disable_fallback()
        creation_seconds = time.perf_counter() - compile_started
        if {item.name for item in session.get_inputs()} != set(feeds):
            raise RuntimeError("compiled segment feed contract drift")
        output_names = [item.name for item in session.get_outputs()]
        outputs = session.run(output_names, feeds)
        profile = Path(session.end_profiling()).resolve(strict=True)
        events = json.loads(profile.read_text(encoding="utf-8"))
        counts = Counter(
            str(event["args"]["provider"])
            for event in events
            if event.get("args", {}).get("provider")
        )
        gates = {
            "partial_cache_nonempty": partial.is_file() and partial.stat().st_size > 0,
            "migraphx_events_positive": counts[MGX] > 0,
            "cpu_events_zero": counts[CPU] == 0,
            "all_outputs_finite": bool(outputs)
            and all(np.isfinite(value).all() for value in outputs),
            "strict_provider_first": session.get_providers()[0] == MGX,
        }
        if not all(gates.values()):
            raise RuntimeError(f"strict placement or output gate failed: {gates}")
        partial.replace(args.cache)
        result = {
            "schema": "journal_phase2b_compile_one_session_v1",
            "status": "passed",
            "hardware": "海光 K100 AI 加速卡",
            "config_id": args.config_id,
            "formal_90_image_test_used": False,
            "model": identity(args.model),
            "feeds": identity(args.feeds),
            "cache": identity(args.cache),
            "profile": identity(profile),
            "provider_event_counts": dict(counts),
            "session_creation_seconds": creation_seconds,
            "wall_seconds": time.perf_counter() - wall_started,
            "environment": {key: os.environ.get(key) for key in ENV_KEYS},
            "gates": gates,
        }
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        result = {
            "schema": "journal_phase2b_compile_one_session_v1",
            "status": "failed",
            "hardware": "海光 K100 AI 加速卡",
            "config_id": args.config_id,
            "formal_90_image_test_used": False,
            "model": identity(args.model),
            "feeds": identity(args.feeds),
            "cache": None,
            "quarantined_partial_cache": quarantine_partial(partial, args.output_dir),
            "wall_seconds": time.perf_counter() - wall_started,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        }
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
