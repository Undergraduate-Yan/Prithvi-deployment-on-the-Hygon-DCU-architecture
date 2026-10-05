#!/usr/bin/env python3
'Flood system measurements on a fixed development input.'
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))


import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from flood_runner import K100BundleRunner, identity


EXPECTED_INPUT = {"size_bytes": 1204352, "sha256": "7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80"}


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def write_csv(path: Path, values: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["inference_index", "elapsed_ms"])
        writer.writeheader(); writer.writerows({"inference_index": i, "elapsed_ms": value} for i, value in enumerate(values))


def write_stage(output: Path, stage: str) -> None:
    (output / "stage.txt").write_text(stage + "\n", encoding="utf-8")
    with (output / "stage_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, "unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}) + "\n")


def summary(values: list[float], wall_seconds: float) -> dict:
    mean = statistics.fmean(values)
    return {
        "count": len(values), "mean_ms": mean, "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99),
        "std_ms": statistics.pstdev(values), "cv": statistics.pstdev(values) / mean,
        "minimum_ms": min(values), "maximum_ms": max(values), "wall_seconds": wall_seconds,
        "throughput_per_second": len(values) / wall_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "cold-trial", "vram", "stability", "placement"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    if args.warmup < 0 or args.measured < 1 or args.duration_seconds <= 0: raise RuntimeError("invalid protocol counts")
    input_id = identity(args.input)
    if {key: input_id[key] for key in EXPECTED_INPUT} != EXPECTED_INPUT: raise RuntimeError("configuration-validation input identity drift")
    raw = np.load(args.input, allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != (1, 6, 224, 224) or not np.isfinite(raw).all(): raise RuntimeError("input contract drift")
    raw = np.ascontiguousarray(raw)
    args.output_dir.mkdir(parents=True)
    write_stage(args.output_dir, "process_started")
    total_started = time.perf_counter()
    profiles_dir = args.output_dir / "profiles" if args.mode == "placement" else None
    runner = K100BundleRunner(args.config, device=0, profile_dir=profiles_dir)
    write_stage(args.output_dir, "sessions_loaded")
    logits, prediction, first_ms = runner.run(raw)
    write_stage(args.output_dir, "first_inference")
    baseline_logits_hash = array_sha256(logits); baseline_prediction_hash = array_sha256(prediction)
    if not np.isfinite(logits).all(): raise RuntimeError("first result nonfinite")
    timings: list[float] = []
    drift = 0; nonfinite = 0; errors = 0
    measured_started_ns = None; measured_ended_ns = None

    if args.mode in {"benchmark", "placement"}:
        for _ in range(args.warmup): runner.run(raw)
        measured_started_ns = time.time_ns(); wall_started = time.perf_counter()
        for _ in range(args.measured):
            value_logits, value_prediction, elapsed = runner.run(raw); timings.append(elapsed)
            nonfinite += int(not np.isfinite(value_logits).all())
            drift += int(array_sha256(value_prediction) != baseline_prediction_hash)
        wall_seconds = time.perf_counter() - wall_started; measured_ended_ns = time.time_ns()
    elif args.mode == "cold-trial":
        timings = [first_ms]; wall_seconds = first_ms / 1000.0
        measured_started_ns = measured_ended_ns = time.time_ns()
    elif args.mode == "vram":
        write_stage(args.output_dir, "steady_state")
        wall_started = time.perf_counter(); measured_started_ns = time.time_ns()
        for _ in range(args.measured):
            value_logits, value_prediction, elapsed = runner.run(raw); timings.append(elapsed)
            nonfinite += int(not np.isfinite(value_logits).all()); drift += int(array_sha256(value_prediction) != baseline_prediction_hash)
        wall_seconds = time.perf_counter() - wall_started; measured_ended_ns = time.time_ns()
    else:
        write_stage(args.output_dir, "steady_state")
        wall_started = time.perf_counter(); measured_started_ns = time.time_ns(); last_progress = wall_started
        while time.perf_counter() - wall_started < args.duration_seconds:
            try:
                value_logits, value_prediction, elapsed = runner.run(raw); timings.append(elapsed)
                nonfinite += int(not np.isfinite(value_logits).all()); drift += int(array_sha256(value_prediction) != baseline_prediction_hash)
            except Exception:
                errors += 1; raise
            if time.perf_counter() - last_progress >= 60:
                print(f"stability variant={args.label} elapsed={time.perf_counter()-wall_started:.1f}s calls={len(timings)}", flush=True); last_progress = time.perf_counter()
        wall_seconds = time.perf_counter() - wall_started; measured_ended_ns = time.time_ns()

    write_stage(args.output_dir, "complete")
    timing_path = args.output_dir / "inference_timings.csv"; write_csv(timing_path, timings)
    profiles = runner.finish_profiles() if args.mode == "placement" else None
    result = {
        "schema": "journal_phase6_system_run_v1", "status": "PASSED", "mode": args.mode,
        "hardware": "海光 K100 AI 加速卡", "hostname": platform.node(), "variant": args.label,
        "config": identity(args.config), "input": input_id, "data_role": "fixed configuration input identified by SHA256",
        
        "runtime": {**runner.runtime, "load_seconds": runner.load_seconds, "first_inference_ms": first_ms},
        "protocol": {"warmup": args.warmup, "measured": args.measured, "requested_duration_seconds": args.duration_seconds if args.mode == "stability" else None},
        "measurement_window": {"started_unix_ns": measured_started_ns, "ended_unix_ns": measured_ended_ns, "wall_seconds": wall_seconds},
        "measurements": {**summary(timings, wall_seconds), "errors": errors, "nonfinite": nonfinite, "prediction_drift": drift},
        "baseline": {"logits_sha256": baseline_logits_hash, "prediction_sha256": baseline_prediction_hash},
        "profiles": profiles, "timings": identity(timing_path),
        "gates": {"all_outputs_finite": nonfinite == 0, "prediction_hash_stable": drift == 0, "zero_errors": errors == 0, "cpu_fallback_disabled": runner.runtime["cpu_fallback_disabled"]},
        "total_process_work_seconds": time.perf_counter() - total_started,
    }
    if not all(result["gates"].values()): raise RuntimeError(f"System measurement gate failed: {result['gates']}")
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASSED", "mode": args.mode, "variant": args.label, "calls": len(timings), "median_ms": result["measurements"]["median_ms"]}), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
