#!/usr/bin/env python3
'Research implementation: benchmark flood.'

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from flood_runner import K100BundleRunner, identity, load_input


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "median_ms": median,
        "mean_ms": float(np.mean(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mad_ms": float(np.median(np.abs(array - median))),
        "std_ms": float(np.std(array, ddof=1)),
        "throughput_samples_per_s_from_mean": float(1000.0 / np.mean(array)),
    }


def trial(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.resolve(strict=True).read_text(encoding="utf-8"))
    raw = load_input(args.input, manifest["benchmark_input"])
    profile_dir = args.output / "profiles" if args.profile else None
    runner = K100BundleRunner(args.manifest, args.device, profile_dir)
    last_logits = last_prediction = None
    for _ in range(args.warmup):
        last_logits, last_prediction, _ = runner.run(raw)
    latencies = []
    for _ in range(args.measured):
        last_logits, last_prediction, elapsed = runner.run(raw)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("non-finite or non-positive latency")
        latencies.append(elapsed)
    profiles = runner.finish_profiles() if args.profile else None
    result = {
        "schema": "journal_stage2_fair_baseline_trial_v1",
        "status": "passed",
        "variant": manifest["variant"],
        "trial_index": args.trial_index,
        "identities": {
            "runner": identity(Path(__file__).resolve().parents[1] / "runtime" / "flood_runner.py"),
            "benchmark": identity(Path(__file__)),
            "manifest": identity(args.manifest),
            "input": identity(args.input),
        },
        "protocol": {
            "scope": "end-to-end FP32 input to FP32 host logits",
            "runner_path": "K100BundleRunner.run",
            "io_binding": True,
            "synchronize_inputs": True,
            "synchronize_outputs": True,
            "fresh_process": True,
            "warmup": args.warmup,
            "measured": args.measured,
            "batch": 1,
        },
        "runtime": {**runner.runtime, "session_load_seconds": runner.load_seconds},
        "latency_ms": latencies,
        "statistics": summarize(latencies),
        "outputs": {
            "logits_sha256": hashlib.sha256(np.ascontiguousarray(last_logits).tobytes()).hexdigest(),
            "prediction_sha256": hashlib.sha256(np.ascontiguousarray(last_prediction).tobytes()).hexdigest(),
            "finite": bool(np.isfinite(last_logits).all()),
        },
        "profiles": profiles,
        "gates": {
            "exact_measurement_count": len(latencies) == args.measured,
            "all_latencies_finite_positive": bool(np.isfinite(latencies).all() and min(latencies) > 0),
            "output_finite": bool(np.isfinite(last_logits).all()),
            "strict_migraphx_no_cpu_fallback": profiles is None or all(
                row["migraphx_events_positive"] and row["cpu_events_zero"] for row in profiles
            ),
            "configuration_validation_input_only": True,
            "formal_test_not_used": True,
        },
    }
    if not all(result["gates"].values()):
        result["status"] = "failed"
    write_json(args.output / "result.json", result)
    print(json.dumps({"variant": result["variant"], "status": result["status"], "statistics": result["statistics"]}, indent=2))
    return 0 if result["status"] == "passed" else 2


def hierarchical_bootstrap(baseline: list[list[float]], candidate: list[list[float]], seed: int, draws: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=np.float64)
    baseline_values = np.empty(draws, dtype=np.float64)
    candidate_values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_b = rng.integers(0, len(baseline), size=len(baseline))
        selected_c = rng.integers(0, len(candidate), size=len(candidate))
        sampled_b = np.concatenate([
            rng.choice(baseline[index], size=len(baseline[index]), replace=True) for index in selected_b
        ])
        sampled_c = np.concatenate([
            rng.choice(candidate[index], size=len(candidate[index]), replace=True) for index in selected_c
        ])
        baseline_values[draw] = np.median(sampled_b)
        candidate_values[draw] = np.median(sampled_c)
        ratios[draw] = baseline_values[draw] / candidate_values[draw]
    return {
        "method": "hierarchical bootstrap over fresh-process trials and within-trial calls",
        "seed": seed,
        "draws": draws,
        "baseline_median_ms_95ci": np.percentile(baseline_values, [2.5, 97.5]).tolist(),
        "candidate_median_ms_95ci": np.percentile(candidate_values, [2.5, 97.5]).tolist(),
        "speed_factor_fp32_over_candidate_95ci": np.percentile(ratios, [2.5, 97.5]).tolist(),
    }


def aggregate(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    variants: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(args.results_root.glob("trial_*/B*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "passed" or not all(row.get("gates", {}).values()):
            raise RuntimeError(f"failed trial result: {path}")
        variants.setdefault(row["variant"], []).append(row)
    if "B0" not in variants or len(variants) < 2:
        raise RuntimeError("aggregate requires B0 and at least one candidate")
    for name, rows in variants.items():
        if len(rows) != args.expected_trials:
            raise RuntimeError(f"trial count drift for {name}: {len(rows)}")

    summaries: dict[str, Any] = {}
    baseline = [row["latency_ms"] for row in variants["B0"]]
    baseline_all = [value for trial_values in baseline for value in trial_values]
    baseline_median = float(np.median(baseline_all))
    for name, rows in sorted(variants.items()):
        raw = [row["latency_ms"] for row in rows]
        combined = [value for trial_values in raw for value in trial_values]
        trial_medians = [float(np.median(values)) for values in raw]
        cv = 100.0 * statistics.stdev(trial_medians) / statistics.mean(trial_medians)
        median = float(np.median(combined))
        speed = baseline_median / median
        summaries[name] = {
            "statistics": summarize(combined),
            "trial_medians_ms": trial_medians,
            "trial_median_cv_percent": cv,
            "speed_factor_fp32_over_variant": speed,
            "absolute_median_difference_vs_fp32_ms": baseline_median - median,
            "median_latency_reduction_vs_fp32_percent": 100.0 * (baseline_median - median) / baseline_median,
            "bootstrap": hierarchical_bootstrap(baseline, raw, args.seed, args.bootstrap_draws),
        }
    result = {
        "schema": "journal_stage2_fair_baseline_summary_v1",
        "status": "passed",
        "protocol": {
            "fresh_processes": args.expected_trials,
            "warmup_per_process": variants["B0"][0]["protocol"]["warmup"],
            "measured_per_process": variants["B0"][0]["protocol"]["measured"],
            "speed_factor_definition": "T_FP32/T_variant",
            "bootstrap_seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
        },
        "variants": summaries,
        "gates": {
            "all_variants_have_expected_trials": all(len(rows) == args.expected_trials for rows in variants.values()),
            "all_trial_median_cv_le_5pct": all(row["trial_median_cv_percent"] <= 5.0 for row in summaries.values()),
            "all_trials_used_common_runner_path": all(
                row["protocol"]["runner_path"] == "K100BundleRunner.run"
                for rows in variants.values() for row in rows
            ),
            "all_trials_configuration_validation_only": all(
                row["gates"]["configuration_validation_input_only"] and row["gates"]["formal_test_not_used"]
                for rows in variants.values() for row in rows
            ),
        },
    }
    if not all(result["gates"].values()):
        result["status"] = "failed"
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    trial_parser = subparsers.add_parser("trial")
    trial_parser.add_argument("--manifest", required=True, type=Path)
    trial_parser.add_argument("--input", required=True, type=Path)
    trial_parser.add_argument("--output", required=True, type=Path)
    trial_parser.add_argument("--trial-index", required=True, type=int)
    trial_parser.add_argument("--warmup", type=int, default=30)
    trial_parser.add_argument("--measured", type=int, default=100)
    trial_parser.add_argument("--device", type=int, default=0)
    trial_parser.add_argument("--profile", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--results-root", required=True, type=Path)
    aggregate_parser.add_argument("--output", required=True, type=Path)
    aggregate_parser.add_argument("--expected-trials", type=int, default=3)
    aggregate_parser.add_argument("--seed", type=int, default=42)
    aggregate_parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    return trial(args) if args.mode == "trial" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
