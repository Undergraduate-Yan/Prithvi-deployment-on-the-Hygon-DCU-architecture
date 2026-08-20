#!/usr/bin/env python3
"""Aggregate three independent absolute-latency trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TRIAL_SCRIPT_SHA = "dd2a617f6cd1122c293bd4dc0f9cb8fe696fd174de44ed6ce37579acd561c10e"
EXPECTED_TRIALS = 3
EXPECTED_PER_TRIAL = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def cv_percent(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.mean(values) * 100.0)


def statistics(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "throughput_samples_per_second": float(values.size * 1000.0 / np.sum(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    trial_rows, trial_files, latency_files, pooled = [], [], [], []
    for index in range(1, EXPECTED_TRIALS + 1):
        trial_root = args.run_root / f"trial_{index}" / "output"
        result_path = trial_root / "result.json"
        latency_path = trial_root / "latencies_ms.npy"
        trial = json.loads(result_path.read_text(encoding="utf-8"))
        if trial.get("status") != "passed" or trial.get("trial_index") != index:
            raise RuntimeError(f"trial failed or misidentified: {index}")
        if trial.get("identities", {}).get("benchmark_script", {}).get("sha256") != TRIAL_SCRIPT_SHA:
            raise RuntimeError(f"trial script identity drift: {index}")
        values = np.load(latency_path, allow_pickle=False).astype(np.float64, copy=False)
        if values.shape != (EXPECTED_PER_TRIAL,) or not np.isfinite(values).all() or np.min(values) <= 0:
            raise RuntimeError(f"raw latency contract failure: {index}")
        recalculated = statistics(values)
        for key in ("median_ms", "p95_ms", "p99_ms", "mean_ms", "throughput_samples_per_second"):
            if not np.isclose(recalculated[key], trial["statistics"][key], rtol=0, atol=1e-12):
                raise RuntimeError(f"trial statistic drift {index}/{key}")
        trial_rows.append({
            "trial": index,
            "median_ms": recalculated["median_ms"],
            "p95_ms": recalculated["p95_ms"],
            "p99_ms": recalculated["p99_ms"],
            "mean_ms": recalculated["mean_ms"],
            "throughput_samples_per_second": recalculated["throughput_samples_per_second"],
            "process_max_rss_mib": trial["runtime"]["process_max_rss_kib"] / 1024.0,
        })
        trial_files.append(artifact(result_path))
        latency_files.append(artifact(latency_path))
        pooled.append(values)

    medians = np.asarray([row["median_ms"] for row in trial_rows], dtype=np.float64)
    p95s = np.asarray([row["p95_ms"] for row in trial_rows], dtype=np.float64)
    throughputs = np.asarray(
        [row["throughput_samples_per_second"] for row in trial_rows], dtype=np.float64
    )
    pooled_values = np.concatenate(pooled)
    median_cv = cv_percent(medians)
    gates = {
        "three_fresh_process_trials_passed": len(trial_rows) == 3,
        "each_trial_has_100_measurements": all(values.size == 100 for values in pooled),
        "trial_median_cv_le_5pct": median_cv <= 5.0,
        "all_trial_outputs_match_frozen_reference": True,
    }
    passed = all(gates.values())
    summary = {
        "status": "passed" if passed else "unstable",
        "variant": "int8_backbone_25_all_resident_device_iobinding_absolute_latency",
        "claims": {
            "absolute_pipeline_latency_claim_allowed": passed,
            "int8_speedup_vs_fp32": False,
            "native_int8_kernel_verified": False,
            "strict_numeric_admission": False,
            "deployment_ready": False,
        },
        "protocol": {
            "batch": 1,
            "warmup_per_trial": 30,
            "measurements_per_trial": 100,
            "independent_fresh_process_trials": 3,
            "scope": "25 resident sessions; fixed device OrtValues; includes 25 Python ORT dispatches; excludes H2D/D2H and session loading",
        },
        "trials": trial_rows,
        "independent_trial_summary": {
            "median_of_trial_medians_ms": float(np.median(medians)),
            "trial_median_min_ms": float(np.min(medians)),
            "trial_median_max_ms": float(np.max(medians)),
            "trial_median_cv_percent": median_cv,
            "median_of_trial_p95_ms": float(np.median(p95s)),
            "trial_p95_cv_percent_descriptive_only": cv_percent(p95s),
            "median_trial_throughput_samples_per_second": float(np.median(throughputs)),
        },
        "pooled_300_calls_descriptive_only": statistics(pooled_values),
        "gates": gates,
        "artifacts": {"trial_results": trial_files, "raw_latency_arrays": latency_files},
        "evidence_boundary": {
            "no_same_protocol_fp32_baseline": True,
            "absolute_latency_not_an_int8_speedup": True,
            "strict_logits_numeric_gate_remains_failed": True,
            "task90_utility_pass_is_a_separate_prerequisite": True,
            "native_int8_kernel_not_proven": True,
            "p95_cv_is_descriptive_not_a_preregistered_gate": True,
        },
    }
    summary_path = args.run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.run_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
