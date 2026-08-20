#!/usr/bin/env python3
"""Aggregate three fresh-process trials of the admitted FP32 25-segment baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_TRIAL_SCRIPT_SHA256 = "31af48acd99b1ddbc614854bbf99721ca0f1d86bc984b21605883740a7f743dd"
EXPECTED_CANDIDATE = (
    60_551_388,
    "593db691abcb2e07dd8df2f1fabc8315eeab08380161c93199778bc1963f299f",
)
EXPECTED_TEST90 = (
    39_291,
    "0904da34078801106b6b414dcc17d1b612b448a997460563fc7055c4e6988b3b",
)
EXPECTED_TRIALS = 3
EXPECTED_MEASURED = 100
MEDIAN_CV_MAX_PERCENT = 5.0
SCOPES = ("model_only", "end_to_end_logits")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, expected: dict | tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if expected is not None:
        pair = (
            (int(expected["size_bytes"]), str(expected["sha256"]))
            if isinstance(expected, dict)
            else expected
        )
        if (item["size_bytes"], item["sha256"]) != pair:
            raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.min(values) <= 0:
        raise RuntimeError("latency array contract failure")
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


def cv_percent(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all() or np.mean(array) <= 0:
        raise RuntimeError("CV input contract failure")
    return float(np.std(array, ddof=1) / np.mean(array) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    if (run_root / "summary.json").exists() or (run_root / "summary.csv").exists():
        raise RuntimeError("refusing to overwrite an existing frozen performance summary")

    records = []
    csv_rows = []
    prerequisite_pairs = []
    result_hashes = set()
    for trial_index in range(1, EXPECTED_TRIALS + 1):
        trial_root = run_root / f"trial_{trial_index}"
        output_root = trial_root / "output"
        result_path = output_root / "result.json"
        result = load_json(result_path)
        if result.get("schema") != "phase11_fp32_segment25_headbarrier_performance_trial_v1":
            raise RuntimeError(f"trial result schema drift: {trial_index}")
        if result.get("status") != "passed" or not result.get("claims", {}).get(
            "same_protocol_fp32_performance_trial_valid"
        ):
            raise RuntimeError(f"failed FP32 performance trial: {trial_index}")
        if not result.get("claims", {}).get("eligible_as_same_protocol_speedup_denominator"):
            raise RuntimeError(f"trial is not an eligible FP32 denominator: {trial_index}")
        if result.get("trial_index") != trial_index:
            raise RuntimeError(f"trial index drift: {trial_index}")
        expected_order = (
            ["model_only", "end_to_end_logits"]
            if trial_index in (1, 3)
            else ["end_to_end_logits", "model_only"]
        )
        if result.get("protocol", {}).get("scope_order") != expected_order:
            raise RuntimeError(f"scope order drift: {trial_index}")
        identities = result.get("identities", {})
        if identities.get("benchmark_script", {}).get("sha256") != EXPECTED_TRIAL_SCRIPT_SHA256:
            raise RuntimeError(f"trial script identity drift: {trial_index}")
        candidate_pair = (
            int(identities.get("candidate", {}).get("size_bytes", -1)),
            str(identities.get("candidate", {}).get("sha256")),
        )
        test90_pair = (
            int(identities.get("test90_result", {}).get("size_bytes", -1)),
            str(identities.get("test90_result", {}).get("sha256")),
        )
        if candidate_pair != EXPECTED_CANDIDATE or test90_pair != EXPECTED_TEST90:
            raise RuntimeError(f"frozen candidate/admission identity drift: {trial_index}")
        prerequisite_pairs.append((candidate_pair, test90_pair))

        raw_arrays = {}
        recomputed = {}
        for scope in SCOPES:
            latency_path = output_root / f"{scope}_latencies_ms.npy"
            latency_identity = identity(
                latency_path, result.get("artifacts", {}).get(f"{scope}_latencies_ms")
            )
            values = np.load(latency_path, allow_pickle=False).astype(np.float64, copy=False)
            if values.shape != (EXPECTED_MEASURED,):
                raise RuntimeError(f"latency vector shape drift: {trial_index}/{scope}")
            current = statistics(values)
            stored = result.get("statistics", {}).get(scope, {})
            for key in (
                "median_ms",
                "p95_ms",
                "p99_ms",
                "mean_ms",
                "throughput_samples_per_second",
            ):
                if not np.isclose(current[key], stored.get(key), rtol=0, atol=1.0e-12):
                    raise RuntimeError(f"stored statistic drift: {trial_index}/{scope}/{key}")
            raw_arrays[scope] = values
            recomputed[scope] = current
            csv_rows.append(
                {
                    "trial": trial_index,
                    "scope": scope,
                    "median_ms": current["median_ms"],
                    "p95_ms": current["p95_ms"],
                    "p99_ms": current["p99_ms"],
                    "mean_ms": current["mean_ms"],
                    "throughput_samples_per_second": current[
                        "throughput_samples_per_second"
                    ],
                    "process_max_rss_mib": (
                        None
                        if result.get("runtime", {}).get("process_max_rss_kib") is None
                        else result["runtime"]["process_max_rss_kib"] / 1024.0
                    ),
                }
            )
            # Keep the already verified identity in the record without rereading.
            result.setdefault("_verified_latency_identities", {})[scope] = latency_identity

        result_identity = identity(result_path)
        if result_identity["sha256"] in result_hashes:
            raise RuntimeError("fresh-process trial result hashes are not unique")
        result_hashes.add(result_identity["sha256"])
        telemetry_path = trial_root / "telemetry.log"
        records.append(
            {
                "trial_index": trial_index,
                "scope_order": expected_order,
                "result": result_identity,
                "telemetry_raw": identity(telemetry_path),
                "statistics": recomputed,
                "raw_arrays": raw_arrays,
                "latency_artifacts": result["_verified_latency_identities"],
                "session_load_seconds_excluded": result.get("runtime", {}).get(
                    "session_load_seconds_excluded"
                ),
                "process_max_rss_kib": result.get("runtime", {}).get("process_max_rss_kib"),
            }
        )

    if len(set(prerequisite_pairs)) != 1:
        raise RuntimeError("candidate or test90 prerequisite drifted across trials")

    scope_summaries = {}
    scope_gates = {}
    for scope in SCOPES:
        medians = np.asarray(
            [row["statistics"][scope]["median_ms"] for row in records], dtype=np.float64
        )
        p95s = np.asarray(
            [row["statistics"][scope]["p95_ms"] for row in records], dtype=np.float64
        )
        p99s = np.asarray(
            [row["statistics"][scope]["p99_ms"] for row in records], dtype=np.float64
        )
        throughputs = np.asarray(
            [
                row["statistics"][scope]["throughput_samples_per_second"]
                for row in records
            ],
            dtype=np.float64,
        )
        pooled = np.concatenate([row["raw_arrays"][scope] for row in records])
        median_cv = cv_percent(medians)
        scope_summaries[scope] = {
            "median_of_trial_medians_ms": float(np.median(medians)),
            "trial_median_min_ms": float(np.min(medians)),
            "trial_median_max_ms": float(np.max(medians)),
            "trial_median_cv_percent": median_cv,
            "median_of_trial_p95_ms": float(np.median(p95s)),
            "median_of_trial_p99_ms": float(np.median(p99s)),
            "median_trial_throughput_samples_per_second": float(np.median(throughputs)),
            "pooled_300_calls_descriptive_only": statistics(pooled),
        }
        scope_gates[f"{scope}_trial_median_cv_le_5pct"] = (
            median_cv <= MEDIAN_CV_MAX_PERCENT
        )

    gates = {
        "three_fresh_process_trials_passed": len(records) == EXPECTED_TRIALS
        and len(result_hashes) == EXPECTED_TRIALS,
        "each_scope_has_100_measurements_per_trial": all(
            row["raw_arrays"][scope].shape == (EXPECTED_MEASURED,)
            for row in records
            for scope in SCOPES
        ),
        "candidate_identity_constant_across_trials": len(set(prerequisite_pairs)) == 1,
        **scope_gates,
    }
    passed = all(gates.values())
    summary = {
        "schema": "phase11_fp32_segment25_headbarrier_performance_summary_v1",
        "status": "passed" if passed else "unstable",
        "identities": {
            "aggregate_script": identity(Path(__file__)),
            "expected_trial_script_sha256": EXPECTED_TRIAL_SCRIPT_SHA256,
            "candidate": {
                "size_bytes": EXPECTED_CANDIDATE[0],
                "sha256": EXPECTED_CANDIDATE[1],
            },
            "passed_test90_result": {
                "size_bytes": EXPECTED_TEST90[0],
                "sha256": EXPECTED_TEST90[1],
            },
        },
        "protocol": {
            "track": "same_protocol_fp32_reference",
            "batch": 1,
            "warmup_per_scope_per_trial": 30,
            "measurements_per_scope_per_trial": 100,
            "fresh_process_trials": 3,
            "trial_median_cv_max_percent": MEDIAN_CV_MAX_PERCENT,
            "model_only_is_primary": True,
            "end_to_end_logits_is_secondary": True,
        },
        "scope_summaries": scope_summaries,
        "gates": gates,
        "claims": {
            "same_protocol_fp32_baseline_valid": passed,
            "formal_speedup_denominator_allowed": passed,
            "model_only_absolute_latency_claim_allowed": passed,
            "end_to_end_logits_absolute_latency_claim_allowed": passed,
            "monolithic_fp32_latency_claim": False,
            "deployment_ready": False,
        },
        "trial_artifacts": [
            {key: value for key, value in row.items() if key != "raw_arrays"}
            for row in records
        ],
        "evidence_boundary": {
            "baseline_is_25_segments_not_monolithic": True,
            "speedup_requires_a_candidate_measured_under_the_matching_scope": True,
            "pooled_300_call_statistics_are_descriptive_only": True,
            "provider_placement_is_inherited_from_frozen_admission_evidence": True,
            "raw_hy_smi_telemetry_is_preserved_but_not_parsed_as_device_memory_here": True,
            "process_max_rss_is_not_K100_device_memory": True,
        },
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = run_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
