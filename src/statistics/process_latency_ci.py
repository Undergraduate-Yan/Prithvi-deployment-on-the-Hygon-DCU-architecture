#!/usr/bin/env python3
'Process-level uncertainty from retained latency calls.'

from __future__ import annotations

import argparse
import os
import statistics
from collections import defaultdict
from pathlib import Path

from analysis_utils import bootstrap_median_ci, load_json, provenance, read_csv, require_file, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = require_file(args.config.resolve())
    cfg = load_json(config_path)
    section = cfg["same_node_ci"]
    source = require_file((config_path.parent / section["latency_csv"]).resolve())
    by_trial: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in read_csv(source):
        if row.get("error"):
            raise ValueError(f"raw latency record contains an error: {row}")
        by_trial[(row["node"], row["role"], row["trial"])].append(float(row["latency_ms"]))
    trial_medians: dict[tuple[str, str], list[float]] = defaultdict(list)
    call_counts: dict[tuple[str, str], int] = defaultdict(int)
    for (node, role, _trial), values in by_trial.items():
        trial_medians[(node, role)].append(statistics.median(values))
        call_counts[(node, role)] += len(values)
    output: list[dict[str, object]] = []
    seed = int(cfg["seed"])
    resamples = int(cfg["bootstrap_resamples"])
    for (node, role), medians in sorted(trial_medians.items()):
        low, high = bootstrap_median_ci(medians, seed, resamples)
        output.append({
            "node": node, "role": role, "fresh_processes": len(medians), "measured_calls": call_counts[(node, role)],
            "median_of_process_medians_ms": statistics.median(medians),
            "process_level_bootstrap_ci95_low_ms": low, "process_level_bootstrap_ci95_high_ms": high,
            "resamples": resamples, "seed": seed,
            "scope": "existing randomized common C++ runner; one tile; pipeline plus FP32-logits D2H",
        })
    csv_path = args.output_dir / "process_latency_ci.csv"
    json_path = args.output_dir / "process_latency_ci.json"
    fields = ["node", "role", "fresh_processes", "measured_calls", "median_of_process_medians_ms", "process_level_bootstrap_ci95_low_ms", "process_level_bootstrap_ci95_high_ms", "resamples", "seed", "scope"]
    write_csv(csv_path, fields, output)
    write_json(json_path, {
        "schema": "cloud_process_level_latency_ci_v1", "status": "DERIVED_FROM_EXISTING_RAW_LATENCIES",
        "new_benchmark_performed": False, "provenance": provenance(config_path.parents[2], [source], seed),
    })
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
