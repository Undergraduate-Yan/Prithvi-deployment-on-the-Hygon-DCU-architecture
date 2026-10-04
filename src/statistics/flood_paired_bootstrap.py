#!/usr/bin/env python3
'Research implementation: flood paired bootstrap.'

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from common import identity, write_json


METRICS = ("miou", "water_iou", "boundary_water_iou", "pixel_accuracy", "agreement")


def coverage_bin(value: float) -> str:
    if not math.isfinite(value):
        return "undefined"
    if value == 0:
        return "0%"
    if value <= 0.01:
        return "(0,1%]"
    if value <= 0.10:
        return "(1,10%]"
    if value <= 0.40:
        return "(10,40%]"
    return ">40%"


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            row["water_coverage"] = float(raw["water_coverage"])
            for metric in METRICS:
                row[metric] = float(raw[metric])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def finite_pairs(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def bootstrap_mean_ci(values: np.ndarray, iterations: int, rng: np.random.RandomState) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    draws = rng.randint(0, values.size, size=(iterations, values.size))
    means = values[draws].mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def make_plots(
    differences: Sequence[Dict[str, Any]],
    stratified: Sequence[Dict[str, Any]],
    output_prefix: Path,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted({str(row["variant"]) for row in differences})
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS[:4]):
        series = [
            finite_pairs(float(row["delta_" + metric + "_pp"]) for row in differences if row["variant"] == variant)
            for variant in variants
        ]
        axis.boxplot(series, labels=variants, showmeans=True)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(metric.replace("_", " ").title() + " paired delta")
        axis.set_ylabel("percentage points")
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("Scene-level paired differences versus the frozen FP32 reference")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    distribution_png = output_prefix.with_name("scene_metric_distribution.png")
    distribution_pdf = output_prefix.with_name("scene_metric_distribution.pdf")
    figure.savefig(distribution_png, dpi=180)
    figure.savefig(distribution_pdf)
    plt.close(figure)

    bins = ["0%", "(0,1%]", "(1,10%]", "(10,40%]", ">40%"]
    figure, axes = plt.subplots(len(variants), 1, figsize=(10, max(3.5, 3 * len(variants))), squeeze=False, constrained_layout=True)
    for index, variant in enumerate(variants):
        axis = axes[index, 0]
        selected = [row for row in stratified if row["variant"] == variant and row["metric"] == "water_iou"]
        mapping = {str(row["coverage_bin"]): float(row["mean_delta_pp"]) for row in selected}
        values = [mapping.get(name, float("nan")) for name in bins]
        axis.bar(bins, values, color="#1E4FA8")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title("%s: Water IoU delta by water coverage" % variant)
        axis.set_ylabel("percentage points")
    stratified_png = output_prefix.with_name("coverage_stratified_results.png")
    stratified_pdf = output_prefix.with_name("coverage_stratified_results.pdf")
    figure.savefig(stratified_png, dpi=180)
    figure.savefig(stratified_pdf)
    plt.close(figure)
    return [distribution_png, distribution_pdf, stratified_png, stratified_pdf]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("iterations must be at least 1000")

    rows = read_rows(args.input)
    by_variant_sample: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row["variant"]), str(row["sample_id"])): row for row in rows
    }
    variants = sorted({str(row["variant"]) for row in rows if row["variant"] != args.reference})
    reference_ids = sorted(str(row["sample_id"]) for row in rows if row["variant"] == args.reference)
    if not reference_ids:
        raise ValueError("reference variant is absent")

    differences: List[Dict[str, Any]] = []
    for variant in variants:
        variant_ids = sorted(str(row["sample_id"]) for row in rows if row["variant"] == variant)
        if variant_ids != reference_ids:
            raise ValueError("scene identity mismatch for %s" % variant)
        for sample_id in reference_ids:
            reference_row = by_variant_sample[(args.reference, sample_id)]
            variant_row = by_variant_sample[(variant, sample_id)]
            output: Dict[str, Any] = {
                "variant": variant,
                "reference": args.reference,
                "sample_id": sample_id,
                "water_coverage": reference_row["water_coverage"],
                "coverage_bin": coverage_bin(float(reference_row["water_coverage"])),
            }
            for metric in METRICS:
                delta = float(variant_row[metric]) - float(reference_row[metric])
                output["delta_" + metric] = delta
                output["delta_" + metric + "_pp"] = delta * 100.0
            differences.append(output)

    difference_fields = ["variant", "reference", "sample_id", "water_coverage", "coverage_bin"]
    for metric in METRICS:
        difference_fields.extend(["delta_" + metric, "delta_" + metric + "_pp"])
    paired_path = args.output_dir / "paired_differences.csv"
    write_csv(paired_path, differences, difference_fields)

    rng = np.random.RandomState(args.seed)
    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        for metric in METRICS:
            values = finite_pairs(
                float(row["delta_" + metric]) for row in differences if row["variant"] == variant
            )
            low, high = bootstrap_mean_ci(values, args.iterations, rng)
            summary_rows.append(
                {
                    "variant": variant,
                    "reference": args.reference,
                    "metric": metric,
                    "defined_scene_pairs": int(values.size),
                    "undefined_scene_pairs": len(reference_ids) - int(values.size),
                    "mean_delta": float(values.mean()) if values.size else float("nan"),
                    "mean_delta_pp": float(values.mean() * 100.0) if values.size else float("nan"),
                    "median_delta_pp": float(np.median(values) * 100.0) if values.size else float("nan"),
                    "bootstrap_95ci_low_pp": low * 100.0,
                    "bootstrap_95ci_high_pp": high * 100.0,
                    "positive_pairs": int((values > 0).sum()),
                    "zero_pairs": int((values == 0).sum()),
                    "negative_pairs": int((values < 0).sum()),
                }
            )
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    summary_path = args.output_dir / "bootstrap_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    stratified_rows: List[Dict[str, Any]] = []
    grouped: DefaultDict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in differences:
        for metric in METRICS:
            value = float(row["delta_" + metric])
            if math.isfinite(value):
                grouped[(str(row["variant"]), str(row["coverage_bin"]), metric)].append(value)
    for (variant, bin_name, metric), values_list in sorted(grouped.items()):
        values = np.asarray(values_list, dtype=np.float64)
        low, high = bootstrap_mean_ci(values, args.iterations, rng)
        stratified_rows.append(
            {
                "variant": variant,
                "coverage_bin": bin_name,
                "metric": metric,
                "scene_pairs": int(values.size),
                "mean_delta_pp": float(values.mean() * 100.0),
                "bootstrap_95ci_low_pp": low * 100.0,
                "bootstrap_95ci_high_pp": high * 100.0,
            }
        )
    stratified_path = args.output_dir / "coverage_stratified_results.csv"
    write_csv(stratified_path, stratified_rows, list(stratified_rows[0].keys()))
    plot_paths = make_plots(differences, stratified_rows, args.output_dir / "plots")

    payload = {
        "schema": "paired_scene_bootstrap_v1",
        "status": "passed",
        "reference": args.reference,
        "variants": variants,
        "scene_count": len(reference_ids),
        "seed": args.seed,
        "iterations": args.iterations,
        "estimand": "mean of paired scene-level metric differences",
        "nan_rule": "exclude only scene pairs undefined for the selected metric; never replace with 0 or 1",
        "input": identity(args.input),
        "outputs": {
            "paired_differences": identity(paired_path),
            "bootstrap_summary": identity(summary_path),
            "coverage_stratified_results": identity(stratified_path),
            "plots": [identity(path) for path in plot_paths],
        },
    }
    write_json(args.output_dir / "bootstrap_manifest.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
