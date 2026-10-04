#!/usr/bin/env python3
'Research implementation: flood metrics.'

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from common import identity, write_json


METRIC_FIELDS = (
    "miou",
    "water_iou",
    "boundary_water_iou",
    "pixel_accuracy",
    "agreement",
)


def parse_variant(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("variant must be LABEL=PATH")
    label, raw_path = text.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("variant label must be non-empty")
    return label.strip(), Path(raw_path)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_dilation(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("binary_dilation expects an HxW array")
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    height, width = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def water_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    mask_bool = mask.astype(bool)
    dilated = binary_dilation(mask_bool, radius=radius)
    eroded = ~binary_dilation(~mask_bool, radius=radius)
    return dilated & ~eroded


def confusion_counts(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Dict[str, int]:
    pred = prediction.astype(np.uint8)
    truth = target.astype(np.int64)
    return {
        "tn": int(((pred == 0) & (truth == 0) & valid).sum()),
        "fp": int(((pred == 1) & (truth == 0) & valid).sum()),
        "fn": int(((pred == 0) & (truth == 1) & valid).sum()),
        "tp": int(((pred == 1) & (truth == 1) & valid).sum()),
    }


def metrics_for_scene(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_prediction: np.ndarray,
) -> Dict[str, Any]:
    valid = target != -1
    valid_pixels = int(valid.sum())
    if valid_pixels <= 0:
        return {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "valid_pixels": 0,
            "water_pixels": 0,
            "water_coverage": float("nan"),
            "background_iou": float("nan"),
            "water_iou": float("nan"),
            "miou": float("nan"),
            "boundary_intersection": 0,
            "boundary_union": 0,
            "boundary_water_iou": float("nan"),
            "pixel_accuracy": float("nan"),
            "agreement": float("nan"),
            "changed_valid_pixels": 0,
        }
    counts = confusion_counts(prediction, target, valid)
    water_union = counts["tp"] + counts["fp"] + counts["fn"]
    background_union = counts["tn"] + counts["fp"] + counts["fn"]
    water_iou = safe_ratio(counts["tp"], water_union)
    background_iou = safe_ratio(counts["tn"], background_union)
    defined_ious = [value for value in (background_iou, water_iou) if math.isfinite(value)]
    miou = float(np.mean(defined_ious)) if defined_ious else float("nan")

    prediction_boundary = water_boundary(prediction == 1, radius=2)
    target_boundary = water_boundary(target == 1, radius=2)
    boundary_intersection = int((prediction_boundary & target_boundary & valid).sum())
    boundary_union = int(((prediction_boundary | target_boundary) & valid).sum())

    water_pixels = int(((target == 1) & valid).sum())
    correct = counts["tn"] + counts["tp"]
    changed = int(((prediction != reference_prediction) & valid).sum())
    return {
        **counts,
        "valid_pixels": valid_pixels,
        "water_pixels": water_pixels,
        "water_coverage": water_pixels / valid_pixels,
        "background_iou": background_iou,
        "water_iou": water_iou,
        "miou": miou,
        "boundary_intersection": boundary_intersection,
        "boundary_union": boundary_union,
        "boundary_water_iou": safe_ratio(boundary_intersection, boundary_union),
        "pixel_accuracy": correct / valid_pixels,
        "agreement": 1.0 - changed / valid_pixels,
        "changed_valid_pixels": changed,
    }


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as payload:
        required = {"predictions", "targets", "sample_ids"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError("%s is missing NPZ keys: %s" % (path, sorted(missing)))
        predictions = np.asarray(payload["predictions"])
        targets = np.asarray(payload["targets"])
        sample_ids = np.asarray(payload["sample_ids"])
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must both have shape [N,H,W]")
    if sample_ids.shape != (predictions.shape[0],):
        raise ValueError("sample_ids length does not match predictions")
    return predictions, targets, sample_ids


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "sample_index",
        "sample_id",
        "source_npz_sha256",
        "valid_pixels",
        "water_pixels",
        "water_coverage",
        "tn",
        "fp",
        "fn",
        "tp",
        "background_iou",
        "water_iou",
        "miou",
        "boundary_intersection",
        "boundary_union",
        "boundary_water_iou",
        "pixel_accuracy",
        "agreement",
        "changed_valid_pixels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    totals = {key: sum(int(row[key]) for row in materialized) for key in ("tn", "fp", "fn", "tp")}
    water_union = totals["tp"] + totals["fp"] + totals["fn"]
    background_union = totals["tn"] + totals["fp"] + totals["fn"]
    water_iou = safe_ratio(totals["tp"], water_union)
    background_iou = safe_ratio(totals["tn"], background_union)
    boundary_intersection = sum(int(row["boundary_intersection"]) for row in materialized)
    boundary_union = sum(int(row["boundary_union"]) for row in materialized)
    valid_pixels = sum(int(row["valid_pixels"]) for row in materialized)
    changed = sum(int(row["changed_valid_pixels"]) for row in materialized)
    return {
        "scenes": len(materialized),
        "valid_pixels": valid_pixels,
        "confusion_matrix": [[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]],
        "miou": float(np.nanmean([background_iou, water_iou])),
        "water_iou": water_iou,
        "boundary_water_iou": safe_ratio(boundary_intersection, boundary_union),
        "pixel_accuracy": safe_ratio(totals["tn"] + totals["tp"], valid_pixels),
        "agreement": 1.0 - changed / valid_pixels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=parse_variant)
    parser.add_argument("--variant", action="append", default=[], type=parse_variant)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--frozen-test-ids", type=Path)
    args = parser.parse_args()

    reference_label, reference_path = args.reference
    variants = [(reference_label, reference_path)] + list(args.variant)
    reference_predictions, reference_targets, reference_ids = load_npz(reference_path)
    rows: List[Dict[str, Any]] = []
    inputs = []
    for label, path in variants:
        predictions, targets, sample_ids = load_npz(path)
        if not np.array_equal(sample_ids, reference_ids):
            raise ValueError("sample identity/order mismatch for %s" % label)
        if not np.array_equal(targets, reference_targets):
            raise ValueError("target tensor mismatch for %s" % label)
        source_identity = identity(path)
        inputs.append({"label": label, **source_identity})
        for index, sample_id in enumerate(sample_ids.tolist()):
            row = metrics_for_scene(
                predictions[index],
                targets[index],
                reference_predictions[index],
            )
            row.update(
                {
                    "variant": label,
                    "sample_index": index,
                    "sample_id": str(sample_id),
                    "source_npz_sha256": source_identity["sha256"],
                }
            )
            rows.append(row)

    write_csv(args.output_csv, rows)
    if args.frozen_test_ids is not None:
        args.frozen_test_ids.parent.mkdir(parents=True, exist_ok=True)
        with args.frozen_test_ids.open("w", encoding="utf-8", newline="\n") as handle:
            for sample_id in reference_ids.tolist():
                handle.write(str(sample_id) + "\n")
    summary = {
        "schema": "journal_scene_metrics_v1",
        "status": "passed",
        "reference_variant": reference_label,
        "inputs": inputs,
        "metric_definition": {
            "unit": "scene",
            "ignore_index": -1,
            "water_class": 1,
            "boundary_radius_pixels": 2,
            "empty_union": "NaN",
            "aggregate_rows": "global confusion/boundary counts; not mean of scene metrics",
        },
        "variants": {
            label: aggregate(row for row in rows if row["variant"] == label)
            for label, _ in variants
        },
        "outputs": {
            "per_scene_metrics": identity(args.output_csv),
            **(
                {"frozen_test_ids": identity(args.frozen_test_ids)}
                if args.frozen_test_ids is not None
                else {}
            ),
        },
    }
    write_json(args.summary_json, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
