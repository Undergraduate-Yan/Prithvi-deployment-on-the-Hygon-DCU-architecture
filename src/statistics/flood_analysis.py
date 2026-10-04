"""Flood metrics and paired scene-bootstrap used for the descriptive public-pool analysis."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

VARIANTS = [
    ("Mono-FP32", "Mono_FP32"),
    ("RCS13-FP32", "RCS13_FP32"),
    ("RCS13-FP16-Opt", "RCS13_FP16_OPT_retry1"),
    ("Mono-FP16-Opt", "Mono_FP16_OPT"),
    ("MP-RCS-Opt", "MP_RCS_OPT"),
    ("Legacy-M5-25S", "Legacy_M5_25S"),
]

REFERENCE = "RCS13-FP32"

METRICS = ["miou", "water_iou", "boundary_water_iou", "pixel_accuracy", "agreement"]

BOOTSTRAP_SEED = 42

BOOTSTRAP_RESAMPLES = 10_000

def dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            result |= padded[y : y + height, x : x + width]
    return result

def boundary(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(bool)
    return dilate(value) & ~(~dilate(~value))

def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")

def scene_metrics(prediction: np.ndarray, target: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    valid = (target >= 0) & (target < 2)
    encoded = target[valid] * 2 + prediction[valid]
    matrix = np.bincount(encoded, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    background_iou = safe_ratio(tn, tn + fp + fn)
    water_iou = safe_ratio(tp, tp + fp + fn)
    defined = [value for value in (background_iou, water_iou) if math.isfinite(value)]
    pred_boundary = boundary(prediction == 1)
    target_boundary = boundary(target == 1)
    boundary_intersection = int((pred_boundary & target_boundary & valid).sum())
    boundary_union = int(((pred_boundary | target_boundary) & valid).sum())
    valid_pixels = int(valid.sum())
    changed = int(((prediction != reference) & valid).sum())
    water_pixels = int(((target == 1) & valid).sum())
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "valid_pixels": valid_pixels,
        "water_pixels": water_pixels,
        "water_coverage": safe_ratio(water_pixels, valid_pixels),
        "background_iou": background_iou,
        "water_iou": water_iou,
        "miou": float(np.mean(defined)) if defined else float("nan"),
        "boundary_intersection": boundary_intersection,
        "boundary_union": boundary_union,
        "boundary_water_iou": safe_ratio(boundary_intersection, boundary_union),
        "pixel_accuracy": safe_ratio(tn + tp, valid_pixels),
        "agreement": 1.0 - safe_ratio(changed, valid_pixels) if valid_pixels else float("nan"),
        "changed_valid_pixels": changed,
    }

def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {key: sum(int(row[key]) for row in rows) for key in ("tn", "fp", "fn", "tp")}
    valid = sum(int(row["valid_pixels"]) for row in rows)
    changed = sum(int(row["changed_valid_pixels"]) for row in rows)
    boundary_intersection = sum(int(row["boundary_intersection"]) for row in rows)
    boundary_union = sum(int(row["boundary_union"]) for row in rows)
    background_iou = safe_ratio(totals["tn"], totals["tn"] + totals["fp"] + totals["fn"])
    water_iou = safe_ratio(totals["tp"], totals["tp"] + totals["fp"] + totals["fn"])
    return {
        "scene_count": len(rows),
        "valid_pixels": valid,
        "miou": float(np.nanmean([background_iou, water_iou])),
        "water_iou": water_iou,
        "boundary_water_iou": safe_ratio(boundary_intersection, boundary_union),
        "pixel_accuracy": safe_ratio(totals["tn"] + totals["tp"], valid),
        "agreement": 1.0 - changed / valid,
        "changed_valid_pixels": changed,
        "confusion_matrix": [[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]],
    }

def bootstrap_rows(per_variant: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reference = per_variant[REFERENCE]
    output = []
    for label, _ in VARIANTS:
        if label == REFERENCE:
            continue
        for metric in METRICS:
            values = np.asarray([row[metric] - ref[metric] for row, ref in zip(per_variant[label], reference)], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                raise RuntimeError(f"no finite paired scenes for {label}/{metric}")
            estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
            chunk = 500
            for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
                count = min(chunk, BOOTSTRAP_RESAMPLES - start)
                indexes = rng.integers(0, len(values), size=(count, len(values)))
                estimates[start : start + count] = values[indexes].mean(axis=1)
            low, high = np.percentile(estimates, [2.5, 97.5])
            direction = "positive" if low > 0 else "negative" if high < 0 else "not_significant"
            output.append({
                "variant": label,
                "reference": REFERENCE,
                "metric": metric,
                "estimand": "mean paired scene-level difference (variant - reference)",
                "point_estimate": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "finite_paired_scenes": int(len(values)),
                "excluded_nan_scenes": int(90 - len(values)),
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "significant_direction": direction,
            })
    return output
