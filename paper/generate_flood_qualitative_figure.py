#!/usr/bin/env python3
"""Generate bilingual qualitative flood-segmentation figures from frozen evidence.

The three scenes are selected deterministically from non-empty ground-truth masks
at the 25th, 75th, and 95th percentiles of valid-pixel water coverage.  This
selection is independent of model accuracy.  The script verifies sample and
target identity across the FP32, full-FP16, and M5 prediction artifacts before
rendering the figure and writes a machine-readable provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
EVIDENCE_ROOT = HERE.parents[1]
DEFAULT_INPUTS = HERE / "qualitative_evidence" / "frozen_fp32_test90_inputs.npz"
DEFAULT_FP32 = (
    EVIDENCE_ROOT
    / "06_Phase11_FP32同协议基线"
    / "Phase11_FP32_25segment_headbarrier_20260818"
    / "output"
    / "predictions_and_targets.npz"
)
DEFAULT_FP16 = (
    EVIDENCE_ROOT
    / "05_MIGraphX环境诊断"
    / "Phase11_FP16_MIGraphX_compatibility_diagnostic_20260814"
    / "06_task_level_confirmation"
    / "predictions_and_targets.npz"
)
DEFAULT_M5 = HERE / "qualitative_evidence" / "m5_predictions_and_targets.npz"
DEFAULT_OUTPUT = HERE / "figures"

GT_COLOR = np.array([0x00, 0xB7, 0xC7], dtype=np.float32) / 255.0
TP_COLOR = np.array([0x2C, 0xA0, 0x2C], dtype=np.float32) / 255.0
FP_COLOR = np.array([0x1F, 0x77, 0xB4], dtype=np.float32) / 255.0
FN_COLOR = np.array([0xD6, 0x27, 0x28], dtype=np.float32) / 255.0
IGNORE_COLOR = np.array([0.82, 0.82, 0.82], dtype=np.float32)
SELECTION_QUANTILES = (0.25, 0.75, 0.95)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        missing = sorted(set(required) - set(pack.files))
        if missing:
            raise RuntimeError(f"{path} is missing arrays: {missing}")
        return {key: np.asarray(pack[key]) for key in required}


def validate_and_load(
    inputs_path: Path,
    fp32_path: Path,
    fp16_path: Path,
    m5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    inputs = load_npz(inputs_path, ("raw_inputs", "targets", "sample_ids"))
    models = {
        "FP32": load_npz(fp32_path, ("predictions", "targets", "sample_ids")),
        "FP16": load_npz(fp16_path, ("predictions", "targets", "sample_ids")),
        "M5": load_npz(m5_path, ("predictions", "targets", "sample_ids")),
    }

    raw = inputs["raw_inputs"]
    targets = inputs["targets"]
    sample_ids = inputs["sample_ids"]
    if raw.shape != (90, 6, 224, 224) or raw.dtype != np.float32:
        raise RuntimeError(f"unexpected raw input contract: {raw.shape}, {raw.dtype}")
    if targets.shape != (90, 224, 224):
        raise RuntimeError(f"unexpected target shape: {targets.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError("raw inputs contain NaN or Inf")

    predictions: dict[str, np.ndarray] = {}
    for label, pack in models.items():
        if pack["predictions"].shape != (90, 224, 224):
            raise RuntimeError(f"{label}: unexpected prediction shape")
        if not np.array_equal(pack["sample_ids"].astype(str), sample_ids.astype(str)):
            raise RuntimeError(f"{label}: sample ID identity mismatch")
        if not np.array_equal(pack["targets"], targets):
            raise RuntimeError(f"{label}: target identity mismatch")
        values = np.unique(pack["predictions"])
        if not set(values.tolist()).issubset({0, 1}):
            raise RuntimeError(f"{label}: predictions are not binary: {values}")
        predictions[label] = pack["predictions"].astype(np.uint8, copy=False)
    return raw, targets, sample_ids.astype(str), predictions


def coverage(target: np.ndarray) -> float:
    valid = (target >= 0) & (target < 2)
    if not valid.any():
        return float("nan")
    return float(np.mean(target[valid] == 1))


def select_indices(targets: np.ndarray) -> tuple[list[int], list[float]]:
    positive = [
        (index, coverage(target))
        for index, target in enumerate(targets)
        if np.isfinite(coverage(target)) and coverage(target) > 0.0
    ]
    values = np.array([value for _, value in positive], dtype=np.float64)
    selected: list[int] = []
    selected_coverage: list[float] = []
    for quantile in SELECTION_QUANTILES:
        target_value = float(np.quantile(values, quantile))
        index, value = min(positive, key=lambda row: (abs(row[1] - target_value), row[0]))
        if index in selected:
            raise RuntimeError("coverage quantiles selected a duplicate scene")
        selected.append(index)
        selected_coverage.append(value)
    return selected, selected_coverage


def true_color(raw: np.ndarray) -> np.ndarray:
    # Input order is BLUE, GREEN, RED, NIR, SWIR1, SWIR2.
    rgb = np.moveaxis(raw[[2, 1, 0]], 0, -1).astype(np.float32, copy=True)
    finite = np.isfinite(rgb)
    for channel in range(3):
        values = rgb[..., channel][finite[..., channel]]
        low, high = np.percentile(values, (2.0, 98.0))
        if not high > low:
            low, high = float(values.min()), float(values.max() + 1e-6)
        rgb[..., channel] = np.clip((rgb[..., channel] - low) / (high - low), 0.0, 1.0)
    return np.clip(rgb ** 0.82, 0.0, 1.0)


def muted_background(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gray = np.sum(rgb * np.array([0.2126, 0.7152, 0.0722]), axis=-1, keepdims=True)
    base = 0.56 * rgb + 0.44 * gray
    base = 0.70 * base + 0.12
    base[~valid] = IGNORE_COLOR
    return np.clip(base, 0.0, 1.0)


def blend(base: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    base[mask] = (1.0 - alpha) * base[mask] + alpha * color


def ground_truth_overlay(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = (target >= 0) & (target < 2)
    output = muted_background(rgb, valid)
    blend(output, valid & (target == 1), GT_COLOR, 0.80)
    return output


def error_overlay(rgb: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    valid = (target >= 0) & (target < 2)
    output = muted_background(rgb, valid)
    true_positive = valid & (target == 1) & (prediction == 1)
    false_positive = valid & (target == 0) & (prediction == 1)
    false_negative = valid & (target == 1) & (prediction == 0)
    blend(output, true_positive, TP_COLOR, 0.88)
    blend(output, false_positive, FP_COLOR, 0.92)
    blend(output, false_negative, FN_COLOR, 0.92)
    return output


def sample_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    valid = (target >= 0) & (target < 2)
    tp = int(np.sum(valid & (target == 1) & (prediction == 1)))
    fp = int(np.sum(valid & (target == 0) & (prediction == 1)))
    fn = int(np.sum(valid & (target == 1) & (prediction == 0)))
    denominator = tp + fp + fn
    return {
        "water_iou": float(tp / denominator) if denominator else 1.0,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
    }


def render(
    language: str,
    raw: np.ndarray,
    targets: np.ndarray,
    sample_ids: np.ndarray,
    predictions: dict[str, np.ndarray],
    indices: list[int],
    selected_coverage: list[float],
    output_dir: Path,
) -> tuple[Path, Path]:
    if language == "en":
        row_labels = ("Ground truth", "FP32", "Full FP16", "M5 hybrid")
        coverage_label = "water"
        legend_labels = ("Ground-truth water", "True positive", "False positive", "False negative")
        font_family = "DejaVu Sans"
        stem = "fig_flood_qualitative_en"
    elif language == "cn":
        row_labels = ("真值", "FP32", "完整 FP16", "M5 混合精度")
        coverage_label = "水体占比"
        legend_labels = ("真值水体", "正确检出", "误检", "漏检")
        font_family = "Microsoft YaHei"
        stem = "fig_flood_qualitative_cn"
    else:
        raise ValueError(language)

    with mpl.rc_context(
        {
            "font.family": font_family,
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        # A compact single-column IEEE layout: scenes are rows and execution
        # paths are columns.  Keeping all four paths beside each other makes the
        # qualitative comparison readable at a 3.5-inch column width.
        fig, axes = plt.subplots(3, 4, figsize=(3.50, 2.96), dpi=220)
        figure_columns = (None, "FP32", "FP16", "M5")
        for row, (index, water_fraction) in enumerate(zip(indices, selected_coverage)):
            rgb = true_color(raw[index])
            target = targets[index]
            for column, model_label in enumerate(figure_columns):
                axis = axes[row, column]
                image = (
                    ground_truth_overlay(rgb, target)
                    if model_label is None
                    else error_overlay(rgb, target, predictions[model_label][index])
                )
                axis.imshow(image, interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#D4D4D4")
                    spine.set_linewidth(0.45)
                if row == 0:
                    axis.set_title(row_labels[column], pad=2.0, color="#222222", fontsize=6.4)
                if column == 0:
                    scene_id = str(sample_ids[index]).replace("_", "\n", 1)
                    axis.set_ylabel(
                        f"{scene_id}\n{coverage_label} {100.0 * water_fraction:.2f}%",
                        labelpad=3.0,
                        color="#222222",
                        fontsize=5.3,
                    )

        legend = [
            Patch(facecolor=GT_COLOR, edgecolor="none", label=legend_labels[0]),
            Patch(facecolor=TP_COLOR, edgecolor="none", label=legend_labels[1]),
            Patch(facecolor=FP_COLOR, edgecolor="none", label=legend_labels[2]),
            Patch(facecolor=FN_COLOR, edgecolor="none", label=legend_labels[3]),
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.59, 0.004),
            columnspacing=0.9,
            handlelength=1.0,
            handleheight=0.65,
            fontsize=5.4,
        )
        fig.subplots_adjust(left=0.205, right=0.995, top=0.915, bottom=0.145, wspace=0.025, hspace=0.035)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        # Crop the asymmetric outer canvas introduced by the rotated row labels.
        # Without this crop LaTeX centers the PDF media box, while the visible
        # panels appear shifted to the right inside a single IEEE column.
        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.015,
            "facecolor": "white",
        }
        fig.savefig(pdf_path, format="pdf", **save_kwargs)
        fig.savefig(png_path, format="png", dpi=400, **save_kwargs)
        plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--fp32", type=Path, default=DEFAULT_FP32)
    parser.add_argument("--fp16", type=Path, default=DEFAULT_FP16)
    parser.add_argument("--m5", type=Path, default=DEFAULT_M5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, targets, sample_ids, predictions = validate_and_load(
        args.inputs, args.fp32, args.fp16, args.m5
    )
    indices, selected_coverage = select_indices(targets)
    outputs: dict[str, dict[str, object]] = {}
    for language in ("en", "cn"):
        pdf_path, png_path = render(
            language,
            raw,
            targets,
            sample_ids,
            predictions,
            indices,
            selected_coverage,
            args.output_dir,
        )
        outputs[language] = {"pdf": identity(pdf_path), "png": identity(png_path)}

    sources = {
        "frozen_inputs": identity(args.inputs),
        "fp32_predictions": identity(args.fp32),
        "fp16_predictions": identity(args.fp16),
        "m5_predictions": identity(args.m5),
    }
    selection = []
    for quantile, index, water_fraction in zip(SELECTION_QUANTILES, indices, selected_coverage):
        row = {
            "target_quantile_among_nonempty_masks": quantile,
            "test_index": index,
            "sample_id": str(sample_ids[index]),
            "valid_pixel_water_fraction": water_fraction,
            "models": {},
        }
        for label in ("FP32", "FP16", "M5"):
            row["models"][label] = sample_metrics(targets[index], predictions[label][index])
        selection.append(row)

    manifest = {
        "schema": "k100_flood_qualitative_figure_v1",
        "selection_policy": {
            "population": "test scenes with at least one valid ground-truth water pixel",
            "criterion": "nearest valid-pixel water-coverage quantile; smallest index breaks ties",
            "quantiles": list(SELECTION_QUANTILES),
            "independent_of_model_accuracy": True,
        },
        "visualization": {
            "rgb_bands": ["RED", "GREEN", "BLUE"],
            "rgb_source_indices_in_six_band_input": [2, 1, 0],
            "rgb_stretch": "per-scene per-channel 2nd--98th percentile with gamma 0.82",
            "prediction_overlay": {
                "true_positive": "green",
                "false_positive": "blue",
                "false_negative": "red",
            },
        },
        "sources": sources,
        "selected_scenes": selection,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "fig_flood_qualitative_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": identity(manifest_path), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
