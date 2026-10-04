'Research implementation: evaluate fp32 baseline.'

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

# Set OpenMP variables before importing PyTorch-dependent packages.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from terratorch.datamodules.sen1floods11 import MEANS, STDS

from train_fp32_baseline import (
    BANDS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    RUN_ROOT,
    SEED,
    build_datamodule,
    build_task,
    validate_inputs,
)


RESULTS_DIR = PROJECT_ROOT / "results/fp32_baseline"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_FILE = RESULTS_DIR / "fp32_metrics.json"
CSV_FILE = RESULTS_DIR / "fp32_metrics.csv"
NUM_PREDICTION_SAMPLES = 4


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_best_checkpoint() -> Path:
    summary_file = RUN_ROOT / "run_summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        candidate = Path(summary.get("best_checkpoint", ""))
        if candidate.is_file():
            return candidate

    candidates = list(CHECKPOINT_DIR.glob("best*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found in {CHECKPOINT_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(numerator, torch.nan, dtype=torch.float64)
    valid = denominator != 0
    result[valid] = numerator[valid].double() / denominator[valid].double()
    return result


def stretch_rgb(image_chw: np.ndarray) -> np.ndarray:
    rgb = np.transpose(image_chw[[2, 1, 0]], (1, 2, 0)).astype(np.float32)
    low = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = (rgb - low) / np.maximum(high - low, 1e-6)
    return np.clip(rgb, 0, 1)


def save_prediction(index: int, raw_image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> None:
    rgb = stretch_rgb(raw_image.cpu().numpy())
    target_np = target.cpu().numpy()
    prediction_np = prediction.cpu().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(target_np, vmin=-1, vmax=1, cmap="viridis")
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction_np, vmin=0, vmax=1, cmap="viridis")
    axes[2].set_title("FP32 prediction")
    for axis in axes:
        axis.axis("off")

    output_file = PREDICTIONS_DIR / f"sample_{index:02d}.png"
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def main() -> None:
    validate_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this evaluation in AutoDL GPU mode.")

    # Strict FP32 reference: no autocast and no TF32 tensor-core approximation.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    checkpoint_path = find_best_checkpoint()
    print(f"Best checkpoint: {checkpoint_path}")

    task = build_task()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    task = task.float().to(device).eval()

    datamodule = build_datamodule()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    means = torch.tensor([MEANS[band] for band in BANDS], device=device).view(1, 6, 1, 1)
    stds = torch.tensor([STDS[band] for band in BANDS], device=device).view(1, 6, 1, 1)

    confusion = torch.zeros((2, 2), dtype=torch.int64)
    total_loss = 0.0
    valid_pixels = 0
    prediction_samples_saved = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    evaluation_start = perf_counter()

    with torch.inference_mode():
        for batch_index, batch in enumerate(test_loader):
            raw_images = batch["image"]
            targets_cpu = batch["mask"]

            images = raw_images.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)
            images = (images - means) / stds

            # Intentionally no torch.autocast context here.
            logits = task(images).output
            if logits.dtype != torch.float32:
                raise TypeError(f"Expected FP32 output, found {logits.dtype}")

            predictions = logits.argmax(dim=1)
            valid = (targets >= 0) & (targets < 2)
            count = int(valid.sum().item())
            valid_pixels += count
            total_loss += float(
                F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=-1,
                    reduction="sum",
                ).item()
            )

            encoded = targets[valid].to(torch.int64) * 2 + predictions[valid].to(torch.int64)
            confusion += torch.bincount(encoded, minlength=4).reshape(2, 2).cpu()

            while (
                prediction_samples_saved < NUM_PREDICTION_SAMPLES
                and prediction_samples_saved < raw_images.shape[0]
            ):
                sample_index = prediction_samples_saved
                save_prediction(
                    prediction_samples_saved,
                    raw_images[sample_index],
                    targets_cpu[sample_index],
                    predictions[sample_index].cpu(),
                )
                prediction_samples_saved += 1

            print(f"Evaluated batch {batch_index + 1}/{len(test_loader)}")

    torch.cuda.synchronize()
    elapsed_seconds = perf_counter() - evaluation_start
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    confusion_fp = confusion.to(torch.float64)
    true_positive = confusion_fp.diag()
    false_positive = confusion_fp.sum(dim=0) - true_positive
    false_negative = confusion_fp.sum(dim=1) - true_positive

    class_iou = safe_divide(
        true_positive,
        true_positive + false_positive + false_negative,
    )
    class_f1 = safe_divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    class_accuracy = safe_divide(true_positive, confusion_fp.sum(dim=1))

    pixel_accuracy = float(true_positive.sum() / confusion_fp.sum())
    metrics = {
        "variant": "pytorch",
        "precision": "fp32_strict",
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_size_mb": checkpoint_path.stat().st_size / 1024**2,
        "batch_size": BATCH_SIZE,
        "num_test_samples": len(datamodule.test_dataset),
        "valid_pixels": valid_pixels,
        "loss": total_loss / valid_pixels,
        "miou": float(torch.nanmean(class_iou)),
        "mf1": float(torch.nanmean(class_f1)),
        "background_iou": float(class_iou[0]),
        "water_iou": float(class_iou[1]),
        "background_f1": float(class_f1[0]),
        "water_f1": float(class_f1[1]),
        "macro_class_accuracy": float(torch.nanmean(class_accuracy)),
        "background_accuracy": float(class_accuracy[0]),
        "water_accuracy": float(class_accuracy[1]),
        "pixel_accuracy": pixel_accuracy,
        "confusion_matrix": confusion.tolist(),
        "evaluation_seconds": elapsed_seconds,
        "eval_peak_memory_batch16_mb": peak_memory_mb,
        "tf32_enabled": False,
        "autocast_enabled": False,
        "input_shape": [6, 224, 224],
    }

    METRICS_FILE.write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "variant",
        "precision",
        "checkpoint_size_mb",
        "miou",
        "mf1",
        "water_iou",
        "water_f1",
        "pixel_accuracy",
        "loss",
        "evaluation_seconds",
        "eval_peak_memory_batch16_mb",
    ]
    with CSV_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({field: metrics[field] for field in csv_fields})

    print("=" * 72)
    print("STRICT FP32 EVALUATION COMPLETE")
    print(f"mIoU:      {metrics['miou']:.6f}")
    print(f"mF1:       {metrics['mf1']:.6f}")
    print(f"Water IoU: {metrics['water_iou']:.6f}")
    print(f"Water F1:  {metrics['water_f1']:.6f}")
    print(f"Metrics:   {METRICS_FILE}")
    print(f"Samples:   {PREDICTIONS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
