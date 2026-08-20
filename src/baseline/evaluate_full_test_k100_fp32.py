"""Evaluate all 90 Sen1Floods11 test samples on Hygon K100 in strict FP32.

Run this script through ``run_full_test_k100_fp32.sh``.  It reuses the
audited checkpoint namespace migration from the one-sample parity test,
requires strict=True loading, evaluates every test sample exactly once, and
saves reproducible metrics and per-sample evidence.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(os.environ.get("PRITHVI_PROJECT_ROOT", "/workspace")).resolve()
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fp32_baseline_full/checkpoints/best-epoch46-step705.ckpt"
)
EXPECTED_TEST_SAMPLES = 90

# Manual strict-FP32 baseline from result/1/fp32_metrics.json.
STRICT_FP32_REFERENCE = {
    "loss": 0.09596340636583688,
    "miou": 0.8605868867283296,
    "mf1": 0.921782142424318,
    "background_iou": 0.9633356721481219,
    "water_iou": 0.7578381013085373,
    "background_f1": 0.981325491930902,
    "water_f1": 0.8622387929177341,
    "macro_class_accuracy": 0.9055010125180227,
    "background_accuracy": 0.9876484937971839,
    "water_accuracy": 0.8233535312388615,
    "pixel_accuracy": 0.9671095213675823,
    "valid_pixels": 3927398,
    "confusion_matrix": [[3393978, 42445], [86729, 404246]],
}

# Lightning/TerraTorch 1.2.8 run_summary.json; the manual baseline did not
# calculate Boundary mIoU.
BOUNDARY_REFERENCE = 0.32543644309043884
BOUNDARY_SOURCE = (
    "https://github.com/torchgeo/terratorch/blob/v1.2.8/"
    "terratorch/tasks/metrics.py"
)

sys.path.insert(0, str(PROJECT_ROOT))

from verify_real_sample_cpu_k100 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    migrate_and_strict_load,
    sha256_file,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--miou-tolerance-pp", type=float, default=0.1)
    parser.add_argument("--water-iou-tolerance-pp", type=float, default=0.2)
    parser.add_argument("--boundary-tolerance-pp", type=float, default=0.5)
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def mean_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return float(sum(available) / len(available)) if available else None


def confusion_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive

    class_iou = [
        safe_ratio(
            float(true_positive[index].item()),
            float((true_positive[index] + false_positive[index] + false_negative[index]).item()),
        )
        for index in range(2)
    ]
    class_f1 = [
        safe_ratio(
            float((2 * true_positive[index]).item()),
            float((2 * true_positive[index] + false_positive[index] + false_negative[index]).item()),
        )
        for index in range(2)
    ]
    class_accuracy = [
        safe_ratio(
            float(true_positive[index].item()),
            float(matrix[index].sum().item()),
        )
        for index in range(2)
    ]
    micro_intersection = float(true_positive.sum().item())
    micro_union = float(
        (true_positive.sum() + false_positive.sum() + false_negative.sum()).item()
    )
    return {
        "miou": mean_available(class_iou),
        "miou_micro": safe_ratio(micro_intersection, micro_union),
        "mf1": mean_available(class_f1),
        "background_iou": class_iou[0],
        "water_iou": class_iou[1],
        "background_f1": class_f1[0],
        "water_f1": class_f1[1],
        "macro_class_accuracy": mean_available(class_accuracy),
        "background_accuracy": class_accuracy[0],
        "water_accuracy": class_accuracy[1],
        "pixel_accuracy": safe_ratio(
            float(true_positive.sum().item()), float(matrix.sum().item())
        ),
        "confusion_matrix": confusion.tolist(),
    }


def single_sample_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    valid = (target >= 0) & (target < 2)
    encoded = target[valid].to(torch.int64) * 2 + prediction[valid].to(torch.int64)
    confusion = torch.bincount(encoded, minlength=4).reshape(2, 2).cpu()
    metrics = confusion_metrics(confusion)
    return {
        "valid_pixels": int(valid.sum().item()),
        "miou": metrics["miou"],
        "water_iou": metrics["water_iou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


@torch.no_grad()
def boundary_counts(
    predictions: torch.Tensor, targets: torch.Tensor, thickness: int = 2
) -> dict[str, int]:
    """Return TerraTorch-1.2.8-compatible and corrected water-boundary counts.

    TerraTorch 1.2.8 applies an (N,H,W) ignore mask directly to an
    (N,1,H,W) boundary tensor.  For N>1 PyTorch broadcasts this to
    (N,N,H,W).  The compatible counts reproduce that historical behavior;
    corrected counts explicitly insert the channel dimension.
    """

    kernel_size = 2 * thickness + 1
    ignore_mask = targets == -1
    prediction_mask = (predictions == 1).float().unsqueeze(1)
    target_mask = (targets == 1).float().unsqueeze(1)

    dilated_prediction = F.max_pool2d(
        prediction_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=thickness,
    )
    eroded_prediction = 1.0 - F.max_pool2d(
        1.0 - prediction_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=thickness,
    )
    prediction_boundary = (
        (dilated_prediction - eroded_prediction).clamp_min(0.0) > 0.5
    )

    dilated_target = F.max_pool2d(
        target_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=thickness,
    )
    eroded_target = 1.0 - F.max_pool2d(
        1.0 - target_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=thickness,
    )
    target_boundary = (dilated_target - eroded_target).clamp_min(0.0) > 0.5

    compatible_prediction = prediction_boundary & ~ignore_mask
    compatible_target = target_boundary & ~ignore_mask
    corrected_ignore = ignore_mask.unsqueeze(1)
    corrected_prediction = prediction_boundary & ~corrected_ignore
    corrected_target = target_boundary & ~corrected_ignore

    return {
        "compatible_intersection": int(
            (compatible_prediction & compatible_target).sum().item()
        ),
        "compatible_union": int(
            (compatible_prediction | compatible_target).sum().item()
        ),
        "corrected_intersection": int(
            (corrected_prediction & corrected_target).sum().item()
        ),
        "corrected_union": int(
            (corrected_prediction | corrected_target).sum().item()
        ),
    }


def finite_json(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    started = perf_counter()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("K100 is not available through torch.cuda")

    checkpoint_hash = None
    if not args.skip_checkpoint_hash:
        checkpoint_hash = sha256_file(checkpoint_path)
        if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Unexpected checkpoint SHA-256: {checkpoint_hash}; "
                f"expected {EXPECTED_CHECKPOINT_SHA256}"
            )

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, BATCH_SIZE, build_datamodule, build_task

    datamodule = build_datamodule()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    dataset_length = len(test_loader.dataset)
    if dataset_length != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, found {dataset_length}"
        )

    split_file = (
        PROJECT_ROOT
        / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.txt"
    )
    sample_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if len(sample_ids) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SAMPLES} test IDs, found {len(sample_ids)}"
        )
    sample_ids_sha256 = hashlib.sha256(
        ("\n".join(sample_ids) + "\n").encode("utf-8")
    ).hexdigest()

    build_started = perf_counter()
    task = build_task().float().eval().cpu()
    build_seconds = perf_counter() - build_started
    checkpoint_started = perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_read_seconds = perf_counter() - checkpoint_started
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    load_started = perf_counter()
    load_report = migrate_and_strict_load(task, checkpoint["state_dict"])
    strict_load_seconds = perf_counter() - load_started
    del checkpoint
    gc.collect()

    device = torch.device("cuda:0")
    task = task.to(device=device, dtype=torch.float32).eval()
    means = torch.tensor(
        [MEANS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    stds = torch.tensor(
        [STDS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)

    confusion = torch.zeros((2, 2), dtype=torch.int64)
    total_loss = 0.0
    valid_pixels = 0
    processed_samples = 0
    nonfinite_batches = 0
    batch_sizes: list[int] = []
    synchronized_forward_seconds = 0.0
    boundary_totals = {
        "compatible_intersection": 0,
        "compatible_union": 0,
        "corrected_intersection": 0,
        "corrected_union": 0,
    }
    per_sample_rows: list[dict[str, Any]] = []
    predictions_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    evaluation_started = perf_counter()

    with torch.inference_mode():
        for batch_index, batch in enumerate(test_loader):
            raw_images = batch["image"].detach().contiguous().float()
            targets_cpu = batch["mask"].detach().contiguous().long()
            if targets_cpu.ndim == 4 and targets_cpu.shape[1] == 1:
                targets_cpu = targets_cpu.squeeze(1)
            if raw_images.ndim != 4 or raw_images.shape[1:] != (6, 224, 224):
                raise ValueError(f"Unexpected image shape: {list(raw_images.shape)}")
            if targets_cpu.shape != (
                raw_images.shape[0],
                224,
                224,
            ):
                raise ValueError(f"Unexpected mask shape: {list(targets_cpu.shape)}")

            current_batch = int(raw_images.shape[0])
            batch_sizes.append(current_batch)
            images = raw_images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
            images = (images - means) / stds

            torch.cuda.synchronize()
            forward_started = perf_counter()
            with torch.autocast(device_type="cuda", enabled=False):
                logits = task(images).output
            torch.cuda.synchronize()
            synchronized_forward_seconds += perf_counter() - forward_started

            if logits.dtype != torch.float32:
                raise TypeError(f"Expected FP32 output, found {logits.dtype}")
            finite = bool(torch.isfinite(logits).all().item())
            nonfinite_batches += int(not finite)
            if not finite:
                raise RuntimeError(f"NaN or Inf found in batch {batch_index}")

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
            encoded = targets[valid].to(torch.int64) * 2 + predictions[valid].to(
                torch.int64
            )
            confusion += torch.bincount(encoded, minlength=4).reshape(2, 2).cpu()

            counts = boundary_counts(predictions, targets, thickness=2)
            for key, value in counts.items():
                boundary_totals[key] += value

            predictions_cpu = predictions.detach().to("cpu", dtype=torch.uint8)
            predictions_all.append(predictions_cpu)
            targets_all.append(targets_cpu.to(dtype=torch.int16))
            for local_index in range(current_batch):
                global_index = processed_samples + local_index
                sample_metrics = single_sample_metrics(
                    predictions_cpu[local_index].to(torch.int64),
                    targets_cpu[local_index],
                )
                per_sample_rows.append(
                    {
                        "sample_index": global_index,
                        "sample_id": sample_ids[global_index],
                        "raw_input_sha256": tensor_sha256(raw_images[local_index]),
                        "target_sha256": tensor_sha256(targets_cpu[local_index]),
                        **sample_metrics,
                    }
                )
            processed_samples += current_batch
            print(
                f"batch {batch_index + 1}/{len(test_loader)}: "
                f"processed {processed_samples}/{dataset_length}",
                flush=True,
            )

    torch.cuda.synchronize()
    evaluation_seconds = perf_counter() - evaluation_started
    if processed_samples != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Processed {processed_samples} samples, expected {EXPECTED_TEST_SAMPLES}; "
            "check test-loader drop_last/shuffle settings"
        )
    if len(per_sample_rows) != processed_samples:
        raise RuntimeError("Per-sample evidence count does not match processed samples")

    metrics = confusion_metrics(confusion)
    metrics["loss"] = float(total_loss / valid_pixels)
    metrics["valid_pixels"] = valid_pixels
    compatible_boundary = safe_ratio(
        boundary_totals["compatible_intersection"],
        boundary_totals["compatible_union"],
    )
    corrected_boundary = safe_ratio(
        boundary_totals["corrected_intersection"],
        boundary_totals["corrected_union"],
    )
    metrics["boundary_miou_terratorch_v1_2_8_compatible"] = compatible_boundary
    metrics["boundary_water_iou_corrected"] = corrected_boundary

    reference_deltas_pp = {
        key: (float(metrics[key]) - float(STRICT_FP32_REFERENCE[key])) * 100.0
        for key in (
            "miou",
            "mf1",
            "background_iou",
            "water_iou",
            "background_f1",
            "water_f1",
            "macro_class_accuracy",
            "background_accuracy",
            "water_accuracy",
            "pixel_accuracy",
        )
    }
    boundary_delta_pp = (
        (float(compatible_boundary) - BOUNDARY_REFERENCE) * 100.0
        if compatible_boundary is not None
        else None
    )
    sample_count_passed = processed_samples == EXPECTED_TEST_SAMPLES
    finite_passed = nonfinite_batches == 0
    miou_passed = abs(reference_deltas_pp["miou"]) <= args.miou_tolerance_pp
    water_iou_passed = (
        abs(reference_deltas_pp["water_iou"]) <= args.water_iou_tolerance_pp
    )
    boundary_passed = bool(
        boundary_delta_pp is not None
        and abs(boundary_delta_pp) <= args.boundary_tolerance_pp
    )
    passed = bool(
        sample_count_passed
        and finite_passed
        and miou_passed
        and water_iou_passed
        and boundary_passed
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "outputs/k100_full_test_fp32" / timestamp
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    predictions_path = output_dir / "predictions_and_targets.pt"
    torch.save(
        {
            "sample_ids": sample_ids,
            "predictions": torch.cat(predictions_all, dim=0),
            "targets": torch.cat(targets_all, dim=0),
            "bands": list(BANDS),
        },
        predictions_path,
    )
    per_sample_csv = output_dir / "per_sample_metrics.csv"
    csv_fields = [
        "sample_index",
        "sample_id",
        "raw_input_sha256",
        "target_sha256",
        "valid_pixels",
        "miou",
        "water_iou",
        "pixel_accuracy",
        "confusion_matrix",
    ]
    with per_sample_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in per_sample_rows:
            serializable = dict(row)
            serializable["confusion_matrix"] = json.dumps(
                serializable["confusion_matrix"], ensure_ascii=False
            )
            writer.writerow(serializable)

    result = {
        "status": "passed" if passed else "failed_acceptance",
        "node": "K100-1",
        "device_name": torch.cuda.get_device_name(0),
        "precision": "strict_fp32_no_autocast_no_tf32",
        "dataset": "Sen1Floods11 test",
        "dataset_length": dataset_length,
        "processed_samples": processed_samples,
        "sample_ids_sha256": sample_ids_sha256,
        "batch_size_configured": BATCH_SIZE,
        "observed_batch_sizes": batch_sizes,
        "input_shape_per_sample": [6, 224, 224],
        "bands": list(BANDS),
        "normalization_means": [float(value) for value in means.flatten().tolist()],
        "normalization_stds": [float(value) for value in stds.flatten().tolist()],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "state_dict_load": load_report,
        "total_parameters": sum(parameter.numel() for parameter in task.parameters()),
        "nonfinite_batch_count": nonfinite_batches,
        "metrics": metrics,
        "boundary_details": {
            "source": BOUNDARY_SOURCE,
            "thickness": 2,
            "include_background": False,
            "historical_v1_2_8_compatible_counts": {
                "intersection": boundary_totals["compatible_intersection"],
                "union": boundary_totals["compatible_union"],
            },
            "corrected_counts": {
                "intersection": boundary_totals["corrected_intersection"],
                "union": boundary_totals["corrected_union"],
            },
            "note": (
                "The compatible value reproduces TerraTorch 1.2.8 ignore-mask "
                "broadcasting for comparison with run_summary.json; the corrected "
                "value explicitly inserts the singleton channel dimension."
            ),
        },
        "reference": {
            "strict_fp32_manual_metrics": STRICT_FP32_REFERENCE,
            "boundary_miou_from_lightning_run_summary": BOUNDARY_REFERENCE,
        },
        "reference_deltas_percentage_points": {
            **reference_deltas_pp,
            "boundary_miou_terratorch_v1_2_8_compatible": boundary_delta_pp,
        },
        "acceptance": {
            "sample_count": {
                "passed": sample_count_passed,
                "expected": EXPECTED_TEST_SAMPLES,
                "actual": processed_samples,
            },
            "finite_outputs": {"passed": finite_passed},
            "miou": {
                "passed": miou_passed,
                "max_absolute_delta_pp": args.miou_tolerance_pp,
                "actual_delta_pp": reference_deltas_pp["miou"],
            },
            "water_iou": {
                "passed": water_iou_passed,
                "max_absolute_delta_pp": args.water_iou_tolerance_pp,
                "actual_delta_pp": reference_deltas_pp["water_iou"],
            },
            "boundary_miou_compatible": {
                "passed": boundary_passed,
                "max_absolute_delta_pp": args.boundary_tolerance_pp,
                "actual_delta_pp": boundary_delta_pp,
            },
        },
        "confusion_matrix_exact_match_to_reference": (
            metrics["confusion_matrix"] == STRICT_FP32_REFERENCE["confusion_matrix"]
        ),
        "valid_pixels_exact_match_to_reference": (
            valid_pixels == STRICT_FP32_REFERENCE["valid_pixels"]
        ),
        "timing_seconds": {
            "model_build": build_seconds,
            "checkpoint_read": checkpoint_read_seconds,
            "strict_load": strict_load_seconds,
            "full_evaluation_including_data_and_metrics": evaluation_seconds,
            "synchronized_forward_sum_diagnostic_only": synchronized_forward_seconds,
            "total": perf_counter() - started,
            "performance_claim_allowed": False,
        },
        "k100_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "k100_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "artifacts": {
            "predictions_and_targets": str(predictions_path),
            "per_sample_metrics_csv": str(per_sample_csv),
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("K100 STRICT FP32 FULL TEST EVALUATION")
    print(f"samples      : {processed_samples}/{dataset_length}")
    print(f"mIoU         : {metrics['miou']:.9f} ({reference_deltas_pp['miou']:+.6f} pp)")
    print(f"mF1          : {metrics['mf1']:.9f} ({reference_deltas_pp['mf1']:+.6f} pp)")
    print(
        f"Water IoU    : {metrics['water_iou']:.9f} "
        f"({reference_deltas_pp['water_iou']:+.6f} pp)"
    )
    print(
        f"Water F1     : {metrics['water_f1']:.9f} "
        f"({reference_deltas_pp['water_f1']:+.6f} pp)"
    )
    print(
        f"Boundary IoU : {compatible_boundary:.9f} compatible "
        f"({boundary_delta_pp:+.6f} pp)"
    )
    print(f"Boundary IoU : {corrected_boundary:.9f} corrected")
    print(f"result       : {result['status']}")
    print(f"result JSON  : {result_path}")
    print("=" * 78)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
