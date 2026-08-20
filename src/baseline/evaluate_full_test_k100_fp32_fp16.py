"""Evaluate strict native FP32 and FP16 on all 90 Sen1Floods11 samples."""

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
REFERENCE_ROOT = PROJECT_ROOT / "outputs/k100_full_test_fp32/20260807-171433"
REFERENCE_RESULT = REFERENCE_ROOT / "result.json"
REFERENCE_CSV = REFERENCE_ROOT / "per_sample_metrics.csv"
REFERENCE_TENSORS = REFERENCE_ROOT / "predictions_and_targets.pt"
REFERENCE_SHA256 = {
    "result.json": "24aaff096c148d2fe4ad296df26b08145a4571fe57d3ddbdd7e42ec9b9e9eef8",
    "per_sample_metrics.csv": "3ebe949de7765e13fdf1fc4134f89b41f5f61dec1bf85bbe8af0ba6cee49ce85",
    "predictions_and_targets.pt": "6c74dd3c3d89f0ae8a1fe7acb42b996f8e5770973d8523f82b0a9b8758d5622c",
}
EXPECTED_SAMPLES = 90

sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_full_test_k100_fp32 import (  # noqa: E402
    boundary_counts,
    confusion_metrics,
    single_sample_metrics,
)
from verify_real_sample_cpu_k100 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    migrate_and_strict_load,
    sha256_file,
    tensor_sha256,
)
from verify_real_sample_fp32_fp16_k100 import floating_dtype_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--miou-max-drop-pp", type=float, default=0.5)
    parser.add_argument("--water-iou-max-drop-pp", type=float, default=1.0)
    parser.add_argument("--boundary-max-drop-pp", type=float, default=1.0)
    parser.add_argument("--min-valid-pixel-agreement", type=float, default=0.995)
    return parser.parse_args()


def read_reference_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def summarize_metrics(
    confusion: torch.Tensor,
    loss_sum: float,
    valid_pixels: int,
    boundary_totals: dict[str, int],
) -> dict[str, Any]:
    result = confusion_metrics(confusion)
    result["loss"] = float(loss_sum / valid_pixels)
    result["valid_pixels"] = valid_pixels
    result["boundary_miou_terratorch_v1_2_8_compatible"] = safe_ratio(
        boundary_totals["compatible_intersection"],
        boundary_totals["compatible_union"],
    )
    result["boundary_water_iou_corrected"] = safe_ratio(
        boundary_totals["corrected_intersection"],
        boundary_totals["corrected_union"],
    )
    return result


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

    if not torch.cuda.is_available():
        raise RuntimeError("K100 is not available through torch.cuda")
    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Unexpected checkpoint SHA-256: {checkpoint_hash}")

    reference_paths = {
        "result.json": REFERENCE_RESULT.resolve(strict=True),
        "per_sample_metrics.csv": REFERENCE_CSV.resolve(strict=True),
        "predictions_and_targets.pt": REFERENCE_TENSORS.resolve(strict=True),
    }
    reference_hashes = {name: sha256_file(path) for name, path in reference_paths.items()}
    if reference_hashes != REFERENCE_SHA256:
        raise RuntimeError(
            f"Historical FP32 artifact identity mismatch: {reference_hashes}"
        )
    reference_result = json.loads(REFERENCE_RESULT.read_text(encoding="utf-8"))
    reference_rows = read_reference_rows(REFERENCE_CSV)
    reference_tensors = torch.load(
        REFERENCE_TENSORS, map_location="cpu", weights_only=False
    )

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, BATCH_SIZE, build_datamodule, build_task

    datamodule = build_datamodule()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    if len(test_loader.dataset) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLES} samples")

    split_path = (
        PROJECT_ROOT
        / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.txt"
    )
    sample_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(sample_ids) != EXPECTED_SAMPLES:
        raise RuntimeError("Test ID count mismatch")
    if sample_ids != list(reference_tensors["sample_ids"]):
        raise RuntimeError("Test sample IDs do not match the frozen FP32 artifact")
    if len(reference_rows) != EXPECTED_SAMPLES:
        raise RuntimeError("Frozen FP32 CSV row count mismatch")

    build_started = perf_counter()
    task_fp32_cpu = build_task().float().eval().cpu()
    task_fp16_cpu = build_task().float().eval().cpu()
    build_seconds = perf_counter() - build_started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    load_fp32 = migrate_and_strict_load(task_fp32_cpu, checkpoint["state_dict"])
    load_fp16 = migrate_and_strict_load(task_fp16_cpu, checkpoint["state_dict"])
    del checkpoint

    device = torch.device("cuda:0")
    task_fp32 = task_fp32_cpu.to(device=device, dtype=torch.float32).eval()
    task_fp16 = task_fp16_cpu.to(device=device, dtype=torch.float16).eval()
    del task_fp32_cpu, task_fp16_cpu
    gc.collect()
    dtype_fp32 = floating_dtype_audit(task_fp32, torch.float32)
    dtype_fp16 = floating_dtype_audit(task_fp16, torch.float16)
    if not dtype_fp32["passed"] or not dtype_fp16["passed"]:
        raise RuntimeError("Model dtype audit failed")

    means = torch.tensor(
        [MEANS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    stds = torch.tensor(
        [STDS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)

    confusions = {
        "fp32": torch.zeros((2, 2), dtype=torch.int64),
        "fp16": torch.zeros((2, 2), dtype=torch.int64),
    }
    loss_sums = {"fp32": 0.0, "fp16": 0.0}
    boundary_totals = {
        precision: {
            "compatible_intersection": 0,
            "compatible_union": 0,
            "corrected_intersection": 0,
            "corrected_union": 0,
        }
        for precision in ("fp32", "fp16")
    }
    nonfinite_batches = {"fp32": 0, "fp16": 0}
    forward_seconds = {"fp32": 0.0, "fp16": 0.0}
    observed_batch_sizes: list[int] = []
    processed = 0
    valid_pixels = 0
    changed_valid_pixels = 0
    compared_valid_pixels = 0
    per_sample_rows: list[dict[str, Any]] = []
    predictions = {"fp32": [], "fp16": []}
    targets_all: list[torch.Tensor] = []
    logit_differences: list[torch.Tensor] = []
    input_identity_mismatches: list[dict[str, Any]] = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    evaluation_started = perf_counter()

    with torch.inference_mode():
        for batch_index, batch in enumerate(test_loader):
            raw_cpu = batch["image"].detach().contiguous().float()
            targets_cpu = batch["mask"].detach().contiguous().long()
            if targets_cpu.ndim == 4 and targets_cpu.shape[1] == 1:
                targets_cpu = targets_cpu.squeeze(1)
            if raw_cpu.ndim != 4 or raw_cpu.shape[1:] != (6, 224, 224):
                raise ValueError(f"Unexpected image shape {list(raw_cpu.shape)}")
            if targets_cpu.shape != (raw_cpu.shape[0], 224, 224):
                raise ValueError(f"Unexpected mask shape {list(targets_cpu.shape)}")

            batch_size = int(raw_cpu.shape[0])
            observed_batch_sizes.append(batch_size)
            images_fp32 = raw_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            images_fp32 = ((images_fp32 - means) / stds).contiguous()
            images_fp16 = images_fp32.to(dtype=torch.float16)
            targets = targets_cpu.to(device=device, dtype=torch.int64, non_blocking=True)

            torch.cuda.synchronize()
            fp32_started = perf_counter()
            with torch.autocast(device_type="cuda", enabled=False):
                logits_fp32 = task_fp32(images_fp32).output
            torch.cuda.synchronize()
            forward_seconds["fp32"] += perf_counter() - fp32_started

            torch.cuda.synchronize()
            fp16_started = perf_counter()
            with torch.autocast(device_type="cuda", enabled=False):
                logits_fp16_native = task_fp16(images_fp16).output
            torch.cuda.synchronize()
            forward_seconds["fp16"] += perf_counter() - fp16_started

            if logits_fp32.dtype != torch.float32:
                raise TypeError(f"Expected FP32 logits, found {logits_fp32.dtype}")
            if logits_fp16_native.dtype != torch.float16:
                raise TypeError(f"Expected FP16 logits, found {logits_fp16_native.dtype}")
            if logits_fp32.shape != (batch_size, 2, 224, 224):
                raise ValueError(f"Unexpected FP32 logits shape {list(logits_fp32.shape)}")
            if logits_fp16_native.shape != logits_fp32.shape:
                raise ValueError("FP16 logits shape mismatch")

            finite_fp32 = bool(torch.isfinite(logits_fp32).all().item())
            finite_fp16 = bool(torch.isfinite(logits_fp16_native).all().item())
            nonfinite_batches["fp32"] += int(not finite_fp32)
            nonfinite_batches["fp16"] += int(not finite_fp16)
            if not finite_fp32 or not finite_fp16:
                raise RuntimeError(f"Non-finite output in batch {batch_index}")

            logits_fp16 = logits_fp16_native.float()
            predictions_device = {
                "fp32": logits_fp32.argmax(dim=1),
                "fp16": logits_fp16.argmax(dim=1),
            }
            valid = (targets >= 0) & (targets < 2)
            batch_valid = int(valid.sum().item())
            valid_pixels += batch_valid
            compared_valid_pixels += batch_valid
            changed_valid_pixels += int(
                (
                    predictions_device["fp32"][valid]
                    != predictions_device["fp16"][valid]
                ).sum().item()
            )

            for precision, logits in (("fp32", logits_fp32), ("fp16", logits_fp16)):
                prediction = predictions_device[precision]
                loss_sums[precision] += float(
                    F.cross_entropy(
                        logits.float(), targets, ignore_index=-1, reduction="sum"
                    ).item()
                )
                encoded = targets[valid] * 2 + prediction[valid]
                confusions[precision] += torch.bincount(
                    encoded.to(torch.int64), minlength=4
                ).reshape(2, 2).cpu()
                counts = boundary_counts(prediction, targets, thickness=2)
                for key, value in counts.items():
                    boundary_totals[precision][key] += value

            difference = (logits_fp16 - logits_fp32).abs()
            logit_differences.append(difference.detach().flatten().cpu())
            pred_cpu = {
                key: value.detach().to("cpu", dtype=torch.uint8)
                for key, value in predictions_device.items()
            }
            predictions["fp32"].append(pred_cpu["fp32"])
            predictions["fp16"].append(pred_cpu["fp16"])
            targets_all.append(targets_cpu.to(torch.int16))

            for local_index in range(batch_size):
                global_index = processed + local_index
                raw_hash = tensor_sha256(raw_cpu[local_index])
                target_hash = tensor_sha256(targets_cpu[local_index])
                reference_row = reference_rows[global_index]
                identity_ok = bool(
                    reference_row["sample_index"] == str(global_index)
                    and reference_row["sample_id"] == sample_ids[global_index]
                    and reference_row["raw_input_sha256"] == raw_hash
                    and reference_row["target_sha256"] == target_hash
                )
                if not identity_ok:
                    input_identity_mismatches.append(
                        {"sample_index": global_index, "sample_id": sample_ids[global_index]}
                    )
                sample_valid = (targets_cpu[local_index] >= 0) & (
                    targets_cpu[local_index] < 2
                )
                sample_changed = int(
                    (
                        pred_cpu["fp32"][local_index][sample_valid]
                        != pred_cpu["fp16"][local_index][sample_valid]
                    ).sum().item()
                )
                sample_valid_count = int(sample_valid.sum().item())
                sample_difference = difference[local_index].detach().float().cpu()
                fp32_sample = single_sample_metrics(
                    pred_cpu["fp32"][local_index].to(torch.int64),
                    targets_cpu[local_index],
                )
                fp16_sample = single_sample_metrics(
                    pred_cpu["fp16"][local_index].to(torch.int64),
                    targets_cpu[local_index],
                )
                per_sample_rows.append(
                    {
                        "sample_index": global_index,
                        "sample_id": sample_ids[global_index],
                        "raw_input_sha256": raw_hash,
                        "target_sha256": target_hash,
                        "identity_matches_fp32_reference": identity_ok,
                        "valid_pixels": sample_valid_count,
                        "changed_valid_pixels": sample_changed,
                        "valid_pixel_agreement": (
                            1.0 - sample_changed / sample_valid_count
                            if sample_valid_count
                            else None
                        ),
                        "fp32_miou": fp32_sample["miou"],
                        "fp16_miou": fp16_sample["miou"],
                        "fp32_water_iou": fp32_sample["water_iou"],
                        "fp16_water_iou": fp16_sample["water_iou"],
                        "fp32_pixel_accuracy": fp32_sample["pixel_accuracy"],
                        "fp16_pixel_accuracy": fp16_sample["pixel_accuracy"],
                        "logit_mean_abs_error": float(sample_difference.mean().item()),
                        "logit_max_abs_error": float(sample_difference.max().item()),
                        "logit_rmse": float(
                            torch.sqrt(torch.mean(sample_difference.square())).item()
                        ),
                    }
                )
            processed += batch_size
            print(
                f"batch {batch_index + 1}/{len(test_loader)}: "
                f"processed {processed}/{EXPECTED_SAMPLES}",
                flush=True,
            )

    torch.cuda.synchronize()
    evaluation_seconds = perf_counter() - evaluation_started
    if processed != EXPECTED_SAMPLES or len(per_sample_rows) != EXPECTED_SAMPLES:
        raise RuntimeError("Full-test sample count mismatch")

    prediction_tensors = {
        precision: torch.cat(values, dim=0) for precision, values in predictions.items()
    }
    targets_tensor = torch.cat(targets_all, dim=0)
    fp32_prediction_exact = bool(
        torch.equal(prediction_tensors["fp32"], reference_tensors["predictions"])
    )
    targets_exact = bool(torch.equal(targets_tensor, reference_tensors["targets"]))
    metrics = {
        precision: summarize_metrics(
            confusions[precision],
            loss_sums[precision],
            valid_pixels,
            boundary_totals[precision],
        )
        for precision in ("fp32", "fp16")
    }
    deltas_pp = {
        key: (float(metrics["fp16"][key]) - float(metrics["fp32"][key])) * 100.0
        for key in (
            "miou",
            "mf1",
            "water_iou",
            "water_f1",
            "pixel_accuracy",
            "boundary_water_iou_corrected",
        )
    }

    all_differences = torch.cat(logit_differences)
    logit_error = {
        "count": int(all_differences.numel()),
        "mean_abs_error": float(all_differences.mean().item()),
        "max_abs_error": float(all_differences.max().item()),
        "rmse": float(torch.sqrt(torch.mean(all_differences.square())).item()),
        "p95_abs_error": float(torch.quantile(all_differences, 0.95).item()),
        "p99_abs_error": float(torch.quantile(all_differences, 0.99).item()),
    }
    valid_agreement = 1.0 - changed_valid_pixels / compared_valid_pixels

    acceptance = {
        "sample_count": processed == EXPECTED_SAMPLES,
        "finite_outputs": nonfinite_batches == {"fp32": 0, "fp16": 0},
        "input_identity": not input_identity_mismatches,
        "targets_exact_to_frozen_fp32": targets_exact,
        "fp32_predictions_exact_to_frozen_fp32": fp32_prediction_exact,
        "fp32_confusion_exact_to_frozen_fp32": (
            metrics["fp32"]["confusion_matrix"]
            == reference_result["metrics"]["confusion_matrix"]
        ),
        "fp16_miou_drop": deltas_pp["miou"] >= -args.miou_max_drop_pp,
        "fp16_water_iou_drop": (
            deltas_pp["water_iou"] >= -args.water_iou_max_drop_pp
        ),
        "fp16_corrected_boundary_drop": (
            deltas_pp["boundary_water_iou_corrected"] >= -args.boundary_max_drop_pp
        ),
        "fp16_valid_pixel_agreement": valid_agreement >= args.min_valid_pixel_agreement,
    }
    passed = all(acceptance.values())

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "outputs/k100_full_test_fp32_fp16" / timestamp
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    tensor_path = output_dir / "predictions_and_targets.pt"
    torch.save(
        {
            "sample_ids": sample_ids,
            "predictions_fp32": prediction_tensors["fp32"],
            "predictions_fp16": prediction_tensors["fp16"],
            "targets": targets_tensor,
            "bands": list(BANDS),
        },
        tensor_path,
    )
    csv_path = output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sample_rows[0]))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    result = {
        "status": "passed" if passed else "failed_acceptance",
        "node": os.environ.get("BENCHMARK_NODE", "unknown"),
        "precision": {
            "fp32": "strict_fp32_no_autocast_no_tf32",
            "fp16": "native_fp16_weights_and_activations_no_autocast",
            "normalization": "FP32_then_cast_normalized_tensor_to_FP16",
            "metrics": "FP32_or_integer_accumulation",
        },
        "dataset": "Sen1Floods11 test",
        "processed_samples": processed,
        "batch_size_configured": BATCH_SIZE,
        "observed_batch_sizes": observed_batch_sizes,
        "sample_ids_sha256": hashlib.sha256(
            ("\n".join(sample_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "state_dict_load_fp32": load_fp32,
        "state_dict_load_fp16": load_fp16,
        "dtype_audit_fp32": dtype_fp32,
        "dtype_audit_fp16": dtype_fp16,
        "metrics": metrics,
        "fp16_minus_fp32_percentage_points": deltas_pp,
        "valid_pixel_comparison": {
            "compared": compared_valid_pixels,
            "changed": changed_valid_pixels,
            "agreement": valid_agreement,
        },
        "logit_error_fp16_vs_fp32": logit_error,
        "nonfinite_batches": nonfinite_batches,
        "input_identity_mismatches": input_identity_mismatches,
        "frozen_fp32_reference": {
            "paths": {key: str(value) for key, value in reference_paths.items()},
            "sha256": reference_hashes,
            "targets_exact": targets_exact,
            "fp32_predictions_exact": fp32_prediction_exact,
        },
        "thresholds": {
            "miou_max_drop_pp": args.miou_max_drop_pp,
            "water_iou_max_drop_pp": args.water_iou_max_drop_pp,
            "corrected_boundary_max_drop_pp": args.boundary_max_drop_pp,
            "min_valid_pixel_agreement": args.min_valid_pixel_agreement,
        },
        "acceptance": acceptance,
        "timing_seconds": {
            "model_build": build_seconds,
            "full_evaluation_including_data_and_metrics": evaluation_seconds,
            "synchronized_forward_sum_diagnostic_only": forward_seconds,
            "total": perf_counter() - started,
            "performance_claim_allowed": False,
        },
        "combined_peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "combined_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "software": {"torch": torch.__version__, "hip": torch.version.hip},
        "artifacts": {
            "predictions_and_targets": str(tensor_path),
            "per_sample_metrics": str(csv_path),
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("K100 FP32/FP16 FULL 90-SAMPLE ACCURACY GATE")
    print(f"samples          : {processed}/{EXPECTED_SAMPLES}")
    print(
        f"mIoU             : FP32={metrics['fp32']['miou']:.9f}, "
        f"FP16={metrics['fp16']['miou']:.9f}, delta={deltas_pp['miou']:+.6f} pp"
    )
    print(
        f"Water IoU        : FP32={metrics['fp32']['water_iou']:.9f}, "
        f"FP16={metrics['fp16']['water_iou']:.9f}, "
        f"delta={deltas_pp['water_iou']:+.6f} pp"
    )
    print(
        f"Boundary corrected: FP32={metrics['fp32']['boundary_water_iou_corrected']:.9f}, "
        f"FP16={metrics['fp16']['boundary_water_iou_corrected']:.9f}, "
        f"delta={deltas_pp['boundary_water_iou_corrected']:+.6f} pp"
    )
    print(f"valid agreement  : {valid_agreement:.8%} ({changed_valid_pixels} changed)")
    print(f"logit MAE/max    : {logit_error['mean_abs_error']:.8g} / {logit_error['max_abs_error']:.8g}")
    print(f"result           : {result['status']}")
    print(f"result JSON      : {result_path}")
    print("=" * 88)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
