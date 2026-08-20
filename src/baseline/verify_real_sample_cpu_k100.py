"""Verify one real Sen1Floods11 sample on CPU FP32 and Hygon K100 FP32.

This script is intended to run inside the pinned K100 Docker image through
``run_real_sample_cpu_k100.sh``.  It performs an audited state-dict namespace
migration, still loads with strict=True, and saves all comparison artifacts.
"""

from __future__ import annotations

import argparse
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


PROJECT_ROOT = Path(os.environ.get("PRITHVI_PROJECT_ROOT", "/workspace")).resolve()
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fp32_baseline_full/checkpoints/best-epoch46-step705.ckpt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "59f031f2eaa60219452c175825162ad30be158560716ccf01497019522cd8256"
)
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--mean-abs-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-abs-tolerance", type=float, default=5.0e-2)
    parser.add_argument("--min-pixel-agreement", type=float, default=0.999)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def extract_logits(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if hasattr(value, "output") and torch.is_tensor(value.output):
        return value.output
    if isinstance(value, dict):
        for key in ("output", "logits", "prediction", "predictions"):
            if key in value and torch.is_tensor(value[key]):
                return value[key]
    raise TypeError(f"Cannot extract logits from output type {type(value)!r}")


def migrate_and_strict_load(
    task: torch.nn.Module, source_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    target_state = task.state_dict()
    source_prefix = "model.encoder."
    target_prefix = "model.encoder._timm_module."
    needs_wrapper_migration = (
        any(key.startswith(target_prefix) for key in target_state)
        and any(
            key.startswith(source_prefix) and not key.startswith(target_prefix)
            for key in source_state
        )
    )

    migrated: dict[str, torch.Tensor] = {}
    renamed_count = 0
    for source_key, tensor in source_state.items():
        migrated_key = source_key
        if needs_wrapper_migration and source_key.startswith(source_prefix):
            migrated_key = target_prefix + source_key[len(source_prefix) :]
        if migrated_key in migrated:
            raise RuntimeError(
                f"State-dict namespace collision: {source_key!r} -> {migrated_key!r}"
            )
        migrated[migrated_key] = tensor
        renamed_count += int(migrated_key != source_key)

    target_keys = set(target_state)
    source_keys = set(migrated)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    shape_mismatches = [
        {
            "key": key,
            "checkpoint": list(migrated[key].shape),
            "model": list(target_state[key].shape),
        }
        for key in sorted(target_keys & source_keys)
        if tuple(migrated[key].shape) != tuple(target_state[key].shape)
    ]
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_key_count": len(missing),
                    "unexpected_key_count": len(unexpected),
                    "shape_mismatch_count": len(shape_mismatches),
                    "missing_keys": missing[:50],
                    "unexpected_keys": unexpected[:50],
                    "shape_mismatches": shape_mismatches[:50],
                },
                ensure_ascii=False,
            )
        )

    incompatible = task.load_state_dict(migrated, strict=True)
    return {
        "strict": True,
        "checkpoint_key_count": len(source_state),
        "model_key_count": len(target_state),
        "renamed_key_count": renamed_count,
        "from_prefix": source_prefix if renamed_count else None,
        "to_prefix": target_prefix if renamed_count else None,
        "collision_count": 0,
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "shape_mismatch_count": 0,
    }


def segmentation_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    valid = (target >= 0) & (target < 2)
    encoded = target[valid].to(torch.int64) * 2 + prediction[valid].to(torch.int64)
    confusion = torch.bincount(encoded, minlength=4).reshape(2, 2)
    intersection = confusion.diag().to(torch.float64)
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag()
    iou = []
    for class_index in range(2):
        denominator = int(union[class_index].item())
        iou.append(
            float(intersection[class_index].item() / denominator)
            if denominator > 0
            else None
        )
    available = [value for value in iou if value is not None]
    return {
        "valid_pixels": int(valid.sum().item()),
        "confusion_matrix": confusion.tolist(),
        "background_iou": iou[0],
        "water_iou": iou[1],
        "miou": float(sum(available) / len(available)) if available else None,
    }


def finite_float(value: float) -> float | None:
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
    from train_fp32_baseline import BANDS, build_datamodule, build_task

    datamodule = build_datamodule()
    datamodule.setup("test")
    dataset = datamodule.test_dataloader().dataset
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index {args.sample_index} outside dataset length {len(dataset)}"
        )
    sample = dataset[args.sample_index]
    raw_image = sample["image"].detach().contiguous().float()
    target = sample["mask"].detach().contiguous().long()
    if raw_image.shape != (6, 224, 224):
        raise ValueError(f"Expected image [6,224,224], found {list(raw_image.shape)}")
    if target.shape != (224, 224):
        raise ValueError(f"Expected mask [224,224], found {list(target.shape)}")
    raw_image = raw_image.unsqueeze(0)
    target = target.unsqueeze(0)

    split_file = (
        PROJECT_ROOT
        / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.txt"
    )
    sample_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    sample_id = sample_ids[args.sample_index]

    means = torch.tensor([MEANS[band] for band in BANDS], dtype=torch.float32).view(
        1, 6, 1, 1
    )
    stds = torch.tensor([STDS[band] for band in BANDS], dtype=torch.float32).view(
        1, 6, 1, 1
    )
    normalized_image = (raw_image - means) / stds

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

    cpu_started = perf_counter()
    with torch.inference_mode():
        logits_cpu = extract_logits(task(normalized_image)).detach().float().cpu()
    cpu_forward_seconds = perf_counter() - cpu_started
    if logits_cpu.shape != (1, 2, 224, 224):
        raise ValueError(f"Unexpected CPU logits shape: {list(logits_cpu.shape)}")

    transfer_started = perf_counter()
    task = task.to(device="cuda:0", dtype=torch.float32).eval()
    input_k100 = normalized_image.to(device="cuda:0", dtype=torch.float32)
    torch.cuda.synchronize()
    model_transfer_seconds = perf_counter() - transfer_started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    k100_started = perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        logits_k100 = extract_logits(task(input_k100))
    torch.cuda.synchronize()
    k100_forward_seconds = perf_counter() - k100_started
    logits_k100 = logits_k100.detach().float().cpu()
    if logits_k100.shape != logits_cpu.shape:
        raise ValueError(
            f"CPU/K100 shape mismatch: {list(logits_cpu.shape)} vs {list(logits_k100.shape)}"
        )

    difference = (logits_k100 - logits_cpu).abs()
    prediction_cpu = logits_cpu.argmax(dim=1)
    prediction_k100 = logits_k100.argmax(dim=1)
    pixel_agreement = float((prediction_cpu == prediction_k100).float().mean().item())
    mean_abs_error = float(difference.mean().item())
    max_abs_error = float(difference.max().item())
    rmse = float(torch.sqrt(torch.mean((logits_k100 - logits_cpu) ** 2)).item())
    cpu_finite = bool(torch.isfinite(logits_cpu).all().item())
    k100_finite = bool(torch.isfinite(logits_k100).all().item())
    passed = bool(
        cpu_finite
        and k100_finite
        and mean_abs_error <= args.mean_abs_tolerance
        and max_abs_error <= args.max_abs_tolerance
        and pixel_agreement >= args.min_pixel_agreement
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            PROJECT_ROOT
            / "outputs/k100_real_sample_parity"
            / f"sample_{args.sample_index:03d}_{timestamp}"
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    tensor_artifact = output_dir / "sample_and_logits.pt"
    torch.save(
        {
            "sample_index": args.sample_index,
            "sample_id": sample_id,
            "bands": list(BANDS),
            "raw_image": raw_image,
            "normalized_image": normalized_image,
            "target": target,
            "logits_cpu_fp32": logits_cpu,
            "logits_k100_fp32": logits_k100,
            "prediction_cpu": prediction_cpu,
            "prediction_k100": prediction_k100,
            "absolute_difference": difference,
        },
        tensor_artifact,
    )

    result = {
        "status": "passed" if passed else "failed_tolerance",
        "sample_index": args.sample_index,
        "sample_id": sample_id,
        "dataset_length": len(dataset),
        "bands": list(BANDS),
        "input_shape": list(raw_image.shape),
        "target_shape": list(target.shape),
        "raw_input_sha256": tensor_sha256(raw_image),
        "target_sha256": tensor_sha256(target),
        "raw_input_min": float(raw_image.min().item()),
        "raw_input_max": float(raw_image.max().item()),
        "normalization_means": [float(value) for value in means.flatten().tolist()],
        "normalization_stds": [float(value) for value in stds.flatten().tolist()],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "state_dict_load": load_report,
        "model_class": f"{type(task).__module__}.{type(task).__name__}",
        "total_parameters": sum(parameter.numel() for parameter in task.parameters()),
        "device_name": torch.cuda.get_device_name(0),
        "precision": "CPU FP32 versus K100 FP32; autocast and TF32 disabled",
        "cpu_output_shape": list(logits_cpu.shape),
        "k100_output_shape": list(logits_k100.shape),
        "cpu_output_finite": cpu_finite,
        "k100_output_finite": k100_finite,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "pixel_class_agreement": pixel_agreement,
        "different_class_pixel_count": int((prediction_cpu != prediction_k100).sum().item()),
        "tolerances": {
            "mean_abs_error_max": args.mean_abs_tolerance,
            "max_abs_error_max": args.max_abs_tolerance,
            "pixel_class_agreement_min": args.min_pixel_agreement,
        },
        "cpu_sample_metrics": segmentation_metrics(prediction_cpu, target),
        "k100_sample_metrics": segmentation_metrics(prediction_k100, target),
        "timing_seconds": {
            "model_build": build_seconds,
            "checkpoint_read": checkpoint_read_seconds,
            "strict_load": strict_load_seconds,
            "cpu_forward": cpu_forward_seconds,
            "model_cpu_to_k100": model_transfer_seconds,
            "k100_forward_single_unwarmed": k100_forward_seconds,
            "total": perf_counter() - started,
        },
        "k100_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "k100_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "tensor_artifact": str(tensor_artifact),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"sample: {sample_id} (test index {args.sample_index})")
    print(f"CPU/K100 mean abs error: {mean_abs_error:.8g}")
    print(f"CPU/K100 max abs error : {max_abs_error:.8g}")
    print(f"pixel class agreement  : {pixel_agreement:.8%}")
    print(f"result                  : {result['status']}")
    print(f"result JSON             : {result_path}")
    print(f"tensor artifact         : {tensor_artifact}")
    print("=" * 72)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
