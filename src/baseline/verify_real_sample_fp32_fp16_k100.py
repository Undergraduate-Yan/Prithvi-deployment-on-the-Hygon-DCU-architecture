"""Compare native FP32 and native FP16 on one real Sen1Floods11 sample.

The FP32 path has already been validated against CPU.  This gate checks that a
fully converted FP16 model and FP16 normalized input execute on K100, remain
finite, and preserve the FP32 prediction closely enough to proceed to the full
90-sample evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
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
sys.path.insert(0, str(PROJECT_ROOT))

from verify_real_sample_cpu_k100 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    extract_logits,
    migrate_and_strict_load,
    segmentation_metrics,
    sha256_file,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mean-abs-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--max-abs-tolerance", type=float, default=2.5e-1)
    parser.add_argument("--min-pixel-agreement", type=float, default=0.999)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def floating_dtype_audit(module: torch.nn.Module, expected: torch.dtype) -> dict[str, Any]:
    parameter_mismatches = [
        {"name": name, "dtype": str(parameter.dtype)}
        for name, parameter in module.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != expected
    ]
    buffer_mismatches = [
        {"name": name, "dtype": str(buffer.dtype)}
        for name, buffer in module.named_buffers()
        if buffer.is_floating_point() and buffer.dtype != expected
    ]
    return {
        "expected": str(expected),
        "floating_parameter_count": sum(
            parameter.is_floating_point() for parameter in module.parameters()
        ),
        "floating_buffer_count": sum(buffer.is_floating_point() for buffer in module.buffers()),
        "parameter_mismatches": parameter_mismatches,
        "buffer_mismatches": buffer_mismatches,
        "passed": not parameter_mismatches and not buffer_mismatches,
    }


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
        raise RuntimeError(
            f"Unexpected checkpoint SHA-256 {checkpoint_hash}; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, build_datamodule, build_task

    datamodule = build_datamodule()
    datamodule.setup("test")
    dataset = datamodule.test_dataloader().dataset
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)
    sample = dataset[args.sample_index]
    raw_image = sample["image"].detach().contiguous().float().unsqueeze(0)
    target = sample["mask"].detach().contiguous().long()
    if target.ndim == 3 and target.shape[0] == 1:
        target = target.squeeze(0)
    target = target.unsqueeze(0)
    if raw_image.shape != (1, 6, 224, 224):
        raise ValueError(f"Unexpected image shape {list(raw_image.shape)}")
    if target.shape != (1, 224, 224):
        raise ValueError(f"Unexpected target shape {list(target.shape)}")

    split_path = (
        PROJECT_ROOT
        / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.txt"
    )
    sample_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    sample_id = sample_ids[args.sample_index]
    means_cpu = torch.tensor(
        [MEANS[band] for band in BANDS], dtype=torch.float32
    ).view(1, 6, 1, 1)
    stds_cpu = torch.tensor(
        [STDS[band] for band in BANDS], dtype=torch.float32
    ).view(1, 6, 1, 1)
    normalized_fp32_cpu = ((raw_image - means_cpu) / stds_cpu).contiguous()

    build_started = perf_counter()
    task_fp32_cpu = build_task().float().eval().cpu()
    task_fp16_cpu = build_task().float().eval().cpu()
    build_seconds = perf_counter() - build_started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    load_report_fp32 = migrate_and_strict_load(
        task_fp32_cpu, checkpoint["state_dict"]
    )
    load_report_fp16 = migrate_and_strict_load(
        task_fp16_cpu, checkpoint["state_dict"]
    )
    del checkpoint

    device = torch.device("cuda:0")
    task_fp32 = task_fp32_cpu.to(device=device, dtype=torch.float32).eval()
    task_fp16 = task_fp16_cpu.to(device=device, dtype=torch.float16).eval()
    del task_fp32_cpu, task_fp16_cpu
    gc.collect()
    fp32_dtype_audit = floating_dtype_audit(task_fp32, torch.float32)
    fp16_dtype_audit = floating_dtype_audit(task_fp16, torch.float16)
    if not fp32_dtype_audit["passed"] or not fp16_dtype_audit["passed"]:
        raise RuntimeError("Model dtype audit failed")

    input_fp32 = normalized_fp32_cpu.to(device=device, dtype=torch.float32)
    input_fp16 = normalized_fp32_cpu.to(device=device, dtype=torch.float16)
    torch.cuda.synchronize()

    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        logits_fp32_device = extract_logits(task_fp32(input_fp32))
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        logits_fp16_device = extract_logits(task_fp16(input_fp16))
    torch.cuda.synchronize()

    if logits_fp32_device.dtype != torch.float32:
        raise TypeError(f"Expected FP32 logits, found {logits_fp32_device.dtype}")
    if logits_fp16_device.dtype != torch.float16:
        raise TypeError(f"Expected FP16 logits, found {logits_fp16_device.dtype}")
    if logits_fp32_device.shape != (1, 2, 224, 224):
        raise ValueError(f"Unexpected FP32 logits shape {list(logits_fp32_device.shape)}")
    if logits_fp16_device.shape != logits_fp32_device.shape:
        raise ValueError("FP32/FP16 output shape mismatch")

    logits_fp32 = logits_fp32_device.detach().float().cpu()
    logits_fp16 = logits_fp16_device.detach().float().cpu()
    fp32_finite = bool(torch.isfinite(logits_fp32).all().item())
    fp16_finite = bool(torch.isfinite(logits_fp16).all().item())
    absolute_difference = (logits_fp16 - logits_fp32).abs()
    mean_abs_error = float(absolute_difference.mean().item())
    max_abs_error = float(absolute_difference.max().item())
    rmse = float(torch.sqrt(torch.mean((logits_fp16 - logits_fp32) ** 2)).item())
    prediction_fp32 = logits_fp32.argmax(dim=1)
    prediction_fp16 = logits_fp16.argmax(dim=1)
    agreement = float((prediction_fp32 == prediction_fp16).float().mean().item())
    different_pixels = int((prediction_fp32 != prediction_fp16).sum().item())
    passed = bool(
        fp32_finite
        and fp16_finite
        and mean_abs_error <= args.mean_abs_tolerance
        and max_abs_error <= args.max_abs_tolerance
        and agreement >= args.min_pixel_agreement
    )

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "outputs/k100_fp32_fp16_real_sample" / timestamp
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    tensor_path = output_dir / "sample_and_logits.pt"
    torch.save(
        {
            "sample_index": args.sample_index,
            "sample_id": sample_id,
            "raw_image_fp32": raw_image,
            "normalized_input_fp32": normalized_fp32_cpu,
            "normalized_input_fp16_as_fp32": input_fp16.detach().float().cpu(),
            "target": target,
            "logits_fp32": logits_fp32,
            "logits_fp16_as_fp32": logits_fp16,
            "prediction_fp32": prediction_fp32,
            "prediction_fp16": prediction_fp16,
            "absolute_difference": absolute_difference,
        },
        tensor_path,
    )
    result = {
        "status": "passed" if passed else "failed_tolerance",
        "node": os.environ.get("BENCHMARK_NODE", "unknown"),
        "sample_index": args.sample_index,
        "sample_id": sample_id,
        "dataset_length": len(dataset),
        "input_shape": list(raw_image.shape),
        "raw_input_sha256": tensor_sha256(raw_image),
        "normalized_fp32_sha256": tensor_sha256(normalized_fp32_cpu),
        "target_sha256": tensor_sha256(target),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "state_dict_load_fp32": load_report_fp32,
        "state_dict_load_fp16": load_report_fp16,
        "fp32_dtype_audit": fp32_dtype_audit,
        "fp16_dtype_audit": fp16_dtype_audit,
        "fp32_output_dtype": str(logits_fp32_device.dtype),
        "fp16_output_dtype": str(logits_fp16_device.dtype),
        "fp32_output_finite": fp32_finite,
        "fp16_output_finite": fp16_finite,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "pixel_class_agreement": agreement,
        "different_class_pixel_count": different_pixels,
        "tolerances": {
            "mean_abs_error_max": args.mean_abs_tolerance,
            "max_abs_error_max": args.max_abs_tolerance,
            "pixel_class_agreement_min": args.min_pixel_agreement,
        },
        "fp32_sample_metrics": segmentation_metrics(prediction_fp32, target),
        "fp16_sample_metrics": segmentation_metrics(prediction_fp16, target),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "terratorch": package_version("terratorch"),
            "timm": package_version("timm"),
        },
        "timing_seconds": {
            "model_build": build_seconds,
            "total": perf_counter() - started,
            "performance_claim_allowed": False,
        },
        "artifacts": {"sample_and_logits": str(tensor_path)},
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print(f"sample                 : {sample_id} (test index {args.sample_index})")
    print(f"FP32/FP16 mean abs     : {mean_abs_error:.8g}")
    print(f"FP32/FP16 max abs      : {max_abs_error:.8g}")
    print(f"pixel class agreement  : {agreement:.8%}")
    print(f"different pixels       : {different_pixels}")
    print(f"result                 : {result['status']}")
    print(f"result JSON            : {result_path}")
    print("=" * 76)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
