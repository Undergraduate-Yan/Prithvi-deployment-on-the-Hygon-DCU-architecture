from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(os.environ.get("PRITHVI_PROJECT_ROOT", "/workspace")).resolve()
CHECKPOINT = Path(os.environ.get("PRITHVI_CHECKPOINT", PROJECT_ROOT / "outputs/fp32_baseline_full/checkpoints/best-epoch46-step705.ckpt")).resolve()
SEED = int(os.environ.get("PRITHVI_FORWARD_SEED", "42"))
sys.path.insert(0, str(PROJECT_ROOT))


def extract_tensor(value: Any) -> tuple[torch.Tensor, str]:
    if torch.is_tensor(value):
        return value, "tensor"
    if hasattr(value, "output") and torch.is_tensor(value.output):
        return value.output, f"{type(value).__name__}.output"
    if isinstance(value, dict):
        for key in ("output", "logits", "prediction", "predictions"):
            if key in value and torch.is_tensor(value[key]):
                return value[key], f"dict[{key!r}]"
        for key, item in value.items():
            if torch.is_tensor(item):
                return item, f"dict[{key!r}]"
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if torch.is_tensor(item):
                return item, f"{type(value).__name__}[{index}]"
    raise TypeError(f"Cannot extract output tensor from {type(value)!r}")


def main() -> None:
    started = time.perf_counter()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("K100 is not available through torch.cuda")

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, build_task

    build_started = time.perf_counter()
    task = build_task()
    build_seconds = time.perf_counter() - build_started
    target_state = task.state_dict()

    load_file_started = time.perf_counter()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    source_state_original = checkpoint["state_dict"]
    checkpoint_read_seconds = time.perf_counter() - load_file_started

    source_state = {}
    renamed_keys: list[tuple[str, str]] = []
    source_encoder_prefix = "model.encoder."
    target_encoder_prefix = "model.encoder._timm_module."
    needs_encoder_wrapper_migration = (
        any(key.startswith(target_encoder_prefix) for key in target_state)
        and any(
            key.startswith(source_encoder_prefix) and not key.startswith(target_encoder_prefix)
            for key in source_state_original
        )
    )
    for source_key, tensor in source_state_original.items():
        migrated_key = source_key
        if needs_encoder_wrapper_migration and source_key.startswith(source_encoder_prefix):
            migrated_key = target_encoder_prefix + source_key[len(source_encoder_prefix) :]
        if migrated_key in source_state:
            raise RuntimeError(
                f"State-dict namespace migration collision: {source_key!r} -> {migrated_key!r}"
            )
        source_state[migrated_key] = tensor
        if migrated_key != source_key:
            renamed_keys.append((source_key, migrated_key))

    target_keys = set(target_state)
    source_keys = set(source_state)
    missing_keys = sorted(target_keys - source_keys)
    unexpected_keys = sorted(source_keys - target_keys)
    shape_mismatches = [
        {
            "key": key,
            "checkpoint": list(source_state[key].shape),
            "model": list(target_state[key].shape),
        }
        for key in sorted(target_keys & source_keys)
        if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_key_count": len(missing_keys),
                    "unexpected_key_count": len(unexpected_keys),
                    "shape_mismatch_count": len(shape_mismatches),
                    "missing_keys": missing_keys[:50],
                    "unexpected_keys": unexpected_keys[:50],
                    "shape_mismatches": shape_mismatches[:50],
                },
                ensure_ascii=False,
            )
        )

    strict_started = time.perf_counter()
    incompatible = task.load_state_dict(source_state, strict=True)
    strict_load_seconds = time.perf_counter() - strict_started
    del checkpoint, source_state, source_state_original, target_state

    task.eval()
    task.to(device="cuda", dtype=torch.float32)
    parameter = next(task.parameters())
    torch.manual_seed(SEED)
    raw_input_cpu = torch.rand((1, 6, 224, 224), dtype=torch.float32)
    means_cpu = torch.tensor([MEANS[band] for band in BANDS], dtype=torch.float32).view(1, 6, 1, 1)
    stds_cpu = torch.tensor([STDS[band] for band in BANDS], dtype=torch.float32).view(1, 6, 1, 1)
    input_cpu = (raw_input_cpu - means_cpu) / stds_cpu
    input_tensor = input_cpu.to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    forward_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        raw_output = task(input_tensor)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_started
    output, output_source = extract_tensor(raw_output)
    output_float = output.detach().float()
    finite = bool(torch.isfinite(output_float).all().item())
    if not finite:
        raise RuntimeError("Forward output contains NaN or Inf")

    result = {
        "status": "passed",
        "project_root": str(PROJECT_ROOT),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "checkpoint_state_key_count": len(target_keys),
        "state_dict_namespace_migration": {
            "applied": bool(renamed_keys),
            "from_prefix": source_encoder_prefix if renamed_keys else None,
            "to_prefix": target_encoder_prefix if renamed_keys else None,
            "renamed_key_count": len(renamed_keys),
            "collision_count": 0,
        },
        "model_class": f"{type(task).__module__}.{type(task).__name__}",
        "wrapped_model_class": f"{type(task.model).__module__}.{type(task.model).__name__}",
        "total_parameters": sum(p.numel() for p in task.parameters()),
        "trainable_parameters": sum(p.numel() for p in task.parameters() if p.requires_grad),
        "strict_load": True,
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "load_state_dict_missing": list(incompatible.missing_keys),
        "load_state_dict_unexpected": list(incompatible.unexpected_keys),
        "device_name": torch.cuda.get_device_name(0),
        "parameter_device": str(parameter.device),
        "parameter_dtype": str(parameter.dtype),
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "input_semantics": "deterministic synthetic reflectance normalized with Sen1Floods11 training statistics",
        "bands": list(BANDS),
        "normalization_means": [float(value) for value in means_cpu.flatten().tolist()],
        "normalization_stds": [float(value) for value in stds_cpu.flatten().tolist()],
        "raw_input_min": float(raw_input_cpu.min().item()),
        "raw_input_max": float(raw_input_cpu.max().item()),
        "normalized_input_min": float(input_cpu.min().item()),
        "normalized_input_max": float(input_cpu.max().item()),
        "input_seed": SEED,
        "precision": "strict_fp32_no_autocast",
        "raw_output_type": f"{type(raw_output).__module__}.{type(raw_output).__name__}",
        "output_source": output_source,
        "output_shape": list(output.shape),
        "output_device": str(output.device),
        "output_dtype": str(output.dtype),
        "output_finite": finite,
        "output_min": float(output_float.min().item()),
        "output_max": float(output_float.max().item()),
        "output_mean": float(output_float.mean().item()),
        "output_std": float(output_float.std().item()),
        "argmax_class_counts": {
            str(int(key)): int(value)
            for key, value in zip(*torch.unique(output_float.argmax(dim=1), return_counts=True))
        } if output_float.ndim == 4 else {},
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_device_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "build_seconds": build_seconds,
        "checkpoint_read_seconds": checkpoint_read_seconds,
        "strict_load_seconds": strict_load_seconds,
        "forward_seconds_single_unwarmed": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }
    print("DAY2_RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
