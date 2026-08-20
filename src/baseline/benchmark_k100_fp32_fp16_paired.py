"""Audited same-trial FP32/FP16 benchmark for one Hygon K100.

Each process holds independently constructed FP32 and native FP16 models.  In
every batch/scope cell, both precisions are warmed and measured in alternating
blocks so that the speedup denominator is a concurrent FP32 observation rather
than the historical benchmark from another day.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


PROJECT_ROOT = Path(os.environ.get("PRITHVI_PROJECT_ROOT", "/workspace")).resolve()
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fp32_baseline_full/checkpoints/best-epoch46-step705.ckpt"
)
BATCH_SIZES = (1, 2, 4, 8)
SCOPES = ("model_only", "host_to_host_logits", "host_to_host_argmax")
PRECISIONS = ("fp32", "fp16")
TEST_INDICES = tuple(range(8))
WARMUP_BLOCKS = 6
WARMUP_PER_PRECISION_PER_BLOCK = 5
MEASURE_BLOCKS = 10
MEASURE_PER_PRECISION_PER_BLOCK = 10
MEASURED_RUNS = MEASURE_BLOCKS * MEASURE_PER_PRECISION_PER_BLOCK
REQUIRED_ROUNDS = 3
T_CRITICAL_DF2_975 = 4.302652729911275
MEMORY_CLEANUP_TOLERANCE_BYTES = 8 * 1024 * 1024

BATCH_ORDERS = {
    1: [1, 2, 4, 8],
    2: [2, 8, 1, 4],
    3: [4, 1, 8, 2],
}
SCOPE_ORDERS = {
    1: ["model_only", "host_to_host_logits", "host_to_host_argmax"],
    2: ["host_to_host_logits", "host_to_host_argmax", "model_only"],
    3: ["host_to_host_argmax", "model_only", "host_to_host_logits"],
}
MEMORY_ORDERS = {
    1: ["fp32", "fp16"],
    2: ["fp16", "fp32"],
    3: ["fp32", "fp16"],
}

sys.path.insert(0, str(PROJECT_ROOT))

from verify_real_sample_cpu_k100 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    migrate_and_strict_load,
    sha256_file,
    tensor_sha256,
)
from verify_real_sample_fp32_fp16_k100 import floating_dtype_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-protocol", type=Path)
    mode.add_argument("--round-index", type=int, choices=(1, 2, 3))
    mode.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--accuracy-result", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_statistics(latencies_ms: list[float], batch_size: int) -> dict[str, float]:
    median_ms = float(statistics.median(latencies_ms))
    return {
        "count": len(latencies_ms),
        "median_ms": median_ms,
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "mean_ms": float(statistics.fmean(latencies_ms)),
        "std_ms": float(statistics.stdev(latencies_ms)),
        "min_ms": float(min(latencies_ms)),
        "max_ms": float(max(latencies_ms)),
        "throughput_samples_per_second": float(
            batch_size * len(latencies_ms) * 1000.0 / sum(latencies_ms)
        ),
        "throughput_samples_per_second_at_median": float(
            batch_size * 1000.0 / median_ms
        ),
        "amortized_median_ms_per_sample": float(median_ms / batch_size),
    }


def geometric_summary(values: list[float]) -> dict[str, float | list[float]]:
    if len(values) != REQUIRED_ROUNDS or any(value <= 0 for value in values):
        raise ValueError(f"Expected {REQUIRED_ROUNDS} positive trial values: {values}")
    logs = [math.log(value) for value in values]
    log_mean = statistics.fmean(logs)
    log_std = statistics.stdev(logs)
    margin = T_CRITICAL_DF2_975 * log_std / math.sqrt(len(logs))
    mean = statistics.fmean(values)
    cv = statistics.stdev(values) / mean * 100.0
    return {
        "trial_values": values,
        "geometric_mean": math.exp(log_mean),
        "trial_log_t_95ci": [math.exp(log_mean - margin), math.exp(log_mean + margin)],
        "arithmetic_mean": mean,
        "trial_cv_percent": cv,
        "minimum": min(values),
        "maximum": max(values),
    }


def write_protocol(path: Path, accuracy_path: Path, image_id: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    accuracy_path = accuracy_path.resolve(strict=True)
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    if accuracy.get("status") != "passed":
        raise RuntimeError("The frozen 90-sample FP16 accuracy gate did not pass")
    if accuracy.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Accuracy result checkpoint mismatch")
    script_path = Path(__file__).resolve(strict=True)
    protocol = {
        "status": "frozen_before_timing",
        "benchmark": "K100 same-trial paired native FP32 versus native FP16",
        "primary_endpoint": {"scope": "model_only", "batch_size": 1},
        "deployment_secondary_endpoint": {
            "scope": "host_to_host_argmax",
            "batch_size": 1,
        },
        "node": "K100-2",
        "expected_host": "machine2",
        "container_image_tag": "prithvi-k100:20260807-dtk2504",
        "container_image_id": image_id,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "accuracy_result": str(accuracy_path),
        "accuracy_result_sha256": sha256_file(accuracy_path),
        "accuracy_status": accuracy["status"],
        "accuracy_key_results": {
            "fp16_minus_fp32_percentage_points": accuracy[
                "fp16_minus_fp32_percentage_points"
            ],
            "valid_pixel_comparison": accuracy["valid_pixel_comparison"],
        },
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "round_count": REQUIRED_ROUNDS,
        "batch_sizes": list(BATCH_SIZES),
        "batch_orders": BATCH_ORDERS,
        "scopes": list(SCOPES),
        "scope_orders": SCOPE_ORDERS,
        "memory_orders": MEMORY_ORDERS,
        "test_indices": list(TEST_INDICES),
        "warmup": {
            "blocks": WARMUP_BLOCKS,
            "runs_per_precision_per_block": WARMUP_PER_PRECISION_PER_BLOCK,
            "total_per_precision_per_cell": (
                WARMUP_BLOCKS * WARMUP_PER_PRECISION_PER_BLOCK
            ),
            "precision_order": "odd blocks FP32->FP16; even blocks FP16->FP32",
        },
        "measurement": {
            "blocks": MEASURE_BLOCKS,
            "runs_per_precision_per_block": MEASURE_PER_PRECISION_PER_BLOCK,
            "total_per_precision_per_cell": MEASURED_RUNS,
            "precision_order": "odd blocks FP32->FP16; even blocks FP16->FP32",
            "timer": "synchronize; perf_counter_ns; operation; synchronize",
        },
        "precision_definitions": {
            "fp32": "independent strict-loaded FP32 model; FP32 normalized input",
            "fp16": (
                "independent strict-loaded native FP16 model; normalization in FP32, "
                "then normalized tensor cast to FP16; no autocast or precision islands"
            ),
        },
        "scope_definitions": {
            "model_only": (
                "precision-matched normalized input resident on K100; model forward; "
                "native logits remain on K100"
            ),
            "host_to_host_logits": (
                "same pinned FP32 CPU raw input; H2D FP32; FP32 normalization; cast to "
                "model precision; forward; convert logits to FP32; FP32 logits D2H"
            ),
            "host_to_host_argmax": (
                "same pinned FP32 CPU raw input; H2D FP32; FP32 normalization; cast to "
                "model precision; forward; argmax; uint8 mask D2H"
            ),
        },
        "acceptance_thresholds": {
            "fp32_round_median_cv_max_percent": 5.0,
            "fp16_round_median_cv_max_percent": 5.0,
            "paired_median_speedup_cv_max_percent": 5.0,
            "all_trial_median_speedups_must_exceed": 1.0,
            "median_speedup_geometric_mean_min": 1.10,
            "median_speedup_95ci_lower_must_exceed": 1.0,
            "p95_ratio_geometric_mean_max": 1.05,
            "each_trial_p95_ratio_max": 1.10,
            "p95_ratio_95ci_upper_max": 1.10,
            "isolated_peak_memory_reduction_min_percent": 20.0,
            "all_trial_memory_reductions_must_exceed_percent": 0.0,
            "memory_cleanup_tolerance_bytes": MEMORY_CLEANUP_TOLERANCE_BYTES,
        },
        "start_conditions": {
            "gpu_busy_percent": 0,
            "continuous_idle_seconds": 15,
            "edge_temperature_max_c": 50.0,
            "memory_temperature_max_c": 72.0,
            "performance_level": "recorded, not changed",
        },
        "interpretation_limits": [
            "Three process-level trials are the independent units; raw iterations are technical repeats.",
            "Historical FP32 results are context only and are never used as the speedup denominator.",
            "PyTorch allocator memory excludes external KFD and driver/runtime allocation.",
            "K100 is a data-center accelerator proxy, not final edge or flight hardware.",
        ],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"protocol: {path}")
    print(f"protocol sha256: {sha256_file(path)}")


def load_real_inputs() -> tuple[torch.Tensor, list[str], list[str]]:
    from train_fp32_baseline import build_datamodule

    datamodule = build_datamodule()
    datamodule.setup("test")
    dataset = datamodule.test_dataloader().dataset
    split_path = (
        PROJECT_ROOT
        / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.txt"
    )
    all_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    inputs: list[torch.Tensor] = []
    hashes: list[str] = []
    sample_ids: list[str] = []
    for index in TEST_INDICES:
        image = dataset[index]["image"].detach().contiguous().float()
        if image.shape != (6, 224, 224):
            raise ValueError(f"Unexpected sample shape at {index}: {list(image.shape)}")
        inputs.append(image)
        hashes.append(tensor_sha256(image))
        sample_ids.append(all_ids[index])
    return torch.stack(inputs).contiguous(), sample_ids, hashes


def build_loaded_model(
    precision: str,
    source_state: dict[str, torch.Tensor],
    build_task: Callable[[], torch.nn.Module],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    dtype = torch.float32 if precision == "fp32" else torch.float16
    model = build_task().float().eval().cpu()
    load_report = migrate_and_strict_load(model, source_state)
    model = model.to(device=device, dtype=dtype).eval()
    dtype_report = floating_dtype_audit(model, dtype)
    if not dtype_report["passed"]:
        raise RuntimeError(f"{precision} dtype audit failed")
    return model, load_report, dtype_report


def precision_order(block_index: int) -> tuple[str, str]:
    return ("fp32", "fp16") if block_index % 2 == 1 else ("fp16", "fp32")


def validate_output_pair(
    outputs: dict[str, torch.Tensor], batch_size: int, scope: str
) -> dict[str, Any]:
    expected_shapes = {
        "model_only": [batch_size, 2, 224, 224],
        "host_to_host_logits": [batch_size, 2, 224, 224],
        "host_to_host_argmax": [batch_size, 224, 224],
    }
    expected_dtypes = {
        "model_only": {"fp32": "torch.float32", "fp16": "torch.float16"},
        "host_to_host_logits": {"fp32": "torch.float32", "fp16": "torch.float32"},
        "host_to_host_argmax": {"fp32": "torch.uint8", "fp16": "torch.uint8"},
    }
    report: dict[str, Any] = {}
    predictions: dict[str, torch.Tensor] = {}
    for precision, output in outputs.items():
        torch.cuda.synchronize()
        if list(output.shape) != expected_shapes[scope]:
            raise RuntimeError(f"Unexpected {scope}/{precision} shape: {list(output.shape)}")
        if str(output.dtype) != expected_dtypes[scope][precision]:
            raise RuntimeError(f"Unexpected {scope}/{precision} dtype: {output.dtype}")
        finite = bool(torch.isfinite(output).all().item())
        if not finite:
            raise RuntimeError(f"Non-finite {scope}/{precision} output")
        prediction = output.argmax(dim=1) if output.ndim == 4 else output.to(torch.int64)
        predictions[precision] = prediction.detach().cpu()
        report[precision] = {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": finite,
            "location": str(output.device),
        }
    agreement = float(
        (predictions["fp32"] == predictions["fp16"]).to(torch.float64).mean().item()
    )
    report["pixel_class_agreement"] = agreement
    report["different_pixel_count"] = int(
        (predictions["fp32"] != predictions["fp16"]).sum().item()
    )
    return report


def run_paired_cell(
    operations: dict[str, Callable[[], torch.Tensor]],
    batch_size: int,
    scope: str,
) -> dict[str, Any]:
    last_outputs: dict[str, torch.Tensor] = {}
    warmup_blocks: list[dict[str, Any]] = []
    for block_index in range(1, WARMUP_BLOCKS + 1):
        order = precision_order(block_index)
        block_started = datetime.now().astimezone().isoformat()
        for precision in order:
            torch.cuda.synchronize()
            for _ in range(WARMUP_PER_PRECISION_PER_BLOCK):
                last_outputs[precision] = operations[precision]()
            torch.cuda.synchronize()
        warmup_blocks.append(
            {
                "block_index": block_index,
                "precision_order": list(order),
                "started_at": block_started,
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )

    validation_outputs = {precision: operations[precision]() for precision in PRECISIONS}
    torch.cuda.synchronize()
    output_validation = validate_output_pair(validation_outputs, batch_size, scope)
    del validation_outputs, last_outputs

    latencies: dict[str, list[float]] = {precision: [] for precision in PRECISIONS}
    measurement_blocks: list[dict[str, Any]] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for block_index in range(1, MEASURE_BLOCKS + 1):
            order = precision_order(block_index)
            block_record: dict[str, Any] = {
                "block_index": block_index,
                "precision_order": list(order),
                "started_at": datetime.now().astimezone().isoformat(),
                "precision": {},
            }
            for precision in order:
                start_offset = len(latencies[precision])
                output = None
                for _ in range(MEASURE_PER_PRECISION_PER_BLOCK):
                    torch.cuda.synchronize()
                    start_ns = perf_counter_ns()
                    output = operations[precision]()
                    torch.cuda.synchronize()
                    latencies[precision].append((perf_counter_ns() - start_ns) / 1.0e6)
                end_offset = len(latencies[precision])
                block_values = latencies[precision][start_offset:end_offset]
                block_record["precision"][precision] = {
                    "start_index": start_offset,
                    "end_index_exclusive": end_offset,
                    "median_ms": float(statistics.median(block_values)),
                    "mean_ms": float(statistics.fmean(block_values)),
                }
                del output
            block_record["finished_at"] = datetime.now().astimezone().isoformat()
            measurement_blocks.append(block_record)
    finally:
        if gc_was_enabled:
            gc.enable()

    if any(len(values) != MEASURED_RUNS for values in latencies.values()):
        raise RuntimeError("Paired measurement count mismatch")
    statistics_by_precision = {
        precision: latency_statistics(values, batch_size)
        for precision, values in latencies.items()
    }
    paired = {
        "median_speedup_fp32_over_fp16": (
            statistics_by_precision["fp32"]["median_ms"]
            / statistics_by_precision["fp16"]["median_ms"]
        ),
        "throughput_speedup_fp16_over_fp32": (
            statistics_by_precision["fp16"]["throughput_samples_per_second"]
            / statistics_by_precision["fp32"]["throughput_samples_per_second"]
        ),
        "p95_ratio_fp16_over_fp32": (
            statistics_by_precision["fp16"]["p95_ms"]
            / statistics_by_precision["fp32"]["p95_ms"]
        ),
    }
    print(
        f"scope={scope:22s} batch={batch_size}: "
        f"FP32={statistics_by_precision['fp32']['median_ms']:.3f} ms, "
        f"FP16={statistics_by_precision['fp16']['median_ms']:.3f} ms, "
        f"speedup={paired['median_speedup_fp32_over_fp16']:.3f}x, "
        f"P95 ratio={paired['p95_ratio_fp16_over_fp32']:.3f}",
        flush=True,
    )
    return {
        "scope": scope,
        "batch_size": batch_size,
        "warmup_blocks": warmup_blocks,
        "measurement_blocks": measurement_blocks,
        "latencies_ms": latencies,
        "statistics": statistics_by_precision,
        "paired": paired,
        "output_validation": output_validation,
    }


def model_storage_bytes(model: torch.nn.Module) -> dict[str, int]:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "total_parameter_and_buffer_bytes": int(parameter_bytes + buffer_bytes),
    }


def measure_isolated_memory(
    precision: str,
    raw_inputs: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    source_state: dict[str, torch.Tensor],
    build_task: Callable[[], torch.nn.Module],
    device: torch.device,
) -> dict[str, Any]:
    dtype = torch.float32 if precision == "fp32" else torch.float16
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    process_baseline_allocated = int(torch.cuda.memory_allocated())
    process_baseline_reserved = int(torch.cuda.memory_reserved())
    model, load_report, dtype_report = build_loaded_model(
        precision, source_state, build_task, device
    )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    model_resident_allocated = int(torch.cuda.memory_allocated())
    model_resident_reserved = int(torch.cuda.memory_reserved())
    batch_results: list[dict[str, Any]] = []

    for batch_size in BATCH_SIZES:
        raw_cpu = raw_inputs[:batch_size]
        normalized_fp32 = raw_cpu.to(device=device, dtype=torch.float32)
        normalized_fp32 = ((normalized_fp32 - means) / stds).contiguous()
        device_input = (
            normalized_fp32
            if precision == "fp32"
            else normalized_fp32.to(dtype=torch.float16)
        )
        if precision == "fp16":
            del normalized_fp32
        torch.cuda.synchronize()

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            validation_output = model(device_input).output
        torch.cuda.synchronize()
        output_shape = list(validation_output.shape)
        output_dtype = str(validation_output.dtype)
        output_finite = bool(torch.isfinite(validation_output).all().item())
        if output_dtype != str(dtype) or not output_finite:
            raise RuntimeError(
                f"Invalid isolated-memory output for {precision}/batch{batch_size}"
            )
        del validation_output
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        baseline_with_input_allocated = int(torch.cuda.memory_allocated())
        baseline_with_input_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for _ in range(5):
                output = model(device_input).output
                torch.cuda.synchronize()
                del output
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        batch_results.append(
            {
                "batch_size": batch_size,
                "baseline_with_model_and_input_allocated_bytes": baseline_with_input_allocated,
                "baseline_with_model_and_input_reserved_bytes": baseline_with_input_reserved,
                "absolute_peak_allocated_bytes": peak_allocated,
                "absolute_peak_reserved_bytes": peak_reserved,
                "incremental_activation_peak_bytes": (
                    peak_allocated - baseline_with_input_allocated
                ),
                "peak_above_process_baseline_bytes": (
                    peak_allocated - process_baseline_allocated
                ),
                "output_shape": output_shape,
                "output_dtype": output_dtype,
                "output_finite": output_finite,
            }
        )
        del device_input
        if precision == "fp32":
            del normalized_fp32
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    storage = model_storage_bytes(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cleanup_allocated = int(torch.cuda.memory_allocated())
    cleanup_reserved = int(torch.cuda.memory_reserved())
    cleanup_delta = cleanup_allocated - process_baseline_allocated
    if abs(cleanup_delta) > MEMORY_CLEANUP_TOLERANCE_BYTES:
        raise RuntimeError(
            f"{precision} memory cleanup delta {cleanup_delta} exceeds "
            f"{MEMORY_CLEANUP_TOLERANCE_BYTES} bytes"
        )
    return {
        "precision": precision,
        "dtype": str(dtype),
        "process_baseline_allocated_bytes": process_baseline_allocated,
        "process_baseline_reserved_bytes": process_baseline_reserved,
        "model_resident_allocated_bytes": model_resident_allocated,
        "model_resident_reserved_bytes": model_resident_reserved,
        "model_resident_incremental_allocated_bytes": (
            model_resident_allocated - process_baseline_allocated
        ),
        "model_storage": storage,
        "state_dict_load": load_report,
        "dtype_audit": dtype_report,
        "batch_results": batch_results,
        "cleanup_allocated_bytes": cleanup_allocated,
        "cleanup_reserved_bytes": cleanup_reserved,
        "cleanup_delta_from_process_baseline_bytes": cleanup_delta,
        "cleanup_tolerance_bytes": MEMORY_CLEANUP_TOLERANCE_BYTES,
        "cleanup_passed": True,
    }


def make_operations(
    scope: str,
    raw_cpu: torch.Tensor,
    task_fp32: torch.nn.Module,
    task_fp16: torch.nn.Module,
    means: torch.Tensor,
    stds: torch.Tensor,
    input_is_pinned: bool,
    device: torch.device,
) -> tuple[dict[str, Callable[[], torch.Tensor]], list[torch.Tensor]]:
    retained: list[torch.Tensor] = []
    if scope == "model_only":
        normalized_fp32 = raw_cpu.to(
            device=device, dtype=torch.float32, non_blocking=input_is_pinned
        )
        normalized_fp32 = ((normalized_fp32 - means) / stds).contiguous()
        normalized_fp16 = normalized_fp32.to(dtype=torch.float16).contiguous()
        retained.extend([normalized_fp32, normalized_fp16])
        torch.cuda.synchronize()

        def fp32_operation() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                return task_fp32(normalized_fp32).output

        def fp16_operation() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                return task_fp16(normalized_fp16).output

    elif scope == "host_to_host_logits":

        def fp32_operation() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                raw_device = raw_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=input_is_pinned
                )
                normalized = (raw_device - means) / stds
                logits = task_fp32(normalized).output
                return logits.float().to("cpu", non_blocking=False)

        def fp16_operation() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                raw_device = raw_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=input_is_pinned
                )
                normalized = ((raw_device - means) / stds).to(torch.float16)
                logits = task_fp16(normalized).output
                return logits.float().to("cpu", non_blocking=False)

    elif scope == "host_to_host_argmax":

        def fp32_operation() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                raw_device = raw_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=input_is_pinned
                )
                normalized = (raw_device - means) / stds
                prediction = task_fp32(normalized).output.argmax(dim=1).to(torch.uint8)
                return prediction.to("cpu", non_blocking=False)

        def fp16_operation() -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                raw_device = raw_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=input_is_pinned
                )
                normalized = ((raw_device - means) / stds).to(torch.float16)
                prediction = task_fp16(normalized).output.argmax(dim=1).to(torch.uint8)
                return prediction.to("cpu", non_blocking=False)

    else:
        raise ValueError(scope)
    return {"fp32": fp32_operation, "fp16": fp16_operation}, retained


def run_round(
    round_index: int,
    output_dir: Path,
    checkpoint_path: Path,
    protocol_path: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol_path = protocol_path.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    if protocol["status"] != "frozen_before_timing":
        raise RuntimeError("Protocol is not frozen")
    if protocol["script_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Benchmark script changed after protocol freeze")
    if protocol["container_image_tag"] != os.environ.get("PRITHVI_IMAGE_TAG"):
        raise RuntimeError("Container image tag differs from the frozen protocol")
    if protocol["container_image_id"] != os.environ.get("PRITHVI_IMAGE_ID"):
        raise RuntimeError("Container image ID differs from the frozen protocol")
    if protocol["accuracy_status"] != "passed":
        raise RuntimeError("Accuracy gate did not pass")
    if not torch.cuda.is_available():
        raise RuntimeError("K100 is not available through torch.cuda")

    started = perf_counter()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 mismatch")
    raw_inputs, sample_ids, input_hashes = load_real_inputs()
    pin_error = None
    try:
        raw_inputs = raw_inputs.pin_memory()
    except RuntimeError as error:
        pin_error = str(error)
    input_is_pinned = bool(raw_inputs.is_pinned())
    if not input_is_pinned:
        raise RuntimeError(
            f"Pinned CPU input is required by the frozen protocol: {pin_error}"
        )

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, build_task

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["state_dict"]
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    device = torch.device("cuda:0")
    task_fp32, load_fp32, audit_fp32 = build_loaded_model(
        "fp32", source_state, build_task, device
    )
    task_fp16, load_fp16, audit_fp16 = build_loaded_model(
        "fp16", source_state, build_task, device
    )
    means = torch.tensor(
        [MEANS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    stds = torch.tensor(
        [STDS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    means_cpu = means.cpu()
    stds_cpu = stds.cpu()
    normalized_fp32_hashes = [
        tensor_sha256((raw_inputs[index : index + 1] - means_cpu) / stds_cpu)
        for index in range(len(TEST_INDICES))
    ]
    normalized_fp16_hashes = [
        tensor_sha256(
            ((raw_inputs[index : index + 1] - means_cpu) / stds_cpu).to(torch.float16)
        )
        for index in range(len(TEST_INDICES))
    ]
    torch.cuda.synchronize()
    timing_models_combined_allocated = int(torch.cuda.memory_allocated())
    timing_models_combined_reserved = int(torch.cuda.memory_reserved())

    results: list[dict[str, Any]] = []
    for batch_size in BATCH_ORDERS[round_index]:
        raw_cpu = raw_inputs[:batch_size]
        for scope in SCOPE_ORDERS[round_index]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            operations, retained = make_operations(
                scope,
                raw_cpu,
                task_fp32,
                task_fp16,
                means,
                stds,
                input_is_pinned,
                device,
            )
            results.append(run_paired_cell(operations, batch_size, scope))
            del operations, retained
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    del task_fp32, task_fp16
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before_isolated_memory_allocated = int(torch.cuda.memory_allocated())
    isolated_memory: dict[str, Any] = {}
    for precision in MEMORY_ORDERS[round_index]:
        isolated_memory[precision] = measure_isolated_memory(
            precision,
            raw_inputs,
            means,
            stds,
            source_state,
            build_task,
            device,
        )

    result = {
        "status": "passed",
        "benchmark": protocol["benchmark"],
        "round_index": round_index,
        "independent_process_round": True,
        "node_label": os.environ.get("BENCHMARK_NODE", "unknown"),
        "host_label": os.environ.get("BENCHMARK_HOSTNAME", platform.node()),
        "container_image_tag": os.environ.get("PRITHVI_IMAGE_TAG"),
        "container_image_id": os.environ.get("PRITHVI_IMAGE_ID"),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "state_dict_load": {"fp32": load_fp32, "fp16": load_fp16},
        "dtype_audit": {"fp32": audit_fp32, "fp16": audit_fp16},
        "batch_order": BATCH_ORDERS[round_index],
        "scope_order_per_batch": SCOPE_ORDERS[round_index],
        "memory_measurement_order": MEMORY_ORDERS[round_index],
        "test_indices": list(TEST_INDICES),
        "test_sample_ids": sample_ids,
        "test_input_sha256": input_hashes,
        "normalized_fp32_input_sha256": normalized_fp32_hashes,
        "normalized_fp16_input_sha256": normalized_fp16_hashes,
        "input_is_pinned": input_is_pinned,
        "pin_memory_error": pin_error,
        "timing_models_combined_memory": {
            "allocated_bytes": timing_models_combined_allocated,
            "reserved_bytes": timing_models_combined_reserved,
            "performance_or_memory_claim_allowed": False,
        },
        "before_isolated_memory_allocated_bytes": before_isolated_memory_allocated,
        "isolated_memory": isolated_memory,
        "results": results,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "terratorch": package_version("terratorch"),
            "timm": package_version("timm"),
            "lightning": package_version("lightning"),
            "torchgeo": package_version("torchgeo"),
            "numpy": np.__version__,
        },
        "timing_seconds": {"total_round": perf_counter() - started},
        "created_at": datetime.now().astimezone().isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"round {round_index} result: {result_path}", flush=True)


def classify_cell(
    fp32_cv: float,
    fp16_cv: float,
    speedup: dict[str, Any],
    p95_ratio: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[str, dict[str, bool]]:
    gates = {
        "fp32_median_cv": fp32_cv <= thresholds["fp32_round_median_cv_max_percent"],
        "fp16_median_cv": fp16_cv <= thresholds["fp16_round_median_cv_max_percent"],
        "paired_speedup_cv": (
            speedup["trial_cv_percent"]
            <= thresholds["paired_median_speedup_cv_max_percent"]
        ),
        "all_trial_speedups_above_one": all(
            value > thresholds["all_trial_median_speedups_must_exceed"]
            for value in speedup["trial_values"]
        ),
        "meaningful_speedup": (
            speedup["geometric_mean"]
            >= thresholds["median_speedup_geometric_mean_min"]
        ),
        "speedup_ci_lower": (
            speedup["trial_log_t_95ci"][0]
            > thresholds["median_speedup_95ci_lower_must_exceed"]
        ),
        "p95_geometric_mean": (
            p95_ratio["geometric_mean"]
            <= thresholds["p95_ratio_geometric_mean_max"]
        ),
        "each_trial_p95": all(
            value <= thresholds["each_trial_p95_ratio_max"]
            for value in p95_ratio["trial_values"]
        ),
        "p95_ci_upper": (
            p95_ratio["trial_log_t_95ci"][1]
            <= thresholds["p95_ratio_95ci_upper_max"]
        ),
    }
    if not (gates["fp32_median_cv"] and gates["fp16_median_cv"] and gates["paired_speedup_cv"]):
        return "unstable", gates
    if not (
        gates["all_trial_speedups_above_one"]
        and gates["meaningful_speedup"]
        and gates["speedup_ci_lower"]
    ):
        return "stable_but_not_meaningful", gates
    if not (
        gates["p95_geometric_mean"]
        and gates["each_trial_p95"]
        and gates["p95_ci_upper"]
    ):
        return "median_accelerated_tail_inconclusive", gates
    return "passed", gates


def aggregate(root: Path) -> None:
    root = root.resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    if protocol["script_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Aggregation script differs from the frozen protocol")
    round_files = sorted(root.glob("round_*/result.json"))
    if len(round_files) != REQUIRED_ROUNDS:
        raise RuntimeError(f"Expected {REQUIRED_ROUNDS} round results")
    rounds = [json.loads(path.read_text(encoding="utf-8")) for path in round_files]
    if any(item.get("status") != "passed" for item in rounds):
        raise RuntimeError("One or more execution rounds failed")
    identity_fields = (
        "checkpoint_sha256",
        "protocol_sha256",
        "benchmark_script_sha256",
        "container_image_id",
        "test_sample_ids",
        "test_input_sha256",
        "normalized_fp32_input_sha256",
        "normalized_fp16_input_sha256",
    )
    for field in identity_fields:
        reference = rounds[0][field]
        if any(item[field] != reference for item in rounds[1:]):
            raise RuntimeError(f"Round identity mismatch in {field}")
    if any(item["protocol_sha256"] != protocol_hash for item in rounds):
        raise RuntimeError("Round protocol hash mismatch")
    if rounds[0]["container_image_id"] != protocol["container_image_id"]:
        raise RuntimeError("Round image ID differs from the frozen protocol")

    thresholds = protocol["acceptance_thresholds"]
    aggregates: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        for scope in SCOPES:
            round_cells = []
            for round_result in rounds:
                matches = [
                    item
                    for item in round_result["results"]
                    if item["batch_size"] == batch_size and item["scope"] == scope
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"Missing {scope}/batch{batch_size}")
                round_cells.append(matches[0])

            precision_summary: dict[str, Any] = {}
            for precision in PRECISIONS:
                pooled = [
                    float(value)
                    for cell in round_cells
                    for value in cell["latencies_ms"][precision]
                ]
                round_medians = [
                    float(cell["statistics"][precision]["median_ms"])
                    for cell in round_cells
                ]
                mean_median = statistics.fmean(round_medians)
                precision_summary[precision] = {
                    "pooled_statistics_descriptive": latency_statistics(
                        pooled, batch_size
                    ),
                    "round_medians_ms": round_medians,
                    "round_median_cv_percent": (
                        statistics.stdev(round_medians) / mean_median * 100.0
                    ),
                    "round_p95_ms": [
                        float(cell["statistics"][precision]["p95_ms"])
                        for cell in round_cells
                    ],
                }

            median_speedups = [
                float(cell["paired"]["median_speedup_fp32_over_fp16"])
                for cell in round_cells
            ]
            throughput_speedups = [
                float(cell["paired"]["throughput_speedup_fp16_over_fp32"])
                for cell in round_cells
            ]
            p95_ratios = [
                float(cell["paired"]["p95_ratio_fp16_over_fp32"])
                for cell in round_cells
            ]
            speedup_summary = geometric_summary(median_speedups)
            throughput_summary = geometric_summary(throughput_speedups)
            p95_summary = geometric_summary(p95_ratios)
            classification, gates = classify_cell(
                precision_summary["fp32"]["round_median_cv_percent"],
                precision_summary["fp16"]["round_median_cv_percent"],
                speedup_summary,
                p95_summary,
                thresholds,
            )
            aggregates.append(
                {
                    "scope": scope,
                    "batch_size": batch_size,
                    "classification": classification,
                    "gates": gates,
                    "precision": precision_summary,
                    "paired_median_speedup": speedup_summary,
                    "paired_throughput_speedup": throughput_summary,
                    "paired_p95_ratio_fp16_over_fp32": p95_summary,
                    "independent_trial_count": REQUIRED_ROUNDS,
                    "technical_repeats_per_precision_per_trial": MEASURED_RUNS,
                }
            )

    memory_summary: dict[str, Any] = {}
    for precision in PRECISIONS:
        model_values = [
            int(item["isolated_memory"][precision]["model_resident_incremental_allocated_bytes"])
            for item in rounds
        ]
        memory_summary[precision] = {
            "model_resident_incremental_allocated_bytes_per_round": model_values,
            "model_resident_incremental_allocated_bytes_median": int(
                statistics.median(model_values)
            ),
            "model_storage_bytes_per_round": [
                int(
                    item["isolated_memory"][precision]["model_storage"][
                        "total_parameter_and_buffer_bytes"
                    ]
                )
                for item in rounds
            ],
            "batch": {},
        }
        for batch_size in BATCH_SIZES:
            peaks = []
            activations = []
            for item in rounds:
                matches = [
                    row
                    for row in item["isolated_memory"][precision]["batch_results"]
                    if row["batch_size"] == batch_size
                ]
                if len(matches) != 1:
                    raise RuntimeError("Isolated memory batch result missing")
                peaks.append(int(matches[0]["peak_above_process_baseline_bytes"]))
                activations.append(int(matches[0]["incremental_activation_peak_bytes"]))
            memory_summary[precision]["batch"][str(batch_size)] = {
                "isolated_peak_above_process_baseline_bytes_per_round": peaks,
                "isolated_peak_above_process_baseline_bytes_median": int(
                    statistics.median(peaks)
                ),
                "incremental_activation_peak_bytes_per_round": activations,
                "incremental_activation_peak_bytes_median": int(
                    statistics.median(activations)
                ),
            }

    memory_comparison_by_batch: dict[str, Any] = {}
    for batch_size in BATCH_SIZES:
        key = str(batch_size)
        fp32_peaks = memory_summary["fp32"]["batch"][key][
            "isolated_peak_above_process_baseline_bytes_per_round"
        ]
        fp16_peaks = memory_summary["fp16"]["batch"][key][
            "isolated_peak_above_process_baseline_bytes_per_round"
        ]
        paired_ratios = [
            fp16_peak / fp32_peak
            for fp32_peak, fp16_peak in zip(fp32_peaks, fp16_peaks, strict=True)
        ]
        paired_reductions = [
            (fp32_peak - fp16_peak) / fp32_peak * 100.0
            for fp32_peak, fp16_peak in zip(fp32_peaks, fp16_peaks, strict=True)
        ]
        memory_comparison_by_batch[key] = {
            "scope": "model_only_isolated_memory",
            "fp32_isolated_peak_bytes_per_round": fp32_peaks,
            "fp16_isolated_peak_bytes_per_round": fp16_peaks,
            "fp32_isolated_peak_bytes_median": int(statistics.median(fp32_peaks)),
            "fp16_isolated_peak_bytes_median": int(statistics.median(fp16_peaks)),
            "paired_fp16_over_fp32_peak_ratio": geometric_summary(paired_ratios),
            "paired_reduction_percent_per_round": paired_reductions,
            "paired_reduction_percent_median": float(
                statistics.median(paired_reductions)
            ),
            "all_rounds_reduce_memory": all(
                value
                > thresholds["all_trial_memory_reductions_must_exceed_percent"]
                for value in paired_reductions
            ),
        }
    memory_reduction_percent = memory_comparison_by_batch["1"][
        "paired_reduction_percent_median"
    ]
    memory_gate = (
        memory_reduction_percent
        >= thresholds["isolated_peak_memory_reduction_min_percent"]
        and memory_comparison_by_batch["1"]["all_rounds_reduce_memory"]
    )

    primary = next(
        item
        for item in aggregates
        if item["scope"] == protocol["primary_endpoint"]["scope"]
        and item["batch_size"] == protocol["primary_endpoint"]["batch_size"]
    )
    deployment_secondary = next(
        item
        for item in aggregates
        if item["scope"] == protocol["deployment_secondary_endpoint"]["scope"]
        and item["batch_size"]
        == protocol["deployment_secondary_endpoint"]["batch_size"]
    )
    summary = {
        "status": primary["classification"],
        "speed_claim_allowed": primary["classification"] == "passed",
        "memory_claim_allowed": memory_gate,
        "benchmark": protocol["benchmark"],
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "protocol_content": protocol,
        "primary_endpoint": primary,
        "deployment_secondary_endpoint": deployment_secondary,
        "aggregates": aggregates,
        "isolated_memory": memory_summary,
        "isolated_memory_comparison_by_batch": memory_comparison_by_batch,
        "batch1_model_only_isolated_peak_memory_reduction_percent_median": (
            memory_reduction_percent
        ),
        "batch1_model_only_isolated_peak_memory_gate_passed": memory_gate,
        "round_result_files": [str(path) for path in round_files],
        "container_image_id": rounds[0]["container_image_id"],
        "checkpoint_sha256": rounds[0]["checkpoint_sha256"],
        "device": rounds[0]["device"],
        "software": rounds[0]["software"],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    csv_path = root / "summary.csv"
    fields = [
        "scope",
        "batch_size",
        "classification",
        "fp32_pooled_median_ms",
        "fp16_pooled_median_ms",
        "fp32_pooled_p95_ms",
        "fp16_pooled_p95_ms",
        "fp32_pooled_throughput_samples_per_second",
        "fp16_pooled_throughput_samples_per_second",
        "fp32_round_median_cv_percent",
        "fp16_round_median_cv_percent",
        "median_speedup_geometric_mean",
        "median_speedup_ci_lower",
        "median_speedup_ci_upper",
        "median_speedup_trial_cv_percent",
        "throughput_speedup_geometric_mean",
        "p95_ratio_geometric_mean",
        "p95_ratio_ci_lower",
        "p95_ratio_ci_upper",
        "fp32_model_only_isolated_peak_bytes_median",
        "fp16_model_only_isolated_peak_bytes_median",
        "model_only_isolated_peak_memory_reduction_percent_median",
        "fp32_model_resident_bytes_median",
        "fp16_model_resident_bytes_median",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in aggregates:
            speed_ci = item["paired_median_speedup"]["trial_log_t_95ci"]
            p95_ci = item["paired_p95_ratio_fp16_over_fp32"]["trial_log_t_95ci"]
            writer.writerow(
                {
                    "scope": item["scope"],
                    "batch_size": item["batch_size"],
                    "classification": item["classification"],
                    "fp32_pooled_median_ms": item["precision"]["fp32"][
                        "pooled_statistics_descriptive"
                    ]["median_ms"],
                    "fp16_pooled_median_ms": item["precision"]["fp16"][
                        "pooled_statistics_descriptive"
                    ]["median_ms"],
                    "fp32_pooled_p95_ms": item["precision"]["fp32"][
                        "pooled_statistics_descriptive"
                    ]["p95_ms"],
                    "fp16_pooled_p95_ms": item["precision"]["fp16"][
                        "pooled_statistics_descriptive"
                    ]["p95_ms"],
                    "fp32_pooled_throughput_samples_per_second": item[
                        "precision"
                    ]["fp32"]["pooled_statistics_descriptive"][
                        "throughput_samples_per_second"
                    ],
                    "fp16_pooled_throughput_samples_per_second": item[
                        "precision"
                    ]["fp16"]["pooled_statistics_descriptive"][
                        "throughput_samples_per_second"
                    ],
                    "fp32_round_median_cv_percent": item["precision"]["fp32"][
                        "round_median_cv_percent"
                    ],
                    "fp16_round_median_cv_percent": item["precision"]["fp16"][
                        "round_median_cv_percent"
                    ],
                    "median_speedup_geometric_mean": item["paired_median_speedup"][
                        "geometric_mean"
                    ],
                    "median_speedup_ci_lower": speed_ci[0],
                    "median_speedup_ci_upper": speed_ci[1],
                    "median_speedup_trial_cv_percent": item[
                        "paired_median_speedup"
                    ]["trial_cv_percent"],
                    "throughput_speedup_geometric_mean": item[
                        "paired_throughput_speedup"
                    ]["geometric_mean"],
                    "p95_ratio_geometric_mean": item[
                        "paired_p95_ratio_fp16_over_fp32"
                    ]["geometric_mean"],
                    "p95_ratio_ci_lower": p95_ci[0],
                    "p95_ratio_ci_upper": p95_ci[1],
                    "fp32_model_only_isolated_peak_bytes_median": memory_comparison_by_batch[
                        str(item["batch_size"])
                    ]["fp32_isolated_peak_bytes_median"],
                    "fp16_model_only_isolated_peak_bytes_median": memory_comparison_by_batch[
                        str(item["batch_size"])
                    ]["fp16_isolated_peak_bytes_median"],
                    "model_only_isolated_peak_memory_reduction_percent_median": memory_comparison_by_batch[
                        str(item["batch_size"])
                    ]["paired_reduction_percent_median"],
                    "fp32_model_resident_bytes_median": memory_summary["fp32"][
                        "model_resident_incremental_allocated_bytes_median"
                    ],
                    "fp16_model_resident_bytes_median": memory_summary["fp16"][
                        "model_resident_incremental_allocated_bytes_median"
                    ],
                }
            )

    print("=" * 100)
    print(f"aggregate primary status: {summary['status']}")
    for item in aggregates:
        fp32 = item["precision"]["fp32"]["pooled_statistics_descriptive"]
        fp16 = item["precision"]["fp16"]["pooled_statistics_descriptive"]
        speed = item["paired_median_speedup"]
        tail = item["paired_p95_ratio_fp16_over_fp32"]
        print(
            f"{item['scope']:22s} batch={item['batch_size']}: "
            f"FP32={fp32['median_ms']:.3f}ms FP16={fp16['median_ms']:.3f}ms "
            f"speedup={speed['geometric_mean']:.3f}x "
            f"CI=[{speed['trial_log_t_95ci'][0]:.3f},{speed['trial_log_t_95ci'][1]:.3f}] "
            f"P95ratio={tail['geometric_mean']:.3f} "
            f"status={item['classification']}"
        )
    print(f"batch1 isolated peak memory reduction: {memory_reduction_percent:.2f}%")
    print(f"summary JSON: {summary_path}")
    print(f"summary CSV : {csv_path}")
    print("=" * 100)
    if summary["status"] != "passed":
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.write_protocol is not None:
        if args.accuracy_result is None or not args.image_id:
            raise SystemExit("--accuracy-result and --image-id are required")
        write_protocol(
            args.write_protocol.resolve(), args.accuracy_result, args.image_id
        )
        return
    if args.aggregate_dir is not None:
        aggregate(args.aggregate_dir)
        return
    if args.output_dir is None or args.protocol is None:
        raise SystemExit("--output-dir and --protocol are required with --round-index")
    run_round(
        args.round_index,
        args.output_dir.resolve(),
        args.checkpoint,
        args.protocol,
    )


if __name__ == "__main__":
    main()
