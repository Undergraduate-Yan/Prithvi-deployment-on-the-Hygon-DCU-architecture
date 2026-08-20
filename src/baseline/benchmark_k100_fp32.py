"""Audited Prithvi FP32 performance benchmark for one Hygon K100.

Each round runs in a fresh Docker process.  The launcher executes three rounds
and then invokes this file in aggregate mode to produce pooled and per-round
statistics for batch sizes 1, 2, 4, and 8.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
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
TEST_INDICES = tuple(range(8))
WARMUP_RUNS = 30
MEASURED_RUNS = 100
REQUIRED_ROUNDS = 3

BATCH_ORDERS = {
    1: [1, 2, 4, 8],
    2: [4, 8, 1, 2],
    3: [8, 4, 2, 1],
}
SCOPE_ORDERS = {
    1: ["model_only", "host_to_host_logits", "host_to_host_argmax"],
    2: ["host_to_host_argmax", "host_to_host_logits", "model_only"],
    3: ["host_to_host_logits", "model_only", "host_to_host_argmax"],
}

sys.path.insert(0, str(PROJECT_ROOT))

from verify_real_sample_cpu_k100 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    migrate_and_strict_load,
    sha256_file,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--round-index", type=int, choices=(1, 2, 3))
    mode.add_argument("--aggregate-dir", type=Path)
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
        "std_ms": float(statistics.stdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
        "min_ms": float(min(latencies_ms)),
        "max_ms": float(max(latencies_ms)),
        "throughput_samples_per_second": float(
            batch_size * len(latencies_ms) * 1000.0 / sum(latencies_ms)
        ),
        "throughput_samples_per_second_at_median": float(batch_size * 1000.0 / median_ms),
        "single_sample_latency_ms_at_median": float(median_ms / batch_size),
    }


def read_first_sysfs(paths: list[str]) -> int | None:
    for pattern in paths:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            try:
                return int(path.read_text().strip())
            except (OSError, ValueError):
                continue
    return None


def gpu_sysfs_snapshot() -> dict[str, int | None]:
    return {
        "gpu_busy_percent": read_first_sysfs(
            ["/sys/class/drm/card1/device/gpu_busy_percent", "/sys/class/drm/card*/device/gpu_busy_percent"]
        ),
        "vram_used_bytes": read_first_sysfs(
            ["/sys/class/drm/card1/device/mem_info_vram_used", "/sys/class/drm/card*/device/mem_info_vram_used"]
        ),
        "vram_total_bytes": read_first_sysfs(
            ["/sys/class/drm/card1/device/mem_info_vram_total", "/sys/class/drm/card*/device/mem_info_vram_total"]
        ),
    }


def run_timed_loop(
    operation: Callable[[], torch.Tensor],
    batch_size: int,
    scope: str,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    output: torch.Tensor | None = None
    for _ in range(WARMUP_RUNS):
        output = operation()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(output.float()).all().item()):
            raise RuntimeError(
                f"Non-finite output during {scope} warm-up at batch {batch_size}"
            )
        del output
        output = None

    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    busy_before = gpu_sysfs_snapshot()
    latencies_ms: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for measured_index in range(MEASURED_RUNS):
            torch.cuda.synchronize()
            started_ns = perf_counter_ns()
            output = operation()
            torch.cuda.synchronize()
            latencies_ms.append((perf_counter_ns() - started_ns) / 1.0e6)
            if measured_index < MEASURED_RUNS - 1:
                del output
                output = None
    finally:
        if gc_was_enabled:
            gc.enable()
    busy_after = gpu_sysfs_snapshot()

    if output is None or not bool(torch.isfinite(output.float()).all().item()):
        raise RuntimeError(f"Non-finite output during {scope} measurement at batch {batch_size}")

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    result = {
        "scope": scope,
        "batch_size": batch_size,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "latencies_ms": latencies_ms,
        "statistics": latency_statistics(latencies_ms, batch_size),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "baseline_device_memory_bytes_after_warmup": baseline_allocated,
        "baseline_device_reserved_bytes_after_warmup": baseline_reserved,
        "peak_device_memory_bytes": peak_allocated,
        "peak_device_reserved_bytes": peak_reserved,
        "incremental_peak_device_memory_bytes": peak_allocated - baseline_allocated,
        "incremental_peak_device_reserved_bytes": peak_reserved - baseline_reserved,
        "current_device_memory_bytes": int(torch.cuda.memory_allocated()),
        "current_device_reserved_bytes": int(torch.cuda.memory_reserved()),
        "sysfs_before_measured_phase": busy_before,
        "sysfs_after_measured_phase": busy_after,
        "timing_method": (
            "torch.cuda.synchronize; perf_counter_ns; operation; "
            "torch.cuda.synchronize"
        ),
        "garbage_collector_disabled_during_measurement": True,
    }
    print(
        f"round scope={scope:22s} batch={batch_size}: "
        f"median={result['statistics']['median_ms']:.3f} ms, "
        f"P95={result['statistics']['p95_ms']:.3f} ms, "
        f"throughput={result['statistics']['throughput_samples_per_second']:.2f} sample/s",
        flush=True,
    )
    return result


def load_real_inputs() -> tuple[torch.Tensor, list[str], list[str]]:
    from train_fp32_baseline import build_datamodule

    datamodule = build_datamodule()
    datamodule.setup("test")
    dataset = datamodule.test_dataloader().dataset
    if len(dataset) != 90:
        raise RuntimeError(f"Expected 90 test samples, found {len(dataset)}")

    split_file = (
        PROJECT_ROOT
        / "data/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.txt"
    )
    all_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    inputs: list[torch.Tensor] = []
    sample_ids: list[str] = []
    hashes: list[str] = []
    for index in TEST_INDICES:
        sample = dataset[index]
        image = sample["image"].detach().contiguous().float()
        if image.shape != (6, 224, 224):
            raise ValueError(f"Unexpected sample {index} shape: {list(image.shape)}")
        inputs.append(image)
        sample_ids.append(all_ids[index])
        hashes.append(tensor_sha256(image))
    return torch.stack(inputs, dim=0).contiguous(), sample_ids, hashes


def benchmark_round(round_index: int, output_dir: Path, checkpoint_path: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
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

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Unexpected checkpoint SHA-256: {checkpoint_hash}; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )

    raw_inputs, sample_ids, input_hashes = load_real_inputs()
    pin_memory_error = None
    try:
        raw_inputs = raw_inputs.pin_memory()
    except RuntimeError as error:
        pin_memory_error = str(error)
    input_is_pinned = bool(raw_inputs.is_pinned())

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, build_task

    build_started = perf_counter()
    task = build_task().float().eval().cpu()
    build_seconds = perf_counter() - build_started
    checkpoint_started = perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_read_seconds = perf_counter() - checkpoint_started
    load_started = perf_counter()
    load_report = migrate_and_strict_load(task, checkpoint["state_dict"])
    strict_load_seconds = perf_counter() - load_started
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    del checkpoint
    gc.collect()

    device = torch.device("cuda:0")
    task = task.to(device=device, dtype=torch.float32).eval()
    non_fp32_parameters = [
        name for name, parameter in task.named_parameters() if parameter.dtype != torch.float32
    ]
    if non_fp32_parameters:
        raise RuntimeError(f"Non-FP32 model parameters: {non_fp32_parameters[:20]}")
    means = torch.tensor(
        [MEANS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    stds = torch.tensor(
        [STDS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    means_cpu = means.detach().cpu()
    stds_cpu = stds.detach().cpu()
    normalized_input_hashes = [
        tensor_sha256((raw_inputs[index : index + 1] - means_cpu) / stds_cpu)
        for index in range(len(TEST_INDICES))
    ]
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model_resident_memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }

    results: list[dict[str, Any]] = []
    for batch_size in BATCH_ORDERS[round_index]:
        raw_cpu = raw_inputs[:batch_size]
        for scope in SCOPE_ORDERS[round_index]:
            if scope == "model_only":
                device_input = raw_cpu.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=input_is_pinned,
                )
                device_input = ((device_input - means) / stds).contiguous()
                torch.cuda.synchronize()

                def operation() -> torch.Tensor:
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", enabled=False
                    ):
                        return task(device_input).output

                result = run_timed_loop(operation, batch_size, scope)
                if result["output_shape"] != [batch_size, 2, 224, 224]:
                    raise RuntimeError(f"Unexpected model-only output: {result['output_shape']}")
                if result["output_dtype"] != "torch.float32":
                    raise RuntimeError(f"Unexpected model-only dtype: {result['output_dtype']}")
                result["scope_definition"] = (
                    "normalized FP32 input resident on K100; model forward only; "
                    "logits remain on K100"
                )
                result["input_location_before_timer"] = "cuda:0"
                result["output_location_after_timer"] = "cuda:0"
                del device_input
            elif scope == "host_to_host_logits":

                def operation() -> torch.Tensor:
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", enabled=False
                    ):
                        device_raw = raw_cpu.to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=input_is_pinned,
                        )
                        normalized = (device_raw - means) / stds
                        logits = task(normalized).output
                        return logits.to("cpu", non_blocking=False)

                result = run_timed_loop(operation, batch_size, scope)
                if result["output_shape"] != [batch_size, 2, 224, 224]:
                    raise RuntimeError(f"Unexpected host logits output: {result['output_shape']}")
                if result["output_dtype"] != "torch.float32":
                    raise RuntimeError(f"Unexpected host logits dtype: {result['output_dtype']}")
                result["scope_definition"] = (
                    "raw FP32 reflectance in CPU memory; H2D; K100 normalization; "
                    "model forward; FP32 logits D2H; excludes disk/decode/resize"
                )
                result["input_location_before_timer"] = "pinned_cpu" if input_is_pinned else "cpu"
                result["output_location_after_timer"] = "cpu"
            elif scope == "host_to_host_argmax":

                def operation() -> torch.Tensor:
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", enabled=False
                    ):
                        device_raw = raw_cpu.to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=input_is_pinned,
                        )
                        normalized = (device_raw - means) / stds
                        logits = task(normalized).output
                        prediction = logits.argmax(dim=1).to(torch.uint8)
                        return prediction.to("cpu", non_blocking=False)

                result = run_timed_loop(operation, batch_size, scope)
                if result["output_shape"] != [batch_size, 224, 224]:
                    raise RuntimeError(f"Unexpected host argmax output: {result['output_shape']}")
                if result["output_dtype"] != "torch.uint8":
                    raise RuntimeError(f"Unexpected host argmax dtype: {result['output_dtype']}")
                result["scope_definition"] = (
                    "raw FP32 reflectance in CPU memory; H2D; K100 normalization; "
                    "model forward; argmax uint8 mask; D2H; excludes disk/decode/resize"
                )
                result["input_location_before_timer"] = "pinned_cpu" if input_is_pinned else "cpu"
                result["output_location_after_timer"] = "cpu"
            else:
                raise AssertionError(scope)
            results.append(result)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "passed",
        "benchmark": "Prithvi-EO-2.0-300M+UPerNet K100 FP32",
        "round_index": round_index,
        "independent_process_round": True,
        "node_label": os.environ.get("BENCHMARK_NODE", "unknown"),
        "host_label": os.environ.get("BENCHMARK_HOSTNAME", "unknown"),
        "container_image_tag": os.environ.get("PRITHVI_IMAGE_TAG"),
        "container_image_id": os.environ.get("PRITHVI_IMAGE_ID"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "state_dict_load": load_report,
        "total_parameters": sum(parameter.numel() for parameter in task.parameters()),
        "precision": "strict_fp32_no_autocast_no_tf32",
        "batch_order": BATCH_ORDERS[round_index],
        "scope_order_per_batch": SCOPE_ORDERS[round_index],
        "warmup_runs_per_scope_batch": WARMUP_RUNS,
        "measured_runs_per_scope_batch": MEASURED_RUNS,
        "test_indices": list(TEST_INDICES),
        "test_sample_ids": sample_ids,
        "test_input_sha256": input_hashes,
        "normalized_test_input_sha256": normalized_input_hashes,
        "input_is_pinned": input_is_pinned,
        "pin_memory_error": pin_memory_error,
        "normalization_means": [float(value) for value in means.flatten().tolist()],
        "normalization_stds": [float(value) for value in stds.flatten().tolist()],
        "device": {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "sysfs_at_start": gpu_sysfs_snapshot(),
        },
        "model_resident_device_memory": model_resident_memory,
        "results": results,
        "timing_seconds": {
            "model_build": build_seconds,
            "checkpoint_read": checkpoint_read_seconds,
            "strict_load": strict_load_seconds,
            "total_round": perf_counter() - started,
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
        "cpu": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "affinity_cpu_count": (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"round {round_index} result: {result_path}", flush=True)


def aggregate_results(root: Path) -> None:
    root = root.resolve()
    round_files = sorted(root.glob("round_*/result.json"))
    if len(round_files) != REQUIRED_ROUNDS:
        raise RuntimeError(
            f"Expected {REQUIRED_ROUNDS} round results under {root}, found {len(round_files)}"
        )
    rounds = [json.loads(path.read_text(encoding="utf-8")) for path in round_files]
    if any(item.get("status") != "passed" for item in rounds):
        raise RuntimeError("One or more benchmark rounds did not pass")

    identity_fields = (
        "checkpoint_sha256",
        "container_image_id",
        "test_sample_ids",
        "test_input_sha256",
        "normalized_test_input_sha256",
    )
    for field in identity_fields:
        reference = rounds[0][field]
        if any(item[field] != reference for item in rounds[1:]):
            raise RuntimeError(f"Round identity mismatch in {field}")

    aggregates: list[dict[str, Any]] = []
    all_valid = True
    for batch_size in BATCH_SIZES:
        for scope in ("model_only", "host_to_host_logits", "host_to_host_argmax"):
            round_items = []
            for round_result in rounds:
                matches = [
                    item
                    for item in round_result["results"]
                    if item["batch_size"] == batch_size and item["scope"] == scope
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one {scope}/batch{batch_size} result in round "
                        f"{round_result['round_index']}"
                    )
                round_items.append(matches[0])

            pooled = [
                float(latency)
                for item in round_items
                for latency in item["latencies_ms"]
            ]
            round_medians = [float(item["statistics"]["median_ms"]) for item in round_items]
            round_p95 = [float(item["statistics"]["p95_ms"]) for item in round_items]
            median_mean = float(statistics.fmean(round_medians))
            median_std = float(statistics.stdev(round_medians))
            median_cv = median_std / median_mean if median_mean > 0 else float("inf")
            pooled_stats = latency_statistics(pooled, batch_size)
            experimental_validity = bool(
                len(pooled) == REQUIRED_ROUNDS * MEASURED_RUNS
                and all(len(item["latencies_ms"]) == MEASURED_RUNS for item in round_items)
                and median_cv <= 0.05
            )
            all_valid = all_valid and experimental_validity
            aggregates.append(
                {
                    "scope": scope,
                    "batch_size": batch_size,
                    "round_count": REQUIRED_ROUNDS,
                    "measurements_per_round": MEASURED_RUNS,
                    "pooled_measurement_count": len(pooled),
                    "pooled_statistics": pooled_stats,
                    "round_medians_ms": round_medians,
                    "round_p95_ms": round_p95,
                    "round_median_mean_ms": median_mean,
                    "round_median_std_ms": median_std,
                    "round_median_cv_percent": median_cv * 100.0,
                    "round_median_range_percent_of_mean": (
                        (max(round_medians) - min(round_medians)) / median_mean * 100.0
                    ),
                    "round_p95_mean_ms": float(statistics.fmean(round_p95)),
                    "round_p95_std_ms": float(statistics.stdev(round_p95)),
                    "peak_device_memory_bytes_max_across_rounds": max(
                        int(item["peak_device_memory_bytes"]) for item in round_items
                    ),
                    "peak_device_reserved_bytes_max_across_rounds": max(
                        int(item["peak_device_reserved_bytes"]) for item in round_items
                    ),
                    "experimental_validity": {
                        "passed": experimental_validity,
                        "round_median_cv_max_percent": 5.0,
                    },
                }
            )

    for scope in ("model_only", "host_to_host_logits", "host_to_host_argmax"):
        scope_items = [item for item in aggregates if item["scope"] == scope]
        batch_one = next(item for item in scope_items if item["batch_size"] == 1)
        baseline_throughput = batch_one["pooled_statistics"][
            "throughput_samples_per_second"
        ]
        for item in scope_items:
            speedup = (
                item["pooled_statistics"]["throughput_samples_per_second"]
                / baseline_throughput
            )
            item["throughput_speedup_vs_batch1"] = speedup
            item["batch_efficiency_percent"] = speedup / item["batch_size"] * 100.0

    best_by_scope = {}
    for scope in ("model_only", "host_to_host_logits", "host_to_host_argmax"):
        candidates = [item for item in aggregates if item["scope"] == scope]
        best_by_scope[scope] = max(
            candidates,
            key=lambda item: item["pooled_statistics"][
                "throughput_samples_per_second"
            ],
        )["batch_size"]

    summary = {
        "status": "passed" if all_valid else "unstable",
        "benchmark": rounds[0]["benchmark"],
        "node_label": rounds[0]["node_label"],
        "host_label": rounds[0]["host_label"],
        "precision": rounds[0]["precision"],
        "round_count": REQUIRED_ROUNDS,
        "warmup_runs_per_scope_batch_round": WARMUP_RUNS,
        "measured_runs_per_scope_batch_round": MEASURED_RUNS,
        "batch_sizes": list(BATCH_SIZES),
        "scopes": {
            "model_only": (
                "normalized input resident on K100; forward only; logits remain on K100"
            ),
            "host_to_host_logits": (
                "raw in-memory CPU input; H2D; K100 normalization+forward; "
                "FP32 logits D2H; excludes disk/decode/resize"
            ),
            "host_to_host_argmax": (
                "raw in-memory CPU input; H2D; K100 normalization+forward+argmax; "
                "uint8 prediction D2H; excludes disk/decode/resize"
            ),
        },
        "checkpoint_sha256": rounds[0]["checkpoint_sha256"],
        "container_image_tag": rounds[0]["container_image_tag"],
        "container_image_id": rounds[0]["container_image_id"],
        "device": rounds[0]["device"],
        "software": rounds[0]["software"],
        "test_indices": rounds[0]["test_indices"],
        "test_sample_ids": rounds[0]["test_sample_ids"],
        "test_input_sha256": rounds[0]["test_input_sha256"],
        "normalized_test_input_sha256": rounds[0]["normalized_test_input_sha256"],
        "aggregates": aggregates,
        "highest_throughput_batch_by_scope": best_by_scope,
        "round_result_files": [str(path) for path in round_files],
        "created_at": datetime.now().astimezone().isoformat(),
        "interpretation_limits": [
            "K100 data-center accelerator proxy, not final edge or flight hardware.",
            "Host-to-host scope excludes storage I/O, TIFF decode, and resize.",
            "Power and temperature snapshots are captured by the launcher outside timed loops.",
            "No FP16 speed claim is made by this FP32-only baseline.",
        ],
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
        "round_count",
        "pooled_measurement_count",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "std_ms",
        "throughput_samples_per_second_at_median",
        "throughput_samples_per_second",
        "single_sample_latency_ms_at_median",
        "round_median_cv_percent",
        "throughput_speedup_vs_batch1",
        "batch_efficiency_percent",
        "peak_device_memory_bytes_max_across_rounds",
        "experimental_validity_passed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in aggregates:
            stats = item["pooled_statistics"]
            writer.writerow(
                {
                    "scope": item["scope"],
                    "batch_size": item["batch_size"],
                    "round_count": item["round_count"],
                    "pooled_measurement_count": item["pooled_measurement_count"],
                    "median_ms": stats["median_ms"],
                    "p95_ms": stats["p95_ms"],
                    "p99_ms": stats["p99_ms"],
                    "mean_ms": stats["mean_ms"],
                    "std_ms": stats["std_ms"],
                    "throughput_samples_per_second_at_median": stats[
                        "throughput_samples_per_second_at_median"
                    ],
                    "throughput_samples_per_second": stats[
                        "throughput_samples_per_second"
                    ],
                    "single_sample_latency_ms_at_median": stats[
                        "single_sample_latency_ms_at_median"
                    ],
                    "round_median_cv_percent": item["round_median_cv_percent"],
                    "throughput_speedup_vs_batch1": item[
                        "throughput_speedup_vs_batch1"
                    ],
                    "batch_efficiency_percent": item["batch_efficiency_percent"],
                    "peak_device_memory_bytes_max_across_rounds": item[
                        "peak_device_memory_bytes_max_across_rounds"
                    ],
                    "experimental_validity_passed": item["experimental_validity"][
                        "passed"
                    ],
                }
            )

    print("=" * 92)
    print(f"aggregate status: {summary['status']}")
    for item in aggregates:
        stats = item["pooled_statistics"]
        print(
            f"{item['scope']:22s} batch={item['batch_size']}: "
            f"median={stats['median_ms']:.3f} ms, "
            f"P95={stats['p95_ms']:.3f} ms, "
            f"P99={stats['p99_ms']:.3f} ms, "
            f"throughput={stats['throughput_samples_per_second']:.2f}/s, "
            f"round-CV={item['round_median_cv_percent']:.2f}%"
        )
    print(f"summary JSON: {summary_path}")
    print(f"summary CSV : {csv_path}")
    print("=" * 92)
    if not all_valid:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.aggregate_dir is not None:
        aggregate_results(args.aggregate_dir)
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required with --round-index")
    benchmark_round(
        round_index=args.round_index,
        output_dir=args.output_dir.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
    )


if __name__ == "__main__":
    main()
