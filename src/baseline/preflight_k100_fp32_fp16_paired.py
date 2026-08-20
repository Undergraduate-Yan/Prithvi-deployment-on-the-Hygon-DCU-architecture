"""Non-claiming runtime preflight for the paired benchmark implementation."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path

import torch

import benchmark_k100_fp32_fp16_paired as bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("K100 unavailable")

    checkpoint_path = bench.DEFAULT_CHECKPOINT.resolve(strict=True)
    if bench.sha256_file(checkpoint_path) != bench.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint mismatch")
    raw_inputs, sample_ids, input_hashes = bench.load_real_inputs()
    raw_inputs = raw_inputs.pin_memory()
    if not raw_inputs.is_pinned():
        raise RuntimeError("pinned input required")

    from terratorch.datamodules.sen1floods11 import MEANS, STDS
    from train_fp32_baseline import BANDS, build_task

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["state_dict"]
    device = torch.device("cuda:0")
    task_fp32, load_fp32, audit_fp32 = bench.build_loaded_model(
        "fp32", source_state, build_task, device
    )
    task_fp16, load_fp16, audit_fp16 = bench.build_loaded_model(
        "fp16", source_state, build_task, device
    )
    means = torch.tensor(
        [MEANS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)
    stds = torch.tensor(
        [STDS[band] for band in BANDS], device=device, dtype=torch.float32
    ).view(1, 6, 1, 1)

    scope_validation = {}
    for scope in bench.SCOPES:
        operations, retained = bench.make_operations(
            scope,
            raw_inputs[:1],
            task_fp32,
            task_fp16,
            means,
            stds,
            True,
            device,
        )
        outputs = {precision: operation() for precision, operation in operations.items()}
        torch.cuda.synchronize()
        scope_validation[scope] = bench.validate_output_pair(outputs, 1, scope)
        del outputs, operations, retained
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    del task_fp32, task_fp16
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory = {}
    for precision in bench.PRECISIONS:
        memory[precision] = bench.measure_isolated_memory(
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
        "purpose": "runtime preflight only; no latency or memory claim allowed",
        "sample_ids": sample_ids,
        "input_sha256": input_hashes,
        "checkpoint_sha256": bench.sha256_file(checkpoint_path),
        "script_sha256": bench.sha256_file(Path(bench.__file__).resolve()),
        "state_dict_load": {"fp32": load_fp32, "fp16": load_fp16},
        "dtype_audit": {"fp32": audit_fp32, "fp16": audit_fp16},
        "scope_validation": scope_validation,
        "isolated_memory_preflight": memory,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("=" * 80)
    print("paired benchmark runtime preflight: passed")
    for scope, row in scope_validation.items():
        print(
            f"{scope:22s}: agreement={row['pixel_class_agreement']:.8%}, "
            f"different={row['different_pixel_count']}"
        )
    for precision, row in memory.items():
        print(
            f"{precision} cleanup delta: "
            f"{row['cleanup_delta_from_process_baseline_bytes']} bytes"
        )
    print(f"result: {result_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
