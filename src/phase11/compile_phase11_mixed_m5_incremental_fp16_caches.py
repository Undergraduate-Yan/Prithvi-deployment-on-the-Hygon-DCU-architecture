#!/usr/bin/env python3
"""Compile and place-gate only the five new M5 FP16 segment MXR caches."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import compile_phase11_mixed_fp16_segment_caches as base


SCHEMA = "phase11_mixed_fp16_m5_incremental_cache_compile_v1"
MANIFEST_SCHEMA = "phase11_sensitivity_mixed_precision_candidate_v1"
M5_FP16_BLOCKS = (0, 14, 15, 16, 17, 18, 19, 20, 21)
REUSED_FP16_BLOCKS = (0, 14, 17, 18)
NEW_FP16_BLOCKS = (15, 16, 19, 20, 21)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    path = path.resolve(strict=True)
    bundle_root = path.parent.parent.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("status") != "created_static_pass":
        raise RuntimeError("M5 construction manifest schema/status drift")
    if payload.get("candidate_id") != "M5":
        raise RuntimeError("incremental cache compiler requires candidate M5")
    if tuple(payload.get("fp16_backbone_blocks", ())) != M5_FP16_BLOCKS:
        raise RuntimeError("M5 FP16 mapping drift")
    segments = payload.get("segments", [])
    if len(segments) != 25 or [row.get("index") for row in segments] != list(range(25)):
        raise RuntimeError("M5 construction segment order drift")

    selected = {int(row["index"]): row for row in segments if int(row["index"]) in NEW_FP16_BLOCKS}
    if tuple(sorted(selected)) != NEW_FP16_BLOCKS:
        raise RuntimeError("M5 new FP16 segment set drift")
    for index, row in selected.items():
        if row.get("precision") != "fp16":
            raise RuntimeError(f"M5 new segment {index} is not FP16")
        if row.get("cache") is not None or row.get("cache_identity") is not None:
            raise RuntimeError(f"M5 new FP16 segment {index} unexpectedly has a cache")
        if row.get("cache_action") != "compile_new_fp16_m5_incremental":
            raise RuntimeError(f"M5 new FP16 segment {index} cache action drift")
        model_path = base.resolve_manifest_path(path, str(row["model"]))
        base.require_under(model_path, bundle_root, f"M5 FP16 model {index}")
        base.identity(
            model_path,
            (int(row["model_identity"]["size_bytes"]), str(row["model_identity"]["sha256"])),
        )
        inputs = row.get("inputs", [])
        outputs = row.get("outputs", [])
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"M5 FP16 segment {index} must have one input and one output")
        if inputs[0].get("elem_type") != 1 or outputs[0].get("elem_type") != 1:
            raise RuntimeError(f"M5 FP16 segment {index} external I/O must be float32")

    for index in REUSED_FP16_BLOCKS:
        row = segments[index]
        if row.get("precision") != "fp16" or row.get("cache") is None:
            raise RuntimeError(f"M5 reused FP16 segment {index} cache contract drift")
        if row.get("cache_action") != "reuse_admitted_m4_fp16":
            raise RuntimeError(f"M5 reused FP16 segment {index} cache action drift")
        cache_path = base.resolve_manifest_path(path, str(row["cache"]))
        base.require_under(cache_path, bundle_root, f"M5 reused FP16 cache {index}")
        base.identity(
            cache_path,
            (int(row["cache_identity"]["size_bytes"]), str(row["cache_identity"]["sha256"])),
        )
    return payload, base.identity(path), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.m5_manifest.resolve(strict=True)
    bundle_root = manifest_path.parent.parent.resolve(strict=True)
    base.require_under(args.cache_root, bundle_root, "M5 incremental cache root", strict=False)
    base.require_under(args.output_dir, bundle_root, "M5 incremental cache evidence root", strict=False)
    if args.cache_root.exists() or args.output_dir.exists():
        raise FileExistsError("cache-root and output-dir must both be new")
    args.cache_root.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "claims": {
            "five_new_fp16_caches_compiled": False,
            "five_new_fp16_segments_strict_migraphx_placement": False,
            "five_new_fp16_outputs_finite": False,
            "four_m4_fp16_caches_reused_not_recompiled": False,
            "strict_numeric_equivalence": False,
            "task_accuracy_90": False,
            "performance": False,
            "native_fp16_kernel_verified": False,
        },
    }
    try:
        _, manifest_identity, selected = load_manifest(manifest_path)
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError(
                f"vendor runtime drift: ort={ort.__version__}, providers={ort.get_available_providers()}"
            )
        rows = []
        for index in NEW_FP16_BLOCKS:
            row = base.compile_one(
                ort=ort,
                manifest_path=manifest_path,
                row=selected[index],
                cache_root=args.cache_root.resolve(),
                output_root=args.output_dir.resolve(),
                device_id=args.device_id,
            )
            rows.append(row)
            print(f"M5 incremental FP16 cache {index:02d} compiled and placement-gated", flush=True)
        strict_numeric = all(
            all(row["diagnostic_strict_gates_not_an_admission_precondition"].values())
            for row in rows
        )
        result.update(
            {
                "status": "passed",
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "provider": MGX,
                    "device_id": args.device_id,
                    "cache_mode": "save_new_compiled_model",
                    "newly_compiled_blocks": list(NEW_FP16_BLOCKS),
                    "reused_without_recompile_blocks": list(REUSED_FP16_BLOCKS),
                    "intersegment_io_contract": "FP32 device OrtValue; full chain tested later",
                },
                "construction_manifest": manifest_identity,
                "segments": rows,
                "evidence_boundary": {
                    "random_deterministic_segment_inputs_used": True,
                    "cache_and_provider_placement_only": True,
                    "strict_numeric_gates_are_diagnostic_only": True,
                    "end_to_end_mixed_pipeline_not_tested": True,
                    "task_accuracy_not_tested": True,
                    "performance_not_measured": True,
                },
            }
        )
        result["claims"].update(
            {
                "five_new_fp16_caches_compiled": len(rows) == len(NEW_FP16_BLOCKS),
                "five_new_fp16_segments_strict_migraphx_placement": True,
                "five_new_fp16_outputs_finite": True,
                "four_m4_fp16_caches_reused_not_recompiled": True,
                "strict_numeric_equivalence": strict_numeric,
            }
        )
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    result_path = args.output_dir / "cache_compile_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
