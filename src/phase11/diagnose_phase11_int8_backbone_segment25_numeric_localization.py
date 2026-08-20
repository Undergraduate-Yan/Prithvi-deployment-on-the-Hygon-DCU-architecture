#!/usr/bin/env python3
"""Localize CPU-vs-MIGraphX numerical drift across the frozen 25-segment pipeline.

This is a diagnostic, not a performance benchmark.  Each segment is evaluated in
two ways on MIGraphX:

* local: the segment receives the CPU-reference input, isolating backend drift;
* cumulative: the segment receives the preceding MIGraphX output, measuring the
  end-to-end accumulated drift.

All frozen models and compiled caches remain read-only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


REPORT = (21_342, "e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e")
SAMPLE = (4_820_344, "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62")
FROZEN_DEVICE_PAIRED = (1_943_271, "535cee1507c33759e0b2366d02be16ed6ab963c90710be77e2d41742a2718bb1")
RAW_SHA = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24)) + ("upernet_decoder_head",)
RETAIN_AFTER = (5, 11, 17, 23)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_raw(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found: list[torch.Tensor] = []

    def visit(value) -> None:
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(obj)
    if not found:
        raise RuntimeError("no 1x6x224x224 sample tensor found")
    raw = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if hashlib.sha256(raw.tobytes()).hexdigest() != RAW_SHA:
        raise RuntimeError("raw input identity drift")
    return raw


def session_options(ort, *, strict: bool, profile_prefix: Path | None = None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    if strict:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def parse_profile(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return {
        "provider_event_counts": dict(counts),
        "migraphx_positive": counts[MGX] > 0,
        "cpu_zero": counts[CPU] == 0,
        "passed": counts[MGX] > 0 and counts[CPU] == 0,
    }


def compare(reference: np.ndarray, candidate: np.ndarray, *, with_classes: bool = False) -> dict:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise RuntimeError(f"comparison shape mismatch: {reference64.shape} vs {candidate64.shape}")
    difference = candidate64 - reference64
    absolute = np.abs(difference)
    reference_l2 = float(np.linalg.norm(reference64.ravel()))
    difference_l2 = float(np.linalg.norm(difference.ravel()))
    denominator = float(np.linalg.norm(reference64.ravel()) * np.linalg.norm(candidate64.ravel()))
    item = {
        "shape": list(reference64.shape),
        "element_count": int(reference64.size),
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p50_abs": float(np.percentile(absolute, 50)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "mean_signed": float(difference.mean()),
        "relative_l2": float(difference_l2 / max(reference_l2, np.finfo(np.float64).tiny)),
        "cosine_similarity": float(np.dot(reference64.ravel(), candidate64.ravel()) / denominator) if denominator else 1.0,
        "exact_equal": bool(np.array_equal(reference, candidate)),
        "all_finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }
    if with_classes:
        reference_class = np.argmax(reference, axis=1)
        candidate_class = np.argmax(candidate, axis=1)
        item.update(
            {
                "pixel_class_agreement": float(np.mean(reference_class == candidate_class)),
                "changed_pixels": int(np.count_nonzero(reference_class != candidate_class)),
            }
        )
    return item


def cache_paths(root: Path, index: int) -> tuple[Path, Path]:
    base = (
        root / "segment25_static_cache54g_segment0"
        if index == 0
        else root / "segment25_static_remaining_caches" / f"segment_{index:02d}"
    )
    return base / f"segment_{index:02d}.mxr", base / "output/result.json"


def input_map(session, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = [item.name for item in session.get_inputs()]
    if set(names) != set(arrays):
        raise RuntimeError(f"input contract mismatch: session={names}, supplied={sorted(arrays)}")
    return {name: np.ascontiguousarray(arrays[name]) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    result = {
        "schema": "phase11_int8_backbone_segment25_numeric_localization_v1",
        "status": "failed",
        "variant": "frozen_int8_backbone_25segment_cpu_vs_strict_migraphx_local_and_cumulative",
        "claims": {
            "all_25_segments_profiled": False,
            "first_backend_divergence_localized": False,
            "strict_end_to_end_numeric_admission": False,
            "performance": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers() or CPU not in ort.get_available_providers():
            raise RuntimeError(f"runtime identity drift: {ort.__version__}, {ort.get_available_providers()}")

        report_path = args.root / "segment25_static_build_cpu" / "segment_build_report.json"
        sample_path = args.root / "sample_and_logits.pt"
        paired_path = (
            args.root
            / "segment25_iobinding_end_to_end_single"
            / "output"
            / "cpu_host_staged_and_device_logits.npz"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report.get("segments", [])
        if len(rows) != 25 or [row.get("label") for row in rows] != list(LABELS):
            raise RuntimeError("segment manifest/order drift")

        identities = {
            "segment_report": identity(report_path, REPORT),
            "sample": identity(sample_path, SAMPLE),
            "frozen_device_pipeline_paired_logits": identity(paired_path, FROZEN_DEVICE_PAIRED),
        }
        frozen = np.load(paired_path, allow_pickle=False)
        frozen_cpu_logits = np.ascontiguousarray(frozen["cpu_logits"], dtype=np.float32)
        frozen_migraphx_logits = np.ascontiguousarray(frozen["host_staged_migraphx_logits"], dtype=np.float32)

        cpu_current = load_raw(sample_path)
        mgx_current = cpu_current.copy()
        cpu_retained: dict[str, np.ndarray] = {}
        mgx_retained: dict[str, np.ndarray] = {}
        stage_rows: list[dict] = []
        profiles: list[dict] = []

        for index, manifest_row in enumerate(rows):
            label = LABELS[index]
            model_path = args.root / "segment25_static_build_cpu" / "models" / Path(manifest_row["path"]).name
            identities[f"model_{index:02d}"] = identity(
                model_path, (int(manifest_row["size_bytes"]), str(manifest_row["sha256"]))
            )
            cache_path, cache_result_path = cache_paths(args.root, index)
            cache_result = json.loads(cache_result_path.read_text(encoding="utf-8"))
            cache_meta = cache_result["compiled_cache"]
            identities[f"cache_{index:02d}"] = identity(
                cache_path, (int(cache_meta["size_bytes"]), str(cache_meta["sha256"]))
            )

            cpu_session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options(ort, strict=False),
                providers=[CPU],
            )
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache_path.resolve())
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache_path.resolve())
            profile_prefix = args.output_dir / f"segment_{index:02d}_numeric_profile"
            mgx_session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options(ort, strict=True, profile_prefix=profile_prefix),
                providers=[(MGX, {"device_id": 0})],
            )
            mgx_session.disable_fallback()
            if mgx_session.get_providers()[0] != MGX:
                raise RuntimeError(f"provider priority drift at segment {index}")

            if index < 24:
                input_name = cpu_session.get_inputs()[0].name
                cpu_inputs = {input_name: cpu_current}
                cumulative_inputs = {input_name: mgx_current}
            else:
                cpu_inputs = dict(cpu_retained)
                cumulative_inputs = dict(mgx_retained)

            cpu_inputs = input_map(cpu_session, cpu_inputs)
            local_mgx_inputs = input_map(mgx_session, cpu_inputs)
            cumulative_mgx_inputs = input_map(mgx_session, cumulative_inputs)
            input_drift = {
                name: compare(cpu_inputs[name], cumulative_mgx_inputs[name])
                for name in cpu_inputs
            }

            output_name = cpu_session.get_outputs()[0].name
            started = perf_counter()
            cpu_output = np.ascontiguousarray(cpu_session.run([output_name], cpu_inputs)[0])
            cpu_seconds = perf_counter() - started
            started = perf_counter()
            mgx_local_output = np.ascontiguousarray(mgx_session.run([output_name], local_mgx_inputs)[0])
            mgx_local_seconds = perf_counter() - started
            if index == 0:
                mgx_cumulative_output = mgx_local_output.copy()
                mgx_cumulative_seconds = mgx_local_seconds
            else:
                started = perf_counter()
                mgx_cumulative_output = np.ascontiguousarray(
                    mgx_session.run([output_name], cumulative_mgx_inputs)[0]
                )
                mgx_cumulative_seconds = perf_counter() - started

            if not (
                np.isfinite(cpu_output).all()
                and np.isfinite(mgx_local_output).all()
                and np.isfinite(mgx_cumulative_output).all()
            ):
                raise RuntimeError(f"non-finite output at segment {index}")

            profile_path = Path(mgx_session.end_profiling()).resolve(strict=True)
            placement = parse_profile(profile_path)
            if not placement["passed"]:
                raise RuntimeError(f"strict provider placement failed at segment {index}: {placement}")
            profiles.append({"index": index, "label": label, **identity(profile_path), **placement})

            local_comparison = compare(cpu_output, mgx_local_output, with_classes=index == 24)
            cumulative_comparison = compare(cpu_output, mgx_cumulative_output, with_classes=index == 24)
            propagation_comparison = compare(mgx_local_output, mgx_cumulative_output, with_classes=index == 24)
            stage_rows.append(
                {
                    "index": index,
                    "label": label,
                    "input_drift_cpu_vs_cumulative_migraphx": input_drift,
                    "local_backend_cpu_vs_migraphx_same_cpu_input": local_comparison,
                    "cumulative_cpu_vs_migraphx": cumulative_comparison,
                    "propagation_migraphx_cpu_input_vs_migraphx_cumulative_input": propagation_comparison,
                    "diagnostic_seconds": {
                        "cpu": cpu_seconds,
                        "migraphx_local": mgx_local_seconds,
                        "migraphx_cumulative": mgx_cumulative_seconds,
                    },
                    "profile": profiles[-1],
                }
            )

            cpu_current = cpu_output
            mgx_current = mgx_cumulative_output
            if index in RETAIN_AFTER:
                cpu_retained[output_name] = cpu_output
                mgx_retained[output_name] = mgx_cumulative_output
            del cpu_session, mgx_session
            gc.collect()
            print(
                f"{index:02d} {label}: local_mae={local_comparison['mae']:.9g} "
                f"local_max={local_comparison['max_abs']:.9g} "
                f"cumulative_mae={cumulative_comparison['mae']:.9g}",
                flush=True,
            )

        final_cpu_vs_frozen = compare(frozen_cpu_logits, cpu_current, with_classes=True)
        final_mgx_vs_frozen = compare(frozen_migraphx_logits, mgx_current, with_classes=True)
        final_cpu_vs_mgx = compare(cpu_current, mgx_current, with_classes=True)
        reproduction_gates = {
            "cpu_segmented_reproduces_frozen_cpu_logits_exactly": final_cpu_vs_frozen["exact_equal"],
            "migraphx_segmented_reproduces_frozen_migraphx_logits_mae_le_1e_6": final_mgx_vs_frozen["mae"] <= 1e-6,
            "migraphx_segmented_reproduces_frozen_migraphx_logits_max_le_1e_5": final_mgx_vs_frozen["max_abs"] <= 1e-5,
            "all_25_profiles_migraphx_positive_cpu_zero": len(profiles) == 25 and all(item["passed"] for item in profiles),
        }
        if not all(reproduction_gates.values()):
            raise RuntimeError(f"frozen result reproduction failed: {reproduction_gates}")

        first_mae = next(
            (row["index"] for row in stage_rows if row["local_backend_cpu_vs_migraphx_same_cpu_input"]["mae"] > 1e-5),
            None,
        )
        first_max = next(
            (row["index"] for row in stage_rows if row["local_backend_cpu_vs_migraphx_same_cpu_input"]["max_abs"] > 1e-4),
            None,
        )
        ranked = sorted(
            (
                {
                    "index": row["index"],
                    "label": row["label"],
                    "local_mae": row["local_backend_cpu_vs_migraphx_same_cpu_input"]["mae"],
                    "local_max_abs": row["local_backend_cpu_vs_migraphx_same_cpu_input"]["max_abs"],
                    "local_relative_l2": row["local_backend_cpu_vs_migraphx_same_cpu_input"]["relative_l2"],
                    "cumulative_mae": row["cumulative_cpu_vs_migraphx"]["mae"],
                }
                for row in stage_rows
            ),
            key=lambda item: item["local_mae"],
            reverse=True,
        )

        csv_path = args.output_dir / "per_segment_numeric_localization.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "index",
                    "label",
                    "local_mae",
                    "local_max_abs",
                    "local_rmse",
                    "local_relative_l2",
                    "cumulative_mae",
                    "cumulative_max_abs",
                    "cumulative_relative_l2",
                    "propagation_mae",
                    "propagation_max_abs",
                ],
            )
            writer.writeheader()
            for row in stage_rows:
                local = row["local_backend_cpu_vs_migraphx_same_cpu_input"]
                cumulative = row["cumulative_cpu_vs_migraphx"]
                propagation = row["propagation_migraphx_cpu_input_vs_migraphx_cumulative_input"]
                writer.writerow(
                    {
                        "index": row["index"],
                        "label": row["label"],
                        "local_mae": local["mae"],
                        "local_max_abs": local["max_abs"],
                        "local_rmse": local["rmse"],
                        "local_relative_l2": local["relative_l2"],
                        "cumulative_mae": cumulative["mae"],
                        "cumulative_max_abs": cumulative["max_abs"],
                        "cumulative_relative_l2": cumulative["relative_l2"],
                        "propagation_mae": propagation["mae"],
                        "propagation_max_abs": propagation["max_abs"],
                    }
                )

        logits_path = args.output_dir / "final_cpu_migraphx_and_frozen_logits.npz"
        np.savez_compressed(
            logits_path,
            cpu_segmented_logits=cpu_current,
            migraphx_segmented_logits=mgx_current,
            frozen_cpu_logits=frozen_cpu_logits,
            frozen_migraphx_logits=frozen_migraphx_logits,
        )
        strict_gates = {
            "mae_le_1e_3": final_cpu_vs_mgx["mae"] <= 1e-3,
            "max_abs_le_5e_2": final_cpu_vs_mgx["max_abs"] <= 5e-2,
            "pixel_class_agreement_ge_99_9pct": final_cpu_vs_mgx["pixel_class_agreement"] >= 0.999,
        }
        result.update(
            {
                "status": "diagnostic_completed",
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "providers": ort.get_available_providers(),
                    "migraphx_cache_mode": "load_only",
                    "intermediate_transport": "numpy_host_staging_for_numeric_diagnosis_only",
                },
                "identities": identities,
                "segments": stage_rows,
                "profiles": profiles,
                "localization": {
                    "descriptive_thresholds": {"mae": 1e-5, "max_abs": 1e-4},
                    "first_segment_local_mae_gt_1e_5": first_mae,
                    "first_segment_local_max_abs_gt_1e_4": first_max,
                    "ranked_by_local_mae": ranked,
                },
                "final_comparisons": {
                    "segmented_cpu_vs_frozen_cpu": final_cpu_vs_frozen,
                    "segmented_migraphx_vs_frozen_migraphx": final_mgx_vs_frozen,
                    "segmented_cpu_vs_segmented_migraphx": final_cpu_vs_mgx,
                },
                "reproduction_gates": reproduction_gates,
                "strict_numeric_gates": strict_gates,
                "artifacts": {
                    "per_segment_csv": identity(csv_path),
                    "final_logits": identity(logits_path),
                },
                "evidence_boundary": {
                    "diagnostic_thresholds_do_not_replace_registered_end_to_end_gates": True,
                    "timings_are_unwarmed_and_not_performance_evidence": True,
                    "provider_placement_does_not_prove_native_int8_kernels": True,
                    "no_model_or_cache_was_modified": True,
                },
            }
        )
        result["claims"]["all_25_segments_profiled"] = True
        result["claims"]["first_backend_divergence_localized"] = first_mae is not None or first_max is not None
        result["claims"]["strict_end_to_end_numeric_admission"] = all(strict_gates.values())
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "diagnostic_completed" else 2)


if __name__ == "__main__":
    main()
