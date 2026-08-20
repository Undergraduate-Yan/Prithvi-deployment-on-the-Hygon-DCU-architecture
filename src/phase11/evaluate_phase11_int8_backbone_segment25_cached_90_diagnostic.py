#!/usr/bin/env python3
"""Evaluate the cached 25-segment INT8-backbone pipeline on frozen test90.

The whole-graph strict numerical gate already failed.  This program therefore
produces a task-level diagnostic only.  Passing the task thresholds does not
retroactively authorize strict numerical admission, native-INT8, deployment,
device-resident-pipeline, or performance claims.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np


SOURCE = (368_047_490, "edd2c3e69b64986dc6d0b78fe32230f9a9f7eb0396c2d77fdcb6bf4c59506a7a")
REPORT = (21_342, "e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e")
PARITY = (14_138, "5c66f94a447a3d08fe0024686e2a0dc2d8e74f16d0f10951cc49e4ed81568a6f")
SINGLE = (42_574, "4e010e3e1e2e02b4c845f25b16681455e8f6da4cfca9b65b401f5f6555edaa4f")
BASE = (16_657, "a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f")
FROZEN_INPUTS = (144_553_732, "6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86")
FROZEN_MANIFEST = (28_349, "cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5")
EXPECTED_SAMPLES = 90
EXPECTED_VALID = 3_927_398
LABELS = tuple(f"encoder_block_{i:02d}" for i in range(24)) + ("upernet_decoder_head",)
RETAIN_AFTER = (5, 11, 17, 23)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lock(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_module(path: Path, name: str, expected: tuple[int, str] | None = None):
    item = lock(path, expected)
    spec = importlib.util.spec_from_file_location(name, item["path"])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {item}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, item


def session_options(ort, profile_prefix: Path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
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
        "migraphx_positive": int(counts.get(MGX, 0)) > 0,
        "cpu_zero": int(counts.get(CPU, 0)) == 0,
    }


def write_csv(path: Path, rows: list[dict]) -> dict:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return lock(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--single-result", type=Path, required=True)
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--fp32-result", type=Path, required=True)
    parser.add_argument("--fp32-csv", type=Path, required=True)
    parser.add_argument("--fp32-tensors", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--frozen-input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    result = {
        "status": "failed",
        "variant": "int8_backbone_compat_25_static_batch1_cached_host_staged_test90",
        "claims": {
            "all_25_segments_strict_migraphx_placement": False,
            "strict_end_to_end_numeric_admission": False,
            "task_utility_diagnostic_passed": False,
            "device_resident_intersegment_io": False,
            "performance": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("official ORT 1.19.2/MIGraphX runtime identity drift")

        source = args.root / "build/candidate.onnx"
        segment_root = args.root / "segment25_static_build_cpu"
        report_path = segment_root / "segment_build_report.json"
        identities = {
            "source": lock(source, SOURCE),
            "segment_report": lock(report_path, REPORT),
            "cpu_sequential_parity": lock(segment_root / "cpu_sequential_parity.json", PARITY),
            "failed_single_end_to_end": lock(args.single_result, SINGLE),
            "frozen_inputs": lock(args.frozen_inputs, FROZEN_INPUTS),
            "frozen_input_manifest": lock(args.frozen_input_manifest, FROZEN_MANIFEST),
        }
        single = json.loads(args.single_result.read_text(encoding="utf-8"))
        if single.get("status") != "failed":
            raise RuntimeError("frozen strict single-sample numeric failure must remain failed")
        single_claims = single.get("claims", {})
        if not single_claims.get("all_25_segments_strict_migraphx_placement"):
            raise RuntimeError("frozen single-sample placement did not pass")
        if single_claims.get("strict_end_to_end_numeric_admission"):
            raise RuntimeError("frozen single-sample numeric failure was rewritten")

        base, identities["base_evaluator"] = load_module(
            args.base_evaluator, "phase11_segment25_test90_base", BASE
        )
        helper, identities["metric_helper"] = base.load_helper(args.helper)
        fp32_csv_file, frozen_rows = helper.read_locked_fp32_csv(args.fp32_csv)
        dataset, _, _ = base.load_frozen_dataset(
            args.frozen_inputs, args.frozen_input_manifest, frozen_rows, helper
        )
        frozen = helper.load_fp32_reference(
            args.fp32_result, args.fp32_tensors, fp32_csv_file, dataset
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        segment_rows = report["segments"]
        if len(segment_rows) != 25 or [row["label"] for row in segment_rows] != list(LABELS):
            raise RuntimeError("segment manifest/order drift")

        models: list[Path] = []
        caches: list[Path] = []
        frozen_cache_evidence: list[dict] = []
        for index, row in enumerate(segment_rows):
            model = segment_root / "models" / Path(row["path"]).name
            identities[row["label"]] = lock(model, (int(row["size_bytes"]), row["sha256"]))
            base_dir = (
                args.root / "segment25_static_cache54g_segment0"
                if index == 0
                else args.root / "segment25_static_remaining_caches" / f"segment_{index:02d}"
            )
            frozen_result_path = base_dir / "output/result.json"
            frozen_result = json.loads(frozen_result_path.read_text(encoding="utf-8"))
            if frozen_result.get("target_segment_index") != index:
                raise RuntimeError(f"frozen segment index drift: {index}")
            if frozen_result.get("target_segment_label") != LABELS[index]:
                raise RuntimeError(f"frozen segment label drift: {index}")
            counts = frozen_result.get("profile", {}).get("provider_event_counts", {})
            if int(counts.get(MGX, 0)) <= 0 or int(counts.get(CPU, 0)) != 0:
                raise RuntimeError(f"frozen strict placement failed: {index}")
            cache = base_dir / f"segment_{index:02d}.mxr"
            cache_meta = frozen_result["compiled_cache"]
            lock(cache, (int(cache_meta["size_bytes"]), cache_meta["sha256"]))
            models.append(model)
            caches.append(cache)
            frozen_cache_evidence.append({
                "index": index,
                "result": lock(frozen_result_path),
                "cache": cache_meta,
            })

        if len(dataset["raw_inputs"]) != EXPECTED_SAMPLES or len(dataset["targets"]) != EXPECTED_SAMPLES:
            raise RuntimeError("frozen test90 sample count drift")
        values = [np.ascontiguousarray(value, dtype=np.float32) for value in dataset["raw_inputs"]]
        retained: dict[str, list[np.ndarray]] = {}
        runtime_rows: list[dict] = []
        profiles: list[dict] = []
        pipeline_started = perf_counter()

        for index, (model, cache) in enumerate(zip(models, caches, strict=True)):
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve())
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve())
            prefix = args.output_dir / f"segment_{index:02d}_test90_profile"
            started = perf_counter()
            session = ort.InferenceSession(
                str(model),
                sess_options=session_options(ort, prefix),
                providers=[(MGX, {"device_id": 0})],
            )
            session.disable_fallback()
            creation_seconds = perf_counter() - started
            registered_providers = session.get_providers()
            if not registered_providers or registered_providers[0] != MGX:
                raise RuntimeError(f"provider priority drift at segment {index}: {registered_providers}")
            inputs = session.get_inputs()
            output_name = session.get_outputs()[0].name
            started = perf_counter()
            next_values: list[np.ndarray] = []
            if index < 24:
                if len(inputs) != 1:
                    raise RuntimeError(f"encoder input contract drift at segment {index}")
                input_name = inputs[0].name
                for sample_index, value in enumerate(values):
                    output = np.asarray(session.run([output_name], {input_name: value})[0])
                    if output.shape != (1, 197, 1024) or output.dtype != np.float32:
                        raise RuntimeError(f"encoder output contract drift: {index}/{sample_index}")
                    if not np.isfinite(output).all():
                        raise RuntimeError(f"non-finite encoder output: {index}/{sample_index}")
                    next_values.append(np.ascontiguousarray(output))
                values = next_values
                if index in RETAIN_AFTER:
                    retained[output_name] = [value.copy() for value in values]
            else:
                if len(inputs) != 4 or set(item.name for item in inputs) != set(retained):
                    raise RuntimeError("decoder-head retained-feature contract drift")
                for sample_index in range(EXPECTED_SAMPLES):
                    feeds = {item.name: retained[item.name][sample_index] for item in inputs}
                    output = np.asarray(session.run([output_name], feeds)[0])
                    output = helper.validate_logits(output, f"segment25 test90 sample {sample_index}")
                    next_values.append(np.ascontiguousarray(output))
                values = next_values
            inference_seconds = perf_counter() - started
            profile_path = Path(session.end_profiling()).resolve(strict=True)
            placement = parse_profile(profile_path)
            if not placement["migraphx_positive"] or not placement["cpu_zero"]:
                raise RuntimeError(f"live strict placement failed at segment {index}: {placement}")
            runtime_rows.append({
                "index": index,
                "label": LABELS[index],
                "session_creation_seconds": creation_seconds,
                "test90_inference_seconds_diagnostic": inference_seconds,
                "provider_event_counts": placement["provider_event_counts"],
                "registered_providers": registered_providers,
                "output_name": output_name,
                "output_shape": list(values[0].shape),
            })
            profiles.append({"index": index, **lock(profile_path)})
            print(
                f"segment {index + 1:02d}/25 {LABELS[index]}: "
                f"90/90, MGX={placement['provider_event_counts'].get(MGX, 0)}, "
                f"CPU={placement['provider_event_counts'].get(CPU, 0)}",
                flush=True,
            )
            del session
            gc.collect()

        pipeline_seconds = perf_counter() - pipeline_started
        logits_rows = values
        if len(logits_rows) != EXPECTED_SAMPLES:
            raise RuntimeError("final logits sample count drift")

        confusion = np.zeros((2, 2), dtype=np.int64)
        predictions: list[np.ndarray] = []
        per_sample_rows: list[dict] = []
        total_loss = 0.0
        total_valid = 0
        for index, (logits, target) in enumerate(zip(logits_rows, dataset["targets"], strict=True)):
            pred = np.argmax(logits, axis=1).astype(np.uint8)
            sample_confusion, valid = helper.sample_confusion(pred, target)
            confusion += sample_confusion
            total_valid += valid
            total_loss += helper.cross_entropy_sum(logits, target)
            predictions.append(np.ascontiguousarray(pred[0]))
            per_sample_rows.append({
                "sample_index": index,
                "sample_id": dataset["sample_ids"][index],
                "raw_scaled_fp32_sha256": dataset["manifest_rows"][index]["raw_scaled_fp32_sha256"],
                "target_int64_sha256": dataset["manifest_rows"][index]["target_int64_sha256"],
                "logits_fp32_sha256": helper.array_sha256(logits),
                "prediction_uint8_sha256": helper.array_sha256(pred),
                "valid_pixels": valid,
            })

        if total_valid != EXPECTED_VALID:
            raise RuntimeError(f"valid-pixel drift: {total_valid}")
        prediction_array = np.stack(predictions).astype(np.uint8, copy=False)
        target_array = np.concatenate(dataset["targets"], axis=0).astype(np.int64, copy=False)
        metrics = helper.confusion_metrics(confusion)
        metrics["loss"] = float(total_loss / total_valid)
        metrics["valid_pixels"] = total_valid
        boundary = helper.corrected_boundary_counts(prediction_array, target_array, thickness=2)
        metrics["boundary_water_iou_corrected"] = boundary["water_iou_corrected"]

        frozen_metrics = frozen["metrics"]
        metric_keys = ("miou", "water_iou", "pixel_accuracy", "boundary_water_iou_corrected")
        deltas_pp = {
            key: 100.0 * (float(metrics[key]) - float(frozen_metrics[key])) for key in metric_keys
        }
        valid_mask = (target_array >= 0) & (target_array < 2)
        frozen_predictions = frozen["predictions"]
        agreement = float(np.mean(prediction_array[valid_mask] == frozen_predictions[valid_mask]))
        changed_pixels = int(np.count_nonzero(prediction_array[valid_mask] != frozen_predictions[valid_mask]))

        gates = {
            "sample_count_eq_90": len(predictions) == EXPECTED_SAMPLES,
            "valid_pixels_exact": total_valid == EXPECTED_VALID,
            "all_outputs_finite": True,
            "all_25_live_profiles_migraphx_positive_cpu_zero": len(runtime_rows) == 25,
            "miou_drop_le_2_0pp": deltas_pp["miou"] >= -2.0,
            "water_iou_drop_le_3_0pp": deltas_pp["water_iou"] >= -3.0,
            "boundary_drop_le_3_0pp": deltas_pp["boundary_water_iou_corrected"] >= -3.0,
            "valid_prediction_agreement_ge_98pct": agreement >= 0.98,
        }
        task_gate_passed = all(gates.values())

        tensors_path = args.output_dir / "predictions_and_targets.npz"
        np.savez_compressed(
            tensors_path,
            predictions=prediction_array,
            targets=target_array,
            sample_ids=np.asarray(dataset["sample_ids"]),
        )
        csv_file = write_csv(args.output_dir / "per_sample_metrics.csv", per_sample_rows)
        result.update({
            "status": "diagnostic_completed",
            "identities": identities,
            "frozen_cache_evidence": frozen_cache_evidence,
            "runtime": {
                "onnxruntime": ort.__version__,
                "provider_priority_primary": MGX,
                "cpu_ep_may_be_registered_but_session_and_python_fallback_are_disabled": True,
                "intersegment_transport": "numpy_host_staging",
                "static_batch": 1,
                "pipeline_test90_seconds_diagnostic_only": pipeline_seconds,
            },
            "segments": runtime_rows,
            "profiles": profiles,
            "evaluated_samples": len(predictions),
            "metrics": metrics,
            "confusion_matrix": confusion.tolist(),
            "frozen_fp32_metrics": {key: frozen_metrics[key] for key in frozen_metrics},
            "deltas_vs_frozen_fp32_pp": deltas_pp,
            "valid_prediction_agreement_vs_frozen_fp32": agreement,
            "changed_valid_pixels_vs_frozen_fp32": changed_pixels,
            "gates": gates,
            "task_utility_diagnostic_passed": task_gate_passed,
            "evidence_boundary": {
                "strict_single_logits_numeric_gate_remains_failed": True,
                "task_gate_does_not_retroactively_pass_strict_numeric_admission": True,
                "host_staged_24_boundaries_are_not_a_deployment_pipeline": True,
                "timings_are_diagnostic_and_not_a_performance_benchmark": True,
                "profiles_prove_provider placement_not_native_int8_kernels": True,
            },
            "artifacts": {
                "per_sample_csv": csv_file,
                "predictions_and_targets": lock(tensors_path),
            },
        })
        result["claims"]["all_25_segments_strict_migraphx_placement"] = True
        result["claims"]["task_utility_diagnostic_passed"] = task_gate_passed
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "diagnostic_completed" else 2)


if __name__ == "__main__":
    main()
