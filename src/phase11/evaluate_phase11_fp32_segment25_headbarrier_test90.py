#!/usr/bin/env python3
"""Fixed-test90 task admission for the FP32 25-segment fpn4-barrier candidate."""
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


SEGMENT_REPORT = (24_959, "c94692d130a0eb924290b3d56f790e166535fa3049fd9f3e72ef1fe16085c90f")
BASE_EVALUATOR = (16_657, "a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f")
HELPER = (59_386, "cd746e45ac45e8da148e6481e318c3701473088c2bba2397bbf85e6917517435")
FROZEN_INPUTS = (144_553_732, "6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86")
FROZEN_MANIFEST = (28_349, "cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5")
EXPECTED_SAMPLES = 90
EXPECTED_VALID = 3_927_398
LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24)) + ("upernet_decoder_head_fpn4barrier",)
RETAIN_AFTER = (5, 11, 17, 23)
BARRIER = "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_module(path: Path, name: str, expected: tuple[int, str]):
    identity = lock(path, expected)
    spec = importlib.util.spec_from_file_location(name, identity["path"])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, identity


def options(ort, profile: Path):
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    value.intra_op_num_threads = 4
    value.inter_op_num_threads = 1
    value.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    value.enable_profiling = True
    value.profile_file_prefix = str(profile)
    return value


def parse_profile(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return {
        "provider_event_counts": dict(counts),
        "passed": int(counts.get(MGX, 0)) > 0 and int(counts.get(CPU, 0)) == 0,
    }


def write_csv(path: Path, rows: list[dict]) -> dict:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return lock(path)


def validate_candidate(candidate: Path, report_path: Path) -> tuple[dict, dict]:
    report_identity = lock(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "phase11_fp32_segment25_head_fpn4barrier_build_v1":
        raise RuntimeError("candidate build schema drift")
    if report.get("status") != "passed" or not all(report.get("gates", {}).values()):
        raise RuntimeError("candidate build prerequisite failed")
    meta = report.get("candidate", {})
    candidate_identity = lock(candidate, (int(meta["size_bytes"]), str(meta["sha256"])))
    return candidate_identity, report_identity


def cache_paths(root: Path, index: int) -> tuple[Path, Path]:
    base = root / "cache_ln_segment00" if index == 0 else root / "cache_ln_remaining" / f"segment_{index:02d}"
    return base / f"segment_{index:02d}.mxr", base / "output" / "result.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--head-cache", type=Path, required=True)
    parser.add_argument("--single-result", type=Path, required=True)
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--fp32-result", type=Path, required=True)
    parser.add_argument("--fp32-csv", type=Path, required=True)
    parser.add_argument("--fp32-tensors", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "phase11_fp32_segment25_headbarrier_test90_v1",
        "status": "failed",
        "claims": {
            "all_25_segments_strict_migraphx": False,
            "device_resident_intersegment_io_test90": False,
            "task_equivalence_vs_frozen_fp32": False,
            "performance_reference_eligible": False,
            "performance_measured": False,
            "monolithic_deployment_ready": False,
        },
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("runtime identity drift")
        report_path = args.root / "build_ln" / "segment25_report_deconv.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report.get("segments", [])
        if len(rows) != 25 or [row.get("label") for row in rows[:24]] != list(LABELS[:24]):
            raise RuntimeError("segment topology drift")
        candidate_identity, build_identity = validate_candidate(args.candidate, args.build_report)
        single_identity = lock(args.single_result)
        single = json.loads(args.single_result.read_text(encoding="utf-8"))
        if single.get("schema") != "phase11_fp32_segment25_headbarrier_single_v1":
            raise RuntimeError("single result schema drift")
        if single.get("status") != "diagnostic_completed" or not single.get("claims", {}).get("task90_eligible"):
            raise RuntimeError("single-sample task90 prerequisite failed")
        cache_meta = single.get("new_head_cache", {})
        head_cache_identity = lock(args.head_cache, (int(cache_meta["size_bytes"]), str(cache_meta["sha256"])))
        identities = {
            "segment_report": lock(report_path, SEGMENT_REPORT),
            "candidate": candidate_identity,
            "candidate_build_report": build_identity,
            "single_result": single_identity,
            "head_cache": head_cache_identity,
            "frozen_inputs": lock(args.frozen_inputs, FROZEN_INPUTS),
            "frozen_manifest": lock(args.frozen_manifest, FROZEN_MANIFEST),
            "metric_helper": lock(args.helper, HELPER),
        }

        base, identities["base_evaluator"] = load_module(
            args.base_evaluator, "phase11_fp32_headbarrier_test90_base", BASE_EVALUATOR
        )
        helper, helper_identity = base.load_helper(args.helper)
        if helper_identity["sha256"] != identities["metric_helper"]["sha256"]:
            raise RuntimeError("helper identity disagreement")
        fp32_csv_file, frozen_rows = helper.read_locked_fp32_csv(args.fp32_csv)
        dataset, _, _ = base.load_frozen_dataset(
            args.frozen_inputs, args.frozen_manifest, frozen_rows, helper
        )
        frozen = helper.load_fp32_reference(
            args.fp32_result, args.fp32_tensors, fp32_csv_file, dataset
        )
        if len(dataset["raw_inputs"]) != EXPECTED_SAMPLES:
            raise RuntimeError("frozen test90 sample count drift")

        models: list[Path] = []
        caches: list[Path] = []
        frozen_cache_evidence = []
        for index, row in enumerate(rows[:24]):
            model = args.root / "build_ln" / "models" / Path(row["path"]).name
            identities[f"model_{index:02d}"] = lock(model, (int(row["size_bytes"]), str(row["sha256"])))
            cache, cache_result_path = cache_paths(args.root, index)
            cache_result = json.loads(cache_result_path.read_text(encoding="utf-8"))
            old_cache_meta = cache_result.get("compiled_cache", {})
            cache_identity = lock(cache, (int(old_cache_meta["size_bytes"]), str(old_cache_meta["sha256"])))
            counts = cache_result.get("profile", {}).get("provider_event_counts", {})
            if int(counts.get(MGX, 0)) <= 0 or int(counts.get(CPU, 0)) != 0:
                raise RuntimeError(f"frozen cache placement prerequisite failed at segment {index}")
            models.append(model)
            caches.append(cache)
            frozen_cache_evidence.append({"index": index, "cache": cache_identity, "result": lock(cache_result_path)})
        models.append(args.candidate)
        caches.append(args.head_cache)

        current = [
            ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(raw, dtype=np.float32), "cuda", 0)
            for raw in dataset["raw_inputs"]
        ]
        if not all(value.device_name() == "cuda" for value in current):
            raise RuntimeError("initial device allocation failed")
        retained: dict[str, list] = {}
        profiles = []
        runtime_rows = []
        pipeline_started = perf_counter()
        for index, (model, cache) in enumerate(zip(models, caches, strict=True)):
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve())
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve())
            started = perf_counter()
            session = ort.InferenceSession(
                str(model),
                sess_options=options(ort, args.output_dir / f"segment_{index:02d}_profile"),
                providers=[(MGX, {"device_id": 0})],
            )
            session.disable_fallback()
            creation_seconds = perf_counter() - started
            input_items = session.get_inputs()
            output_names = [item.name for item in session.get_outputs()]
            next_values = []
            all_bound_outputs_device = True
            started = perf_counter()
            for sample_index in range(EXPECTED_SAMPLES):
                binding = session.io_binding()
                if index < 24:
                    if len(input_items) != 1:
                        raise RuntimeError(f"encoder input contract drift at segment {index}")
                    binding.bind_ortvalue_input(input_items[0].name, current[sample_index])
                else:
                    if set(item.name for item in input_items) != set(retained):
                        raise RuntimeError("head retained-feature contract drift")
                    for item in input_items:
                        binding.bind_ortvalue_input(item.name, retained[item.name][sample_index])
                for name in output_names:
                    binding.bind_output(name, "cuda", 0)
                binding.synchronize_inputs()
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                output_values = binding.get_outputs()
                output_map = dict(zip(output_names, output_values, strict=True))
                all_bound_outputs_device = all_bound_outputs_device and all(
                    value.device_name() == "cuda" for value in output_values
                )
                value = output_map[output_names[0] if index < 24 else "logits"]
                if value.device_name() != "cuda":
                    raise RuntimeError(f"device logits drift at {index}/{sample_index}")
                next_values.append(value)
                del binding
            inference_seconds = perf_counter() - started
            current = next_values
            if index in RETAIN_AFTER:
                retained[output_names[0]] = list(current)
            profile_path = Path(session.end_profiling()).resolve(strict=True)
            placement = parse_profile(profile_path)
            if not placement["passed"]:
                raise RuntimeError(f"strict placement failed at segment {index}: {placement}")
            runtime_rows.append({
                "index": index,
                "label": LABELS[index],
                "outputs": output_names,
                "session_creation_seconds": creation_seconds,
                "test90_inference_seconds_diagnostic_only": inference_seconds,
                "all_bound_outputs_device": all_bound_outputs_device,
                "provider_event_counts": placement["provider_event_counts"],
            })
            profiles.append({"index": index, **lock(profile_path), **placement})
            del session
            gc.collect()
            print(f"FP32 headbarrier test90 segment {index + 1:02d}/25 passed", flush=True)
        pipeline_seconds = perf_counter() - pipeline_started

        logits_rows = [
            helper.validate_logits(value.numpy(), f"FP32 headbarrier sample {index}")
            for index, value in enumerate(current)
        ]
        confusion = np.zeros((2, 2), dtype=np.int64)
        predictions = []
        per_sample = []
        total_loss = 0.0
        total_valid = 0
        for index, (logits, target) in enumerate(zip(logits_rows, dataset["targets"], strict=True)):
            prediction = np.argmax(logits, axis=1).astype(np.uint8)
            sample_confusion, valid = helper.sample_confusion(prediction, target)
            confusion += sample_confusion
            total_valid += valid
            total_loss += helper.cross_entropy_sum(logits, target)
            predictions.append(np.ascontiguousarray(prediction[0]))
            per_sample.append({
                "sample_index": index,
                "sample_id": dataset["sample_ids"][index],
                "raw_scaled_fp32_sha256": dataset["manifest_rows"][index]["raw_scaled_fp32_sha256"],
                "target_int64_sha256": dataset["manifest_rows"][index]["target_int64_sha256"],
                "logits_fp32_sha256": helper.array_sha256(logits),
                "prediction_uint8_sha256": helper.array_sha256(prediction),
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
            key: 100.0 * (float(metrics[key]) - float(frozen_metrics[key]))
            for key in metric_keys
        }
        valid_mask = (target_array >= 0) & (target_array < 2)
        agreement = float(np.mean(prediction_array[valid_mask] == frozen["predictions"][valid_mask]))
        nonfinite_logits = sum(not np.isfinite(value).all() for value in logits_rows)
        gates = {
            "sample_count_eq_90": len(predictions) == EXPECTED_SAMPLES,
            "valid_pixels_exact": total_valid == EXPECTED_VALID,
            "all_logits_finite": nonfinite_logits == 0,
            "all_25_profiles_migraphx_positive_cpu_zero": len(profiles) == 25,
            "all_intermediate_and_head_outputs_device": all(row["all_bound_outputs_device"] for row in runtime_rows),
            "miou_drop_le_0_5pp": deltas_pp["miou"] >= -0.5,
            "water_iou_drop_le_1_0pp": deltas_pp["water_iou"] >= -1.0,
            "boundary_drop_le_1_0pp": deltas_pp["boundary_water_iou_corrected"] >= -1.0,
            "valid_prediction_agreement_vs_fp32_ge_99_5pct": agreement >= 0.995,
        }
        task_passed = all(gates.values())
        tensors_path = args.output_dir / "predictions_and_targets.npz"
        np.savez_compressed(
            tensors_path,
            predictions=prediction_array,
            targets=target_array,
            sample_ids=np.asarray(dataset["sample_ids"]),
        )
        result.update({
            "status": "passed" if task_passed else "failed",
            "runtime": {
                "onnxruntime": ort.__version__,
                "intersegment_transport": "direct_device_OrtValue",
                "host_intermediate_copies": 0,
                "pipeline_test90_seconds_diagnostic_only": pipeline_seconds,
            },
            "identities": identities,
            "frozen_cache_evidence": frozen_cache_evidence,
            "segments": runtime_rows,
            "profiles": profiles,
            "evaluated_samples": len(predictions),
            "metrics": metrics,
            "confusion_matrix": confusion.tolist(),
            "deltas_vs_frozen_fp32_pp": deltas_pp,
            "valid_prediction_agreement_vs_frozen_fp32": agreement,
            "nonfinite_logits_count": nonfinite_logits,
            "gates": gates,
            "artifacts": {
                "per_sample_csv": write_csv(args.output_dir / "per_sample_metrics.csv", per_sample),
                "predictions_and_targets": lock(tensors_path),
            },
            "evidence_boundary": {
                "timings_are_diagnostic_not_benchmark": True,
                "candidate_is_25_segments_not_monolithic": True,
                "strict_single_logits_result_preserved_separately": True,
                "performance_not_measured": True,
            },
        })
        result["claims"].update({
            "all_25_segments_strict_migraphx": gates["all_25_profiles_migraphx_positive_cpu_zero"],
            "device_resident_intersegment_io_test90": gates["all_intermediate_and_head_outputs_device"],
            "task_equivalence_vs_frozen_fp32": task_passed,
            "performance_reference_eligible": task_passed,
        })
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
