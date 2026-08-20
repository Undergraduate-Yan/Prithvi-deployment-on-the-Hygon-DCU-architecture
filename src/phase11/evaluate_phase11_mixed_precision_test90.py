#!/usr/bin/env python3
"""Fixed-90 task admission for a manifest-defined M0--M5 K100 candidate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

import phase11_mixed_precision_common as common


BASE_EVALUATOR = (
    16_657,
    "a36e81f1fea3e8e91e0ccadd5b8d04252ad3fbf899b1c6b6e74eb86f6d92a70f",
)
FROZEN_INPUTS = (
    144_553_732,
    "6af3179684c9352dc0686682a084ec3ff60615a8d340f59fc9d8a9300f800b86",
)
FROZEN_MANIFEST = (
    28_349,
    "cc657f02d4968f3a23c435084925ef72aa3017dbe98958c459c91f5039e0e8a5",
)
EXPECTED_SINGLE_SCRIPT_SHA256 = "e69b2870303a92d8d2bb91dcfe950256ad66268c82beb7a11a34f08d03119db0"
EXPECTED_COMMON_MODULE_SHA256 = "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8"


def hash_strings(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--single-result", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "phase11_mixed_precision_test90_v1",
        "status": "failed",
        "claims": {
            "all_25_segments_migraphx_positive_cpu_zero": False,
            "device_resident_intersegment_io_test90": False,
            "task_equivalence_admission_passed": False,
            "historical_m0_strict_failure_overridden": False,
            "performance_admission": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
    }
    task_passed = False
    diagnostic_completed = False
    try:
        ort = common.require_runtime()
        manifest, manifest_identity, rows = common.load_candidate(args.bundle, args.manifest)
        common.validate_onnx_contracts(rows)
        single_identity = common.identity(args.single_result)
        single = common.load_json(args.single_result)
        if single.get("schema") != "phase11_mixed_precision_single_v1":
            raise RuntimeError("single-sample prerequisite schema drift")
        if single.get("status") != "diagnostic_completed":
            raise RuntimeError("single-sample operational admission did not complete")
        if single.get("identities", {}).get("evaluation_script", {}).get("sha256") != EXPECTED_SINGLE_SCRIPT_SHA256:
            raise RuntimeError("single-sample evaluator identity drift")
        if single.get("identities", {}).get("common_module", {}).get("sha256") != EXPECTED_COMMON_MODULE_SHA256:
            raise RuntimeError("single-sample common module identity drift")
        common.validate_result_candidate(single, manifest_identity, manifest["candidate_id"])
        required_single_claims = (
            "onnx_and_boundary_contracts_passed",
            "all_25_segments_migraphx_positive_cpu_zero",
            "device_resident_intersegment_io",
        )
        if not all(single.get("claims", {}).get(key) for key in required_single_claims):
            raise RuntimeError("single-sample operational gates did not pass")
        if single.get("claims", {}).get("historical_m0_strict_failure_overridden"):
            raise RuntimeError("historical strict failure was improperly rewritten")

        project = args.project.resolve(strict=True)
        base_path = project / "phase11_frozen_test90/evaluate_phase11_migraphx_barrier_fp32_90.py"
        helper_path = project / "evaluate_full_test_k100_onnx_int8.py"
        frozen_inputs_path = project / "phase11_frozen_test90/frozen_fp32_test90_inputs.npz"
        frozen_manifest_path = project / "phase11_frozen_test90/manifest.json"
        fp32_result_path = project / "outputs/k100_full_test_fp32/20260807-171433/result.json"
        fp32_csv_path = project / "outputs/k100_full_test_fp32/20260807-171433/per_sample_metrics.csv"
        fp32_tensors_path = project / "outputs/k100_full_test_fp32/20260807-171433/predictions_and_targets.pt"
        identities = {
            "evaluation_script": common.identity(Path(__file__)),
            "common_module": common.identity(Path(common.__file__)),
            "single_result": single_identity,
            "base_evaluator": common.identity(base_path, BASE_EVALUATOR),
            "frozen_inputs": common.identity(frozen_inputs_path, FROZEN_INPUTS),
            "frozen_input_manifest": common.identity(frozen_manifest_path, FROZEN_MANIFEST),
        }
        base = common.import_module(base_path, "phase11_mixed_precision_fixed90_base")
        helper, identities["metric_helper"] = base.load_helper(helper_path)
        fp32_csv_file, frozen_rows = helper.read_locked_fp32_csv(fp32_csv_path)
        dataset, _, _ = base.load_frozen_dataset(
            frozen_inputs_path, frozen_manifest_path, frozen_rows, helper
        )
        frozen = helper.load_fp32_reference(
            fp32_result_path, fp32_tensors_path, fp32_csv_file, dataset
        )
        if len(dataset["raw_inputs"]) != common.EXPECTED_SAMPLES:
            raise RuntimeError("fixed test90 sample count drift")

        current = [
            ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(raw, dtype=np.float32), "cuda", 0)
            for raw in dataset["raw_inputs"]
        ]
        if not all(value.device_name() == "cuda" for value in current):
            raise RuntimeError("fixed90 input device allocation failed")
        retained = {}
        runtime_rows = []
        profiles = []
        started_pipeline = perf_counter()
        for index, row in enumerate(rows):
            started = perf_counter()
            session = common.create_migraphx_session(
                ort, row, args.output_dir / f"segment_{index:02d}_test90_profile"
            )
            creation_seconds = perf_counter() - started
            inputs = session.get_inputs()
            output_names = [item.name for item in session.get_outputs()]
            next_values = []
            started = perf_counter()
            for sample_index in range(common.EXPECTED_SAMPLES):
                binding = session.io_binding()
                if index < 24:
                    if len(inputs) != 1:
                        raise RuntimeError(f"encoder input count drift at segment {index}")
                    binding.bind_ortvalue_input(inputs[0].name, current[sample_index])
                else:
                    if {item.name for item in inputs} != set(retained):
                        raise RuntimeError("head retained-feature contract drift")
                    for item in inputs:
                        binding.bind_ortvalue_input(item.name, retained[item.name][sample_index])
                for name in output_names:
                    binding.bind_output(name, "cuda", 0)
                binding.synchronize_inputs()
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                output_values = binding.get_outputs()
                if len(output_values) != len(output_names):
                    raise RuntimeError(f"output count drift: {index}/{sample_index}")
                if not all(value.device_name() == "cuda" for value in output_values):
                    raise RuntimeError(f"host output detected: {index}/{sample_index}")
                output_map = dict(zip(output_names, output_values, strict=True))
                next_values.append(common.choose_primary_output(index, output_map))
                del binding
            inference_seconds = perf_counter() - started
            current = next_values
            if index in common.RETAIN_AFTER:
                retained[output_names[0]] = list(current)
            profile_path = Path(session.end_profiling()).resolve(strict=True)
            placement = common.parse_profile(profile_path)
            if not placement["passed"]:
                raise RuntimeError(f"provider placement failed at segment {index}: {placement}")
            runtime_rows.append(
                {
                    "index": index,
                    "label": row["label"],
                    "precision": row["precision"],
                    "session_creation_seconds": creation_seconds,
                    "test90_inference_seconds_diagnostic_only": inference_seconds,
                    "output_names": output_names,
                    "all_90_outputs_device": all(value.device_name() == "cuda" for value in current),
                    "provider_event_counts": placement["provider_event_counts"],
                }
            )
            profiles.append({"index": index, **common.identity(profile_path), **placement})
            del session
            gc.collect()
            print(f"test90 segment {index + 1:02d}/25 passed", flush=True)
        diagnostic_pipeline_seconds = perf_counter() - started_pipeline

        logits_rows = []
        nonfinite_output_count = 0
        for sample_index, value in enumerate(current):
            logits = np.ascontiguousarray(value.numpy(), dtype=np.float32)
            if logits.shape != (1, 2, 224, 224):
                raise RuntimeError(f"logit shape drift at sample {sample_index}: {logits.shape}")
            if not np.isfinite(logits).all():
                nonfinite_output_count += 1
            logits_rows.append(helper.validate_logits(logits, f"mixed candidate sample {sample_index}"))

        confusion = np.zeros((2, 2), dtype=np.int64)
        predictions = []
        per_sample_rows = []
        total_loss = 0.0
        total_valid = 0
        for index, (logits, target) in enumerate(
            zip(logits_rows, dataset["targets"], strict=True)
        ):
            prediction = np.argmax(logits, axis=1).astype(np.uint8)
            sample_confusion, valid_pixels = helper.sample_confusion(prediction, target)
            confusion += sample_confusion
            total_valid += valid_pixels
            total_loss += helper.cross_entropy_sum(logits, target)
            predictions.append(np.ascontiguousarray(prediction[0]))
            per_sample_rows.append(
                {
                    "sample_index": index,
                    "sample_id": dataset["sample_ids"][index],
                    "raw_scaled_fp32_sha256": dataset["manifest_rows"][index][
                        "raw_scaled_fp32_sha256"
                    ],
                    "target_int64_sha256": dataset["manifest_rows"][index]["target_int64_sha256"],
                    "logits_fp32_sha256": helper.array_sha256(logits),
                    "prediction_uint8_sha256": helper.array_sha256(prediction),
                    "valid_pixels": valid_pixels,
                }
            )
        if total_valid != common.EXPECTED_VALID_PIXELS:
            raise RuntimeError(f"fixed valid-pixel count drift: {total_valid}")
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
        agreement = float(
            np.mean(prediction_array[valid_mask] == frozen["predictions"][valid_mask])
        )
        changed_valid_pixels = int(
            np.count_nonzero(prediction_array[valid_mask] != frozen["predictions"][valid_mask])
        )
        task_gates = {
            "sample_count_eq_90": len(predictions) == common.EXPECTED_SAMPLES,
            "valid_pixels_eq_3927398": total_valid == common.EXPECTED_VALID_PIXELS,
            "nonfinite_outputs_eq_0": nonfinite_output_count
            <= common.TASK_GATES["nonfinite_outputs_max"],
            "all_25_profiles_migraphx_positive_cpu_zero": len(profiles) == 25
            and all(item["passed"] for item in profiles),
            "all_intersegment_outputs_device": len(runtime_rows) == 25
            and all(item["all_90_outputs_device"] for item in runtime_rows),
            "miou_drop_le_0_5pp": deltas_pp["miou"]
            >= -common.TASK_GATES["miou_drop_pp_max"],
            "water_iou_drop_le_1_0pp": deltas_pp["water_iou"]
            >= -common.TASK_GATES["water_iou_drop_pp_max"],
            "corrected_boundary_water_iou_drop_le_1_0pp": deltas_pp[
                "boundary_water_iou_corrected"
            ]
            >= -common.TASK_GATES["corrected_boundary_water_iou_drop_pp_max"],
            "prediction_agreement_vs_fp32_ge_99_5pct": agreement
            >= common.TASK_GATES["prediction_agreement_min"],
        }
        task_passed = all(task_gates.values())
        diagnostic_completed = True
        tensors_path = args.output_dir / "predictions_and_targets.npz"
        np.savez_compressed(
            tensors_path,
            predictions=prediction_array,
            targets=target_array,
            sample_ids=np.asarray(dataset["sample_ids"]),
        )
        csv_identity = common.write_csv(
            args.output_dir / "per_sample_metrics.csv", per_sample_rows
        )
        result.update(
            {
                "status": "diagnostic_completed",
                "candidate_lineage": {
                    "candidate_id": manifest["candidate_id"],
                    "manifest": manifest_identity,
                    "bundle": str(args.bundle.resolve(strict=True)),
                    "fp16_backbone_blocks": manifest["fp16_backbone_blocks"],
                    "int8_backbone_blocks": manifest["int8_backbone_blocks"],
                },
                "identities": identities,
                "dataset": {
                    "samples": len(dataset["sample_ids"]),
                    "valid_pixels": total_valid,
                    "frozen_inputs": identities["frozen_inputs"],
                    "frozen_input_manifest": identities["frozen_input_manifest"],
                    "sample_ids_sha256": hash_strings([str(x) for x in dataset["sample_ids"]]),
                    "targets_array_sha256": common.array_sha256(target_array),
                },
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "intersegment_transport": "direct_device_OrtValue_IOBinding",
                    "host_intermediate_copies": 0,
                    "pipeline_test90_seconds_diagnostic_only": diagnostic_pipeline_seconds,
                },
                "segments": runtime_rows,
                "profiles": profiles,
                "metrics": metrics,
                "frozen_fp32_metrics": frozen_metrics,
                "deltas_vs_frozen_fp32_pp": deltas_pp,
                "valid_prediction_agreement_vs_frozen_fp32": agreement,
                "changed_valid_pixels_vs_frozen_fp32": changed_valid_pixels,
                "nonfinite_output_count": nonfinite_output_count,
                "task_thresholds": common.TASK_GATES,
                "task_gates": task_gates,
                "task_equivalence_admission_passed": task_passed,
                "artifacts": {
                    "per_sample_metrics": csv_identity,
                    "predictions_and_targets": common.identity(tensors_path),
                },
                "evidence_boundary": {
                    "strict_numeric_result_is_not_a_task_gate": True,
                    "historical_m0_strict_failure_is_immutable": True,
                    "task_pass_does_not_retroactively_pass_strict_numeric_equivalence": True,
                    "reported_pipeline_seconds_are_diagnostic_not_benchmark_latency": True,
                    "provider_profiles_do_not_prove_native_int8_kernels": True,
                },
            }
        )
        result["claims"]["all_25_segments_migraphx_positive_cpu_zero"] = task_gates[
            "all_25_profiles_migraphx_positive_cpu_zero"
        ]
        result["claims"]["device_resident_intersegment_io_test90"] = task_gates[
            "all_intersegment_outputs_device"
        ]
        result["claims"]["task_equivalence_admission_passed"] = task_passed
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    common.json_dump(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if not diagnostic_completed:
        raise SystemExit(2)
    raise SystemExit(0 if task_passed else 3)


if __name__ == "__main__":
    main()
