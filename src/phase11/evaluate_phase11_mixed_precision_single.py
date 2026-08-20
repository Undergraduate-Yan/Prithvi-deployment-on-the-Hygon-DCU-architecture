#!/usr/bin/env python3
"""Single-sample K100 admission for one manifest-defined M0--M5 candidate."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import numpy as np

import phase11_mixed_precision_common as common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "phase11_mixed_precision_single_v1",
        "status": "failed",
        "claims": {
            "onnx_and_boundary_contracts_passed": False,
            "all_25_segments_migraphx_positive_cpu_zero": False,
            "device_resident_intersegment_io": False,
            "candidate_strict_numeric_diagnostic_passed": False,
            "historical_m0_strict_failure_overridden": False,
            "task90_admission": False,
            "performance_admission": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
    }
    operational_passed = False
    try:
        ort = common.require_runtime()
        manifest, manifest_identity, rows = common.load_candidate(args.bundle, args.manifest)
        onnx_contracts = common.validate_onnx_contracts(rows)
        raw = common.load_raw_sample(args.sample)
        cpu_logits = common.run_cpu_pipeline(ort, rows, raw)
        if cpu_logits.shape != (1, 2, 224, 224):
            raise RuntimeError(f"CPU logits contract drift: {cpu_logits.shape}")

        current = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        if current.device_name() != "cuda":
            raise RuntimeError("initial input was not allocated on K100")
        retained = {}
        runtime_rows = []
        profiles = []
        for index, row in enumerate(rows):
            session = common.create_migraphx_session(
                ort, row, args.output_dir / f"segment_{index:02d}_profile"
            )
            binding = session.io_binding()
            inputs = session.get_inputs()
            if index < 24:
                if len(inputs) != 1:
                    raise RuntimeError(f"encoder input count drift at segment {index}")
                binding.bind_ortvalue_input(inputs[0].name, current)
            else:
                if {item.name for item in inputs} != set(retained):
                    raise RuntimeError("head retained-feature input contract drift")
                for item in inputs:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                binding.bind_output(name, "cuda", 0)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            output_values = binding.get_outputs()
            if len(output_values) != len(output_names):
                raise RuntimeError(f"output count drift at segment {index}")
            output_map = dict(zip(output_names, output_values, strict=True))
            all_device = all(value.device_name() == "cuda" for value in output_values)
            if not all_device:
                raise RuntimeError(f"host output detected at segment {index}")
            current = common.choose_primary_output(index, output_map)
            if index in common.RETAIN_AFTER:
                retained[output_names[0]] = current
            profile_path = Path(session.end_profiling()).resolve(strict=True)
            placement = common.parse_profile(profile_path)
            if not placement["passed"]:
                raise RuntimeError(f"provider placement failed at segment {index}: {placement}")
            runtime_rows.append(
                {
                    "index": index,
                    "label": row["label"],
                    "precision": row["precision"],
                    "input_names": [item.name for item in inputs],
                    "output_names": output_names,
                    "all_outputs_device": all_device,
                    "provider_event_counts": placement["provider_event_counts"],
                }
            )
            profiles.append({"index": index, **common.identity(profile_path), **placement})
            del binding, session
            gc.collect()
            print(f"single segment {index + 1:02d}/25 passed", flush=True)

        migraphx_logits = np.ascontiguousarray(current.numpy(), dtype=np.float32)
        if migraphx_logits.shape != (1, 2, 224, 224):
            raise RuntimeError(f"MIGraphX logits contract drift: {migraphx_logits.shape}")
        comparison = common.compare_logits(cpu_logits, migraphx_logits)
        strict_gates = common.strict_diagnostic_gates(comparison)
        operational_gates = {
            "onnx_checker_and_boundary_contracts": len(onnx_contracts) == 25,
            "cpu_output_finite": bool(np.isfinite(cpu_logits).all()),
            "migraphx_output_finite": bool(np.isfinite(migraphx_logits).all()),
            "all_25_profiles_migraphx_positive_cpu_zero": len(profiles) == 25
            and all(item["passed"] for item in profiles),
            "all_outputs_device_resident": len(runtime_rows) == 25
            and all(item["all_outputs_device"] for item in runtime_rows),
        }
        operational_passed = all(operational_gates.values())
        pair_path = args.output_dir / "cpu_and_migraphx_logits.npz"
        np.savez_compressed(
            pair_path,
            raw_input=raw,
            cpu_logits=cpu_logits,
            migraphx_logits=migraphx_logits,
        )
        result.update(
            {
                "status": "diagnostic_completed" if operational_passed else "failed",
                "candidate_lineage": {
                    "candidate_id": manifest["candidate_id"],
                    "manifest": manifest_identity,
                    "bundle": str(args.bundle.resolve(strict=True)),
                    "fp16_backbone_blocks": manifest["fp16_backbone_blocks"],
                    "int8_backbone_blocks": manifest["int8_backbone_blocks"],
                },
                "identities": {
                    "evaluation_script": common.identity(Path(__file__)),
                    "common_module": common.identity(Path(common.__file__)),
                    "sample": common.identity(args.sample),
                },
                "input": {
                    "shape": list(raw.shape),
                    "dtype": str(raw.dtype),
                    "array_sha256": common.array_sha256(raw),
                },
                "onnx_contracts": onnx_contracts,
                "segments": runtime_rows,
                "profiles": profiles,
                "comparison_cpu_vs_migraphx": comparison,
                "strict_numeric_diagnostic_thresholds": common.STRICT_DIAGNOSTIC_GATES,
                "strict_numeric_diagnostic_gates": strict_gates,
                "operational_gates": operational_gates,
                "artifacts": {"cpu_and_migraphx_logits": common.identity(pair_path)},
                "evidence_boundary": {
                    "strict_is_diagnostic_not_a_task_or_performance_hard_gate": True,
                    "historical_m0_strict_failure_is_immutable": True,
                    "current_candidate_strict_result_does_not_rewrite_historical_m0": True,
                    "task90_not_run": True,
                    "timing_not_measured": True,
                    "profiles_prove_provider_placement_not_native_kernel_precision": True,
                },
            }
        )
        result["claims"]["onnx_and_boundary_contracts_passed"] = operational_gates[
            "onnx_checker_and_boundary_contracts"
        ]
        result["claims"]["all_25_segments_migraphx_positive_cpu_zero"] = operational_gates[
            "all_25_profiles_migraphx_positive_cpu_zero"
        ]
        result["claims"]["device_resident_intersegment_io"] = operational_gates[
            "all_outputs_device_resident"
        ]
        result["claims"]["candidate_strict_numeric_diagnostic_passed"] = all(
            strict_gates.values()
        )
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    common.json_dump(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if operational_passed else 2)


if __name__ == "__main__":
    main()
