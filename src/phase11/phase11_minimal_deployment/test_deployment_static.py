#!/usr/bin/env python3
"""CPU-only regression tests for deployment safety gates."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build_k100_bundle")
acceptance = load("acceptance_k100")


class SourceRootTests(unittest.TestCase):
    def test_manifest_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "authorized"
            manifest_dir = source_root / "manifests" / "M5"
            manifest_dir.mkdir(parents=True)
            manifest = manifest_dir / "mixed_precision_manifest.final.json"
            manifest.write_text("{}", encoding="utf-8")
            outside = base / "outside.mxr"
            outside.write_bytes(b"not authorized")
            with self.assertRaisesRegex(RuntimeError, "escapes authorized --source-root"):
                builder.source_path(manifest, "../../../outside.mxr", source_root)

    def test_manifest_artifact_inside_source_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / "authorized"
            manifest_dir = source_root / "manifests" / "M5"
            artifact = source_root / "models" / "segment_00.onnx"
            manifest_dir.mkdir(parents=True)
            artifact.parent.mkdir(parents=True)
            manifest = manifest_dir / "mixed_precision_manifest.final.json"
            manifest.write_text("{}", encoding="utf-8")
            artifact.write_bytes(b"inside")
            observed, _ = builder.source_path(
                manifest, "../../models/segment_00.onnx", source_root
            )
            self.assertEqual(observed, artifact.resolve())


class MixedAdmissionTests(unittest.TestCase):
    def candidate_payload(self) -> dict:
        fp16 = builder.EXPECTED_VARIANTS["M5"]
        rows = []
        for index in range(25):
            rows.append(
                {
                    "index": index,
                    "label": (
                        f"encoder_block_{index:02d}"
                        if index < 24
                        else "upernet_decoder_head_fpn4barrier"
                    ),
                    "precision": (
                        "fp32_compat_barrier"
                        if index == 24
                        else "fp16"
                        if index in fp16
                        else "int8_qdq"
                    ),
                    "model": f"models/{index}.onnx",
                    "model_identity": {"size_bytes": 1, "sha256": "0" * 64},
                    "cache": f"caches/{index}.mxr",
                    "cache_identity": {"size_bytes": 1, "sha256": "1" * 64},
                    "source_lineage": (
                        builder.EXPECTED_HEAD_LINEAGES["M5"]
                        if index == 24
                        else "frozen"
                    ),
                }
            )
        return {
            "schema": builder.MIXED_SCHEMA,
            "status": "cache_finalized_static_pass",
            "manifest_stage": builder.MIXED_FINAL_STAGE,
            "candidate_id": "M5",
            "segment_count": 25,
            "retain_encoder_outputs_after_segments": [5, 11, 17, 23],
            "head_precision": "fp32_compat_barrier",
            "fp16_backbone_blocks": list(fp16),
            "int8_backbone_blocks": [index for index in range(24) if index not in fp16],
            "segments": rows,
            "claims": {
                "all_required_mxr_caches_present": True,
                "fp16_cache_compilation_and_placement_passed": True,
            },
        }

    def test_wrong_m5_mapping_is_rejected(self) -> None:
        payload = self.candidate_payload()
        payload["fp16_backbone_blocks"] = [0]
        with self.assertRaisesRegex(RuntimeError, "FP16 block mapping drift"):
            builder.validate_mixed_final_manifest(Path("manifest.json"), payload)

    def test_wrong_repaired_head_lineage_is_rejected(self) -> None:
        payload = self.candidate_payload()
        payload["segments"][24]["source_lineage"] = "unvalidated_head"
        with self.assertRaisesRegex(RuntimeError, "repaired-head lineage drift"):
            builder.validate_mixed_final_manifest(Path("manifest.json"), payload)

    def test_unlocked_summary_generator_is_rejected_before_raw_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary)
            manifest = source_root / "candidate.json"
            manifest.write_text("{}", encoding="utf-8")
            manifest_identity = builder.identity(manifest)
            payload = {
                "schema": builder.TASK_ADMISSION_SCHEMA,
                "status": "passed",
                "claims": {
                    "task_equivalence_repeatability_confirmed": True,
                    "task_equivalence_performance_track_may_proceed": True,
                },
                "candidate_lineage": {
                    "candidate_id": "M5",
                    "manifest": manifest_identity,
                },
                "runs": [{"trial_index": index} for index in (1, 2, 3)],
                "gates": {"passed": True},
                "identities": {
                    "aggregate_script": {"size_bytes": 1, "sha256": "0" * 64},
                    "common_module": builder.MIXED_COMMON,
                },
            }
            with self.assertRaisesRegex(RuntimeError, "generator identity drift"):
                builder.validate_task_admission(
                    source_root / "summary.json",
                    payload,
                    "M5",
                    manifest,
                    source_root,
                )


class FrozenFp16AdmissionTests(unittest.TestCase):
    def test_arbitrary_candidate_cannot_be_declared_by_build_report(self) -> None:
        payload = {
            "status": "created_static_pass",
            "variant": builder.FP16_VARIANT,
            "source": {
                "size": 638_894_819,
                "sha256": builder.FP16_SOURCE_SHA256,
            },
            "candidate": {
                "size": 1,
                "sha256": "0" * 64,
                "outputs": [
                    "logits",
                    "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0",
                ],
            },
            "layernorm": {"count": 49, "remaining": 0},
            "convtranspose": {"count": 3, "remaining": 0},
            "claims": {"static_graph_valid": True},
        }
        with self.assertRaisesRegex(RuntimeError, "candidate identity drift"):
            builder.validate_fp16_build_report(payload)

    def test_arbitrary_cache_cannot_be_paired_with_frozen_fp16(self) -> None:
        payload = {
            "status": "failed",
            "inputs": {
                "candidate": builder.FP16_MODEL,
                "build_report": builder.FP16_BUILD_REPORT,
            },
            "compiled_cache": {"size_bytes": 1, "sha256": "0" * 64},
            "runtime": {
                "onnxruntime": "1.19.2",
                "container_image_id": "sha256:" + "a" * 64,
            },
            "profile": {
                "provider_event_counts": {"MIGraphXExecutionProvider": 1}
            },
            "claims": {
                "strict_migraphx_single_sample_placement": True,
                "formal_exact_transform_equivalence": False,
            },
            "formal_exact_transform_gate": {"passed": False},
        }
        with self.assertRaisesRegex(RuntimeError, "model/cache pairing drift"):
            builder.validate_fp16_single_admission(
                payload, "sha256:" + "a" * 64
            )

    def test_task_summary_cannot_promote_deployment_state(self) -> None:
        payload = {
            "status": "task_level_confirmation_passed",
            "protocol": {"fresh": True},
            "frozen_evidence": {
                "diagnostic_result": builder.FP16_FROZEN_TASK_RESULT,
                "failed_single_gate": builder.FP16_SINGLE_ADMISSION,
            },
            "fresh_evidence": {"result": builder.FP16_FRESH_TASK_RESULT},
            "gates": {"passed": True},
            "metrics": {"valid_pixels": 3_927_398},
            "placement": {
                "provider_event_counts": {"MIGraphXExecutionProvider": 90}
            },
            "claims": {
                "strict_logits_numeric_equivalence": False,
                "task_level_accuracy_confirmed": True,
                "compatibility_variant_only": True,
                "paired_performance_experiment_allowed": True,
                "performance_result_available": False,
                "deployment_complete": True,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "claim boundary drift"):
            builder.validate_fp16_task_admission(payload)


class TelemetryTests(unittest.TestCase):
    EMPTY_SYSFS = {
        "vram_used_bytes": [],
        "temperature_millicelsius": [],
        "power_microwatts": [],
    }

    def row(self, timestamp_ns: int, metrics: dict) -> dict:
        return {"monotonic_ns": timestamp_ns, "metrics": metrics}

    def test_missing_power_cannot_pass_coverage(self) -> None:
        parsed = acceptance.parse_hy_smi(
            "HCU[0] : Temperature (Sensor edge) (C): 41.0",
            "VRAM USED(MiB): 1955",
        )
        metrics = acceptance.canonical_metrics(self.EMPTY_SYSFS, parsed)
        coverage = acceptance.telemetry_coverage(
            [self.row(0, metrics), self.row(5_000_000_000, metrics)]
        )
        self.assertFalse(coverage["all_three_metrics_present_in_every_sample"])
        self.assertEqual(coverage["coverage_sample_counts"]["power_w"], 0)

    def test_complete_metrics_and_five_second_interval_pass(self) -> None:
        parsed = acceptance.parse_hy_smi(
            "\n".join(
                (
                    "HCU[0] : Average Graphics Package Power (W): 109.0",
                    "HCU[0] : Temperature (Sensor edge) (C): 41.0",
                )
            ),
            "VRAM USED(MiB): 1955",
        )
        metrics = acceptance.canonical_metrics(self.EMPTY_SYSFS, parsed)
        coverage = acceptance.telemetry_coverage(
            [self.row(0, metrics), self.row(5_000_000_000, metrics)]
        )
        self.assertTrue(coverage["all_three_metrics_present_in_every_sample"])
        self.assertTrue(coverage["minimum_interval_ge_4_seconds"])
        self.assertTrue(coverage["maximum_interval_le_7_5_seconds"])

    def test_eight_second_gap_fails_maximum_interval(self) -> None:
        metrics = {
            "temperature_c": {"values": [41.0], "source": "test"},
            "power_w": {"values": [109.0], "source": "test"},
            "vram_used_bytes": {"values": [1.0], "source": "test"},
        }
        coverage = acceptance.telemetry_coverage(
            [self.row(0, metrics), self.row(8_000_000_000, metrics)]
        )
        self.assertFalse(coverage["maximum_interval_le_7_5_seconds"])


class OutputIsolationTests(unittest.TestCase):
    def test_acceptance_output_inside_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            with self.assertRaisesRegex(RuntimeError, "outside immutable bundle"):
                acceptance.new_output_dir(bundle / "run", bundle)


if __name__ == "__main__":
    unittest.main()
