#!/usr/bin/env python3
"""CPU-only fail-closed regression tests for the final acceptance aggregator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "final_acceptance", ROOT / "aggregate_final_deployment_acceptance.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load final acceptance aggregator")
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)
PRODUCTION_LSMOD_PAIR = aggregate.LSMOD_PAIR
if PRODUCTION_LSMOD_PAIR != (
    819_664,
    "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12",
):
    raise RuntimeError("production static lsmod identity drift")
# Unit fixtures use a tiny stand-in while exercising the exact same path and
# byte-identity linkage. Production constants above remain explicitly asserted.
TEST_LSMOD_BYTES = b"test-only locked static lsmod"
aggregate.LSMOD_PAIR = (len(TEST_LSMOD_BYTES), hashlib.sha256(TEST_LSMOD_BYTES).hexdigest())


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def origin_fingerprint(capture_row: dict, image_id: str) -> dict:
    static_lsmod = {
        "size_bytes": aggregate.LSMOD_PAIR[0],
        "sha256": aggregate.LSMOD_PAIR[1],
    }
    portable = {
        "official_image_id": image_id,
        "onnxruntime": "1.19.2",
        "static_lsmod": static_lsmod,
    }
    signature = hashlib.sha256(
        json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "phase11_k100_deployment_runtime_fingerprint_v1",
        "status": "captured",
        "generator": capture_row,
        "portable_runtime_signature_sha256": signature,
        "portable_fields": portable,
        "static_lsmod": static_lsmod,
        "observation": {
            "image_identity_attested": True,
            "attested_image_id": image_id,
        },
    }


class BundleInventoryTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[Path, str]:
        bundle = root / "bundle"
        payload = bundle / "payload.bin"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"locked payload")
        capture = bundle / "tools" / "capture_k100_runtime_fingerprint.py"
        capture.parent.mkdir()
        capture.write_bytes(b"locked capture")
        shim = bundle / "tools" / "bin" / "lsmod"
        shim.parent.mkdir()
        shim.write_bytes(TEST_LSMOD_BYTES)
        rows = [
            {
                "path": "payload.bin",
                "size_bytes": payload.stat().st_size,
                "sha256": aggregate.sha256(payload),
            },
            {
                "path": "tools/capture_k100_runtime_fingerprint.py",
                "size_bytes": capture.stat().st_size,
                "sha256": aggregate.sha256(capture),
            },
            {
                "path": "tools/bin/lsmod",
                "size_bytes": shim.stat().st_size,
                "sha256": aggregate.sha256(shim),
            },
        ]
        image_id = "sha256:" + "a" * 64
        origin = bundle / "runtime" / "origin_runtime_fingerprint.json"
        origin.parent.mkdir()
        origin_payload = origin_fingerprint(rows[1], image_id)
        signature = origin_payload["portable_runtime_signature_sha256"]
        origin.write_text(
            json.dumps(origin_payload), encoding="utf-8"
        )
        rows.append(
            {
                "path": "runtime/origin_runtime_fingerprint.json",
                "size_bytes": origin.stat().st_size,
                "sha256": aggregate.sha256(origin),
            }
        )
        manifest = {
            "schema": aggregate.BUNDLE_SCHEMA,
            "status": "static_bundle_identity_locked",
            "bundle_id": "test-m5",
            "claims": {
                "static_bundle_ready": True,
                "runtime_validated": False,
                "deployment_ready": False,
            },
            "claim_boundaries": {
                "historical_int8_strict_logits": "failed_immutable",
                "provider_placement_does_not_prove_kernel_precision": True,
                "native_int8_kernel": "unverified",
            },
            "execution": {"kind": "segment25", "source_candidate_id": "M5"},
            "runtime_contract": {
                "official_image_id": image_id,
                "portable_runtime_signature_sha256": signature,
                "origin_runtime_fingerprint": rows[-1],
                "static_lsmod": rows[2],
                "onnxruntime": "1.19.2",
                "provider": "MIGraphXExecutionProvider",
            },
            "fixed_sample": {"expected_prediction_array_sha256": "c" * 64},
            "payload_file_count": len(rows),
            "payload_total_bytes": sum(row["size_bytes"] for row in rows),
            "files": rows,
        }
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return bundle, aggregate.sha256(manifest_path)

    def test_complete_mock_bundle_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, manifest_sha = self.make_bundle(Path(temporary))
            context = aggregate.inventory_bundle(bundle, manifest_sha, "M5")
            self.assertEqual(context["payload_file_count"], 4)

    def test_payload_mutation_after_static_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, manifest_sha = self.make_bundle(Path(temporary))
            (bundle / "payload.bin").write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "payload identity drift"):
                aggregate.inventory_bundle(bundle, manifest_sha, "M5")


class EnvironmentExclusionTests(unittest.TestCase):
    manifest_sha = "d" * 64
    image_id = "sha256:" + "a" * 64

    def make_ledger(self, base: Path) -> tuple[Path, Path, dict]:
        root = base / "invalid_environment_runs"
        artifacts_dir = root / "runs"
        artifacts_dir.mkdir(parents=True)
        late = artifacts_dir / "late_path.log"
        nohyhal = artifacts_dir / "nohyhal_hip100.log"
        late.write_text("initializer recursive lsmod before entrypoint\n", encoding="utf-8")
        nohyhal.write_text("HIP error 100: no ROCm-capable device\n", encoding="utf-8")
        entries = []
        for entry_id, mode, stage, artifact in (
            (
                "late_path",
                "bdc7_with_node3_host_hyhal_lsmod_recursion_before_model",
                "runtime_import_before_model_inference",
                late,
            ),
            (
                "nohyhal",
                "bdc7_without_host_hyhal_hip100_no_rocm_device",
                "model_session_creation_before_successful_inference",
                nohyhal,
            ),
        ):
            entries.append(
                {
                    "entry_id": entry_id,
                    "failure_mode": mode,
                    "failure_stage": stage,
                    "node_label": "K100-3",
                    "manifest_sha256": self.manifest_sha,
                    "official_image_id": self.image_id,
                    "accepted_environment": False,
                    "counts_toward_acceptance": False,
                    "model_execution_started": False,
                    "successful_model_inferences": 0,
                    "artifacts": [
                        {
                            "kind": "file",
                            "path": artifact.relative_to(root).as_posix(),
                            "size_bytes": artifact.stat().st_size,
                            "sha256": aggregate.sha256(artifact),
                        }
                    ],
                }
            )
        payload = {
            "schema": aggregate.EXCLUSION_SCHEMA,
            "status": "locked_invalid_environmental_runs_excluded",
            "generator": aggregate.identity(
                ROOT / "build_environment_exclusion_ledger.py"
            ),
            "claims": {
                "listed_runs_are_not_acceptance_evidence": True,
                "formal_receipts_are_not_overwritten": True,
                "fresh_formal_rerun_required": True,
                "listed_runs_satisfy_any_deployment_gate": False,
            },
            "entries": entries,
        }
        ledger = root / "environment_exclusion_ledger.json"
        ledger.write_text(json.dumps(payload), encoding="utf-8")
        return root, ledger, payload

    def test_two_known_invalid_runs_are_locked_but_never_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, ledger, _ = self.make_ledger(Path(temporary))
            result = aggregate.validate_environment_exclusion_ledger(
                root,
                ledger,
                aggregate.sha256(ledger),
                {self.manifest_sha},
                {self.image_id},
            )
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["known_invalid_runs_count_toward_acceptance"])

    def test_excluded_run_cannot_be_relabelled_as_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, ledger, payload = self.make_ledger(Path(temporary))
            payload["entries"][0]["counts_toward_acceptance"] = True
            ledger.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "acceptance count"):
                aggregate.validate_environment_exclusion_ledger(
                    root,
                    ledger,
                    aggregate.sha256(ledger),
                    {self.manifest_sha},
                    {self.image_id},
                )

    def test_exclusion_artifact_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, ledger, payload = self.make_ledger(base)
            escaped = base / "escaped.log"
            escaped.write_text("outside", encoding="utf-8")
            payload["entries"][0]["artifacts"][0] = {
                "kind": "file",
                "path": "../escaped.log",
                "size_bytes": escaped.stat().st_size,
                "sha256": aggregate.sha256(escaped),
            }
            ledger.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid.*relative path"):
                aggregate.validate_environment_exclusion_ledger(
                    root,
                    ledger,
                    aggregate.sha256(ledger),
                    {self.manifest_sha},
                    {self.image_id},
                )

    def test_empty_failed_directory_can_be_identity_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, ledger, payload = self.make_ledger(Path(temporary))
            empty = root / "runs" / "empty_failed_attempt"
            empty.mkdir()
            tree = aggregate.directory_tree_identity(empty)
            payload["entries"][0]["artifacts"][0] = {
                "kind": "directory_tree",
                "path": empty.relative_to(root).as_posix(),
                "file_count": tree["file_count"],
                "total_bytes": tree["total_bytes"],
                "tree_sha256": tree["tree_sha256"],
            }
            ledger.write_text(json.dumps(payload), encoding="utf-8")
            result = aggregate.validate_environment_exclusion_ledger(
                root,
                ledger,
                aggregate.sha256(ledger),
                {self.manifest_sha},
                {self.image_id},
            )
            self.assertEqual(result["status"], "passed")


class OperationalToolFailureTests(unittest.TestCase):
    manifest_sha = "e" * 64
    image_id = "sha256:" + "a" * 64

    def make_ledger(self, base: Path) -> tuple[Path, Path, dict]:
        root = base / "operational_failure"
        failed = root / "failed_launcher_run"
        failed.mkdir(parents=True)
        (failed / "result.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )
        tree = aggregate.directory_tree_identity(failed)
        payload = {
            "schema": aggregate.OPERATIONAL_FAILURE_SCHEMA,
            "status": "locked_operational_tool_failure_excluded",
            "generator": aggregate.identity(
                ROOT / "build_operational_tool_failure_ledger.py"
            ),
            "claims": {
                "listed_run_is_not_formal_acceptance_evidence": True,
                "successful_inner_model_smoke_does_not_replace_missing_outer_attestation": True,
                "formal_receipts_are_not_overwritten": True,
                "fresh_formal_launcher_rerun_required": True,
                "listed_run_satisfies_any_deployment_gate": False,
            },
            "entry": {
                "entry_id": "k100_3_launcher_host_output_permission_failure_v1",
                "category": "operational_tool_failure",
                "failure_mode": "host_output_permission_denied_after_successful_container_smoke",
                "failure_stage": "post_container_smoke_host_attestation_write",
                "candidate_label": "M5",
                "node_label": "K100-3",
                "manifest_sha256": self.manifest_sha,
                "official_image_id": self.image_id,
                "inner_model_smoke_status": "passed_observation_only",
                "outer_launch_attestation_written": False,
                "counts_toward_acceptance": False,
                "artifact": {
                    "kind": "directory_tree",
                    "path": failed.relative_to(root).as_posix(),
                    "file_count": tree["file_count"],
                    "total_bytes": tree["total_bytes"],
                    "tree_sha256": tree["tree_sha256"],
                },
            },
        }
        ledger = root / "operational_tool_failure_ledger.json"
        ledger.write_text(json.dumps(payload), encoding="utf-8")
        return root, ledger, payload

    def test_successful_inner_smoke_with_missing_attestation_never_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, ledger, _ = self.make_ledger(Path(temporary))
            result = aggregate.validate_operational_tool_failure_ledger(
                root,
                ledger,
                aggregate.sha256(ledger),
                {"M5": self.manifest_sha},
                {self.image_id},
            )
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["counts_toward_acceptance"])

    def test_operational_failure_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, ledger, payload = self.make_ledger(Path(temporary))
            payload["entry"]["counts_toward_acceptance"] = True
            ledger.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "acceptance count"):
                aggregate.validate_operational_tool_failure_ledger(
                    root,
                    ledger,
                    aggregate.sha256(ledger),
                    {"M5": self.manifest_sha},
                    {self.image_id},
                )


class TelemetryTests(unittest.TestCase):
    def sample(self, timestamp: int, power: bool = True) -> dict:
        metrics = {
            "temperature_c": {"values": [41.0], "source": "test"},
            "vram_used_bytes": {"values": [1024.0], "source": "test"},
        }
        if power:
            metrics["power_w"] = {"values": [109.0], "source": "test"}
        return {
            "monotonic_ns": timestamp,
            "metrics": metrics,
            "runtime_counters": {
                "inferences": timestamp // 1_000_000_000,
                "errors": 0,
                "nonfinite_outputs": 0,
                "fixed_prediction_drifts": 0,
            },
        }

    def test_complete_five_second_telemetry_recomputes(self) -> None:
        result = aggregate.telemetry_summary(
            [self.sample(0), self.sample(5_000_000_000)]
        )
        self.assertTrue(result["all_three_metrics_present_in_every_sample"])
        self.assertTrue(result["minimum_interval_ge_4_seconds"])
        self.assertTrue(result["maximum_interval_le_7_5_seconds"])

    def test_missing_power_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lacks power_w"):
            aggregate.telemetry_summary(
                [self.sample(0), self.sample(5_000_000_000, power=False)]
            )

    def test_excessive_gap_is_reported_failed(self) -> None:
        result = aggregate.telemetry_summary(
            [self.sample(0), self.sample(8_000_000_000)]
        )
        self.assertFalse(result["maximum_interval_le_7_5_seconds"])

    def test_nonzero_runtime_error_counter_is_rejected(self) -> None:
        row = self.sample(5_000_000_000)
        row["runtime_counters"]["errors"] = 1
        with self.assertRaisesRegex(RuntimeError, "nonzero error"):
            aggregate.telemetry_summary([self.sample(0), row])


class RecoveryTests(unittest.TestCase):
    def test_natural_worker_exit_cannot_count_as_active_termination(self) -> None:
        context = {
            "bundle_id": "bundle",
            "kind": "segment25",
            "image_id": "sha256:" + "a" * 64,
            "expected_prediction_sha256": "b" * 64,
            "manifest_identity": {"size_bytes": 1, "sha256": "c" * 64},
        }
        payload = {
            "schema": "phase11_k100_recovery_3_v1",
            "status": "passed",
            "started_at_utc": "2026-08-18T00:00:00+00:00",
            "ended_at_utc": "2026-08-18T00:01:00+00:00",
            "bundle_id": "bundle",
            "manifest_sha256": "c" * 64,
            "claims": {
                "three_active_termination_reloads_passed": True,
                "deployment_ready": False,
            },
            "protocol": {
                "active_termination_cycles": 3,
                "fresh_reload_process_each_cycle": True,
            },
            "cycles": [
                {
                    "cycle": index,
                    "worker_ready": True,
                    "active_termination_returncode": 0,
                    "reload_returncode": 0,
                    "reload_fixed_prediction_passed": True,
                    "reload_receipt": {},
                }
                for index in (1, 2, 3)
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "not actively signal-terminated"):
            aggregate.validate_recovery(payload, context)


class SyntheticEndToEndTests(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def make_bundle(self, base: Path) -> tuple[Path, str, dict]:
        bundle = base / "bundle"
        capture = bundle / "tools" / "capture_k100_runtime_fingerprint.py"
        capture.parent.mkdir(parents=True)
        capture.write_bytes(b"capture generator")
        payload = bundle / "model.bin"
        payload.write_bytes(b"model")
        shim = bundle / "tools" / "bin" / "lsmod"
        shim.parent.mkdir()
        shim.write_bytes(TEST_LSMOD_BYTES)
        rows = [
            aggregate.identity(capture),
            aggregate.identity(payload),
            aggregate.identity(shim),
        ]
        for row, relative in zip(
            rows,
            (
                "tools/capture_k100_runtime_fingerprint.py",
                "model.bin",
                "tools/bin/lsmod",
            ),
        ):
            row["path"] = relative
        image_id = "sha256:" + "a" * 64
        origin = bundle / "runtime" / "origin_runtime_fingerprint.json"
        origin.parent.mkdir()
        origin_payload = origin_fingerprint(rows[0], image_id)
        signature = origin_payload["portable_runtime_signature_sha256"]
        self.write_json(origin, origin_payload)
        origin_row = aggregate.identity(origin)
        origin_row["path"] = "runtime/origin_runtime_fingerprint.json"
        rows.append(origin_row)
        manifest = {
            "schema": aggregate.BUNDLE_SCHEMA,
            "status": "static_bundle_identity_locked",
            "bundle_id": "synthetic-m5",
            "claims": {
                "static_bundle_ready": True,
                "runtime_validated": False,
                "deployment_ready": False,
            },
            "claim_boundaries": {
                "historical_int8_strict_logits": "failed_immutable",
                "provider_placement_does_not_prove_kernel_precision": True,
                "native_int8_kernel": "unverified",
            },
            "execution": {"kind": "segment25", "source_candidate_id": "M5"},
            "runtime_contract": {
                "official_image_id": image_id,
                "portable_runtime_signature_sha256": signature,
                "origin_runtime_fingerprint": origin_row,
                "static_lsmod": rows[2],
                "onnxruntime": "1.19.2",
                "provider": "MIGraphXExecutionProvider",
            },
            "fixed_sample": {"expected_prediction_array_sha256": "c" * 64},
            "payload_file_count": len(rows),
            "payload_total_bytes": sum(row["size_bytes"] for row in rows),
            "files": rows,
        }
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_sha = aggregate.sha256(manifest_path)
        context = aggregate.inventory_bundle(bundle, manifest_sha, "M5")
        return bundle, manifest_sha, context

    def infer_receipt(self, context: dict) -> dict:
        expected = context["expected_prediction_sha256"]
        return {
            "schema": "phase11_k100_inference_receipt_v1",
            "status": "passed",
            "bundle_id": context["bundle_id"],
            "execution_kind": context["kind"],
            "manifest": context["manifest_identity"],
            "runtime": {
                "selected_provider": "MIGraphXExecutionProvider",
                "cpu_fallback_disabled": True,
                "image_identity_attested": True,
                "expected_official_image_id": context["image_id"],
                "attested_image_id": context["image_id"],
                "static_lsmod": {
                    "size_bytes": aggregate.LSMOD_PAIR[0],
                    "sha256": aggregate.LSMOD_PAIR[1],
                    "path_prepend_active": True,
                },
            },
            "timing": {"session_load_seconds": 1.0},
            "input": {
                "shape": [1, 6, 224, 224],
                "dtype": "float32",
                "finite": True,
                "preprocessing_applied_by_cli": False,
            },
            "output": {
                "logits_shape": [1, 2, 224, 224],
                "prediction_shape": [1, 224, 224],
                "logits_dtype": "float32",
                "prediction_dtype": "uint8",
                "finite_logits": True,
            },
            "fixed_sample_gate": {
                "input_matches_bundled_fixed_sample": True,
                "prediction_matches_expected": True,
                "expected_prediction_array_sha256": expected,
                "observed_prediction_array_sha256": expected,
            },
            "evidence_boundary": {
                "strict_logit_equivalence_inferred_from_this_run": False,
                "native_kernel_precision_inferred_from_provider_placement": False,
            },
        }

    def common(self, schema: str, context: dict, claim: str) -> dict:
        return {
            "schema": schema,
            "status": "passed",
            "started_at_utc": "2026-08-18T00:00:00+00:00",
            "ended_at_utc": "2026-08-18T01:01:00+00:00",
            "bundle_id": context["bundle_id"],
            "manifest_sha256": context["manifest_identity"]["sha256"],
            "claims": {claim: True, "deployment_ready": False},
        }

    def make_acceptance(self, base: Path, context: dict) -> Path:
        root = base / "acceptance"
        root.mkdir()
        static = {
            "schema": "phase11_k100_bundle_static_verification_v1",
            "status": "passed",
            "bundle_id": context["bundle_id"],
            "execution_kind": context["kind"],
            "manifest": context["manifest_identity"],
            "payload_file_count": context["payload_file_count"],
            "payload_total_bytes": context["payload_total_bytes"],
            "claims": {
                "static_bundle_identity_verified": True,
                "runtime_validated": False,
                "cache_portability_verified": False,
                "deployment_ready": False,
            },
        }
        self.write_json(root / "static_verification.json", static)
        cold = self.common(
            "phase11_k100_cold_start_5_v1",
            context,
            "five_fresh_processes_passed",
        )
        cold["protocol"] = {
            "fresh_python_processes": 5,
            "full_payload_hash_each_process": True,
            "fixed_prediction_sha_gate_each_process": True,
        }
        cold["trials"] = [
            {"trial": index, "returncode": 0, "receipt": self.infer_receipt(context)}
            for index in range(1, 6)
        ]
        self.write_json(root / "cold_start_5" / "result.json", cold)
        recovery = self.common(
            "phase11_k100_recovery_3_v1",
            context,
            "three_active_termination_reloads_passed",
        )
        recovery["protocol"] = {
            "active_termination_cycles": 3,
            "fresh_reload_process_each_cycle": True,
        }
        recovery["cycles"] = [
            {
                "cycle": index,
                "worker_ready": True,
                "active_termination_returncode": -15,
                "reload_returncode": 0,
                "reload_fixed_prediction_passed": True,
                "reload_receipt": self.infer_receipt(context),
            }
            for index in range(1, 4)
        ]
        self.write_json(root / "recovery_3" / "result.json", recovery)
        capture_row = next(
            row
            for row in context["manifest"]["files"]
            if row["path"] == "tools/capture_k100_runtime_fingerprint.py"
        )
        for index, node in enumerate(("K100-2", "K100-3"), start=2):
            runtime_hostname = (
                f"container-{index}" if node == "K100-3" else f"machine{index}"
            )
            smoke = self.common(
                "phase11_k100_cross_node_smoke_v1",
                context,
                "target_cache_smoke_passed",
            )
            smoke["claims"]["cache_portability_verified"] = True
            smoke.update(
                {
                    "node_label": node,
                    "hostname": runtime_hostname,
                    "runtime_signature_match": True,
                    "inference_returncode": 0,
                    "fixed_prediction_passed": True,
                    "inference_receipt": self.infer_receipt(context),
                }
            )
            directory = root / f"smoke_k100_{index}"
            self.write_json(directory / "result.json", smoke)
            fingerprint = {
                "schema": "phase11_k100_deployment_runtime_fingerprint_v1",
                "status": "captured",
                "generator": capture_row,
                "portable_runtime_signature_sha256": context["manifest"][
                    "runtime_contract"
                ]["portable_runtime_signature_sha256"],
                "portable_fields": {
                    "official_image_id": context["image_id"],
                    "onnxruntime": "1.19.2",
                    "static_lsmod": {
                        "size_bytes": aggregate.LSMOD_PAIR[0],
                        "sha256": aggregate.LSMOD_PAIR[1],
                    },
                },
                "static_lsmod": {
                    "size_bytes": aggregate.LSMOD_PAIR[0],
                    "sha256": aggregate.LSMOD_PAIR[1],
                },
                "observation": {
                    "image_identity_attested": True,
                    "attested_image_id": context["image_id"],
                    "hostname": runtime_hostname,
                },
            }
            self.write_json(directory / "target_runtime_fingerprint.json", fingerprint)
            if node == "K100-3":
                container_smoke = directory / "container_smoke"
                container_smoke.mkdir()
                (container_smoke / "result.json").write_bytes(
                    (directory / "result.json").read_bytes()
                )
                (container_smoke / "target_runtime_fingerprint.json").write_bytes(
                    (directory / "target_runtime_fingerprint.json").read_bytes()
                )
                launcher_stdout = directory / "container_launcher_stdout.log"
                launcher_stderr = directory / "container_launcher_stderr.log"
                launcher_stdout.write_text("passed\n", encoding="utf-8")
                launcher_stderr.write_text("", encoding="utf-8")
                image_id = context["image_id"]
                docker_argv = [
                    "docker",
                    "run",
                    "--rm",
                    "--device=/dev/kfd",
                    "--device=/dev/dri",
                    "-v",
                    "/evidence/bundle:/bundle:ro",
                    "-v",
                    "/evidence/acceptance/smoke_k100_3:/acceptance",
                    "-v",
                    "/usr/local/hyhal:/opt/hyhal:ro",
                    "-e",
                    f"PATH={aggregate.CONTAINER_PATH}",
                    "-e",
                    f"PHASE11_K100_IMAGE_ID={image_id}",
                    "-e",
                    (
                        "PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION="
                        f"{aggregate.PREENTRYPOINT_TOKEN}"
                    ),
                    "--entrypoint",
                    "/bundle/tools/run_cross_node_smoke.sh",
                    image_id,
                    "--bundle",
                    "/bundle",
                    "--expected-manifest-sha256",
                    context["manifest_identity"]["sha256"],
                    "--output-dir",
                    "/acceptance/container_smoke",
                    "--device",
                    "0",
                    "--node-label",
                    "K100-3",
                ]
                launch = {
                    "schema": "phase11_k100_container_launch_attestation_v1",
                    "status": "passed",
                    "started_at_utc": "2026-08-18T00:00:00+00:00",
                    "ended_at_utc": "2026-08-18T02:00:00+00:00",
                    "generator": aggregate.identity(
                        ROOT / "launch_cross_node_smoke_container.py"
                    ),
                    "hostname": "machine3",
                    "node_label": "K100-3",
                    "device": 0,
                    "bundle_id": context["bundle_id"],
                    "manifest": context["manifest_identity"],
                    "official_image_id": image_id,
                    "docker_image_inspect": {
                        "returncode": 0,
                        "observed_image_id": image_id,
                    },
                    "docker_argv": docker_argv,
                    "docker_returncode": 0,
                    "container_bundle_dir": "/bundle",
                    "container_output_dir": "/acceptance/container_smoke",
                    "host_output_dir": "/evidence/acceptance/smoke_k100_3",
                    "host_output_ownership": {
                        "created_exclusively_before_docker": True,
                        "uid": 1000,
                        "gid": 1000,
                        "mode": "0o755",
                    },
                    "launcher_logs": {
                        "stdout": aggregate.identity(launcher_stdout),
                        "stderr": aggregate.identity(launcher_stderr),
                    },
                    "pre_entrypoint_environment": {
                        "PATH": aggregate.CONTAINER_PATH,
                        "PHASE11_K100_IMAGE_ID": image_id,
                        "PHASE11_K100_PREENTRYPOINT_PATH_ATTESTATION": (
                            aggregate.PREENTRYPOINT_TOKEN
                        ),
                    },
                    "static_lsmod": {
                        "container_path": "/bundle/tools/bin/lsmod",
                        "size_bytes": aggregate.LSMOD_PAIR[0],
                        "sha256": aggregate.LSMOD_PAIR[1],
                    },
                    "host_hyhal_mount": {
                        "requested_source": "/opt/hyhal",
                        "source": "/usr/local/hyhal",
                        "target": "/opt/hyhal",
                        "read_only": True,
                        "requested_source_was_symlink": True,
                        "symlink_resolved_before_docker": True,
                    },
                    "container_smoke_result": aggregate.identity(
                        container_smoke / "result.json"
                    ),
                    "container_runtime_fingerprint": aggregate.identity(
                        container_smoke / "target_runtime_fingerprint.json"
                    ),
                    "smoke_result": aggregate.identity(directory / "result.json"),
                    "formal_runtime_fingerprint": aggregate.identity(
                        directory / "target_runtime_fingerprint.json"
                    ),
                    "claims": {
                        "path_injected_by_docker_run_before_entrypoint": True,
                        "locked_static_lsmod_first_on_path": True,
                        "host_hyhal_mounted_read_only": True,
                        "host_output_created_exclusively_before_docker": True,
                        "container_writes_isolated_subdirectory": True,
                        "formal_receipts_copied_without_byte_drift": True,
                        "invalid_prior_launcher_runs_count_toward_acceptance": False,
                        "no_hyhal_diagnostics_count_toward_acceptance": False,
                    },
                    "failure": None,
                }
                self.write_json(directory / "container_launch_attestation.json", launch)
        telemetry_rows = []
        for index in range(720):
            telemetry_rows.append(
                {
                    "monotonic_ns": index * 5_000_000_000,
                    "metrics": {
                        "temperature_c": {"values": [41.0], "source": "test"},
                        "power_w": {"values": [109.0], "source": "test"},
                        "vram_used_bytes": {"values": [1024.0], "source": "test"},
                    },
                    "runtime_counters": {
                        "inferences": index,
                        "errors": 0,
                        "nonfinite_outputs": 0,
                        "fixed_prediction_drifts": 0,
                    },
                }
            )
        stability_dir = root / "stability_60min"
        stability_dir.mkdir()
        telemetry_path = stability_dir / "telemetry_5s.jsonl"
        telemetry_path.write_text(
            "".join(json.dumps(row) + "\n" for row in telemetry_rows), encoding="utf-8"
        )
        stability = self.common(
            "phase11_k100_stability_60min_v1",
            context,
            "continuous_60min_passed",
        )
        stability["protocol"] = {
            "requested_duration_seconds": 3600,
            "observed_duration_seconds": 3600.1,
            "telemetry_interval_seconds": 5,
            "telemetry_allowed_interval_seconds": [4.0, 7.5],
            "max_inferences": None,
        }
        stability["gates"] = {key: True for key in aggregate.EXPECTED_STABILITY_GATES}
        stability["measurements"] = {
            "inferences": 720,
            "errors": 0,
            "error_rate": 0.0,
            "nonfinite_outputs": 0,
            "fixed_prediction_drifts": 0,
            "inference_seconds_median": 0.01,
            "inference_seconds_p95": 0.02,
            "telemetry": aggregate.telemetry_summary(telemetry_rows),
        }
        self.write_json(stability_dir / "result.json", stability)
        return root

    def test_complete_synthetic_candidate_passes_all_six_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle, manifest_sha, context = self.make_bundle(base)
            acceptance = self.make_acceptance(base, context)
            result = aggregate.validate_candidate("M5", bundle, manifest_sha, acceptance)
            self.assertTrue(result["all_six_required_gates_passed"])
            self.assertEqual(set(result["gates"]), set(aggregate.RECEIPT_LAYOUT))

    def test_k1003_smoke_without_pre_entrypoint_path_attestation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle, manifest_sha, context = self.make_bundle(base)
            acceptance = self.make_acceptance(base, context)
            (acceptance / "smoke_k100_3" / "container_launch_attestation.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "launch attestation"):
                aggregate.validate_candidate("M5", bundle, manifest_sha, acceptance)

    def test_wrapper_time_path_cannot_replace_docker_pre_entrypoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle, manifest_sha, context = self.make_bundle(base)
            acceptance = self.make_acceptance(base, context)
            launch_path = (
                acceptance / "smoke_k100_3" / "container_launch_attestation.json"
            )
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["pre_entrypoint_environment"]["PATH"] = "/usr/local/bin:/usr/bin"
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "locked environment drift"):
                aggregate.validate_candidate("M5", bundle, manifest_sha, acceptance)

    def test_output_publish_writes_json_csv_and_optional_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new_aggregate"
            gates = {name: {"passed": True} for name in aggregate.RECEIPT_LAYOUT}
            receipts = {
                name: {"sha256": f"{index:064x}"}
                for index, name in enumerate(aggregate.RECEIPT_LAYOUT, start=1)
            }
            payload = {
                "status": "passed",
                "candidates": [
                    {
                        "label": "M5",
                        "status": "passed",
                        "bundle_id": "bundle",
                        "execution_kind": "segment25",
                        "manifest": {"sha256": "a" * 64},
                        "gates": gates,
                        "receipts": receipts,
                        "all_six_required_gates_passed": True,
                    }
                ],
            }
            rows = aggregate.write_outputs(output, payload, markdown=True)
            self.assertEqual(
                set(rows),
                {
                    "final_deployment_acceptance.json",
                    "final_deployment_acceptance.csv",
                    "final_deployment_acceptance.md",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate|already|exist"):
                aggregate.write_outputs(output, payload, markdown=True)


if __name__ == "__main__":
    unittest.main()
