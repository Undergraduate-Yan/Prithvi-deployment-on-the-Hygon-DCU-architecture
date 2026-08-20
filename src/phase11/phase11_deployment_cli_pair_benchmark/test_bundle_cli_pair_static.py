#!/usr/bin/env python3
"""CPU-only regressions for the K100-2 paired deployment benchmark tools."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trial = load("pair_trial_for_test", "benchmark_bundle_cli_trial.py")
aggregate = load("pair_aggregate_for_test", "aggregate_bundle_cli_pair.py")


class ProtocolTests(unittest.TestCase):
    def test_rotated_sequence_is_maximally_balanced(self) -> None:
        self.assertEqual(trial.expected_sequence(1), ("m5", "fp16"))
        self.assertEqual(trial.expected_sequence(2), ("fp16", "m5"))
        self.assertEqual(trial.expected_sequence(3), ("m5", "fp16"))
        counts = {variant: {1: 0, 2: 0} for variant in ("m5", "fp16")}
        for index in (1, 2, 3):
            for position, variant in enumerate(trial.expected_sequence(index), start=1):
                counts[variant][position] += 1
        self.assertTrue(all(abs(row[1] - row[2]) <= 1 for row in counts.values()))
        with self.assertRaises(RuntimeError):
            trial.expected_sequence(4)

    def test_timing_math_and_count_gate(self) -> None:
        values = [index / 100_000.0 for index in range(1, 101)]
        summary = trial.timing_summary(values)
        self.assertEqual(summary["count"], 100)
        self.assertAlmostEqual(summary["median_ms"], 0.505)
        self.assertGreater(summary["p99_ms"], summary["p95_ms"])
        self.assertGreater(summary["throughput_samples_per_second"], 0)
        with self.assertRaises(RuntimeError):
            trial.timing_summary(values[:-1])

    def test_atomic_receipts_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "trial.json"
            trial.atomic_json(first, {"x": 1})
            with self.assertRaises(RuntimeError):
                trial.atomic_json(first, {"x": 2})
            second = Path(temporary) / "summary.json"
            aggregate.atomic_json(second, {"y": 1})
            with self.assertRaises(RuntimeError):
                aggregate.atomic_json(second, {"y": 2})

    def test_same_input_allows_bundle_specific_prediction_oracles(self) -> None:
        input_sha = "1" * 64
        records = {
            "m5": [
                {
                    "fixed_input_array_sha256": input_sha,
                    "expected_prediction_array_sha256": "a" * 64,
                }
                for _ in range(3)
            ],
            "fp16": [
                {
                    "fixed_input_array_sha256": input_sha,
                    "expected_prediction_array_sha256": "b" * 64,
                }
                for _ in range(3)
            ],
        }
        observed_input, expected_predictions = aggregate.validate_cross_bundle_fixed_contract(
            records
        )
        self.assertEqual(observed_input, input_sha)
        self.assertEqual(expected_predictions, {"m5": "a" * 64, "fp16": "b" * 64})

    def test_raw_input_drift_still_fails_closed(self) -> None:
        records = {
            "m5": [
                {
                    "fixed_input_array_sha256": "1" * 64,
                    "expected_prediction_array_sha256": "a" * 64,
                }
            ]
            * 3,
            "fp16": [
                {
                    "fixed_input_array_sha256": "2" * 64,
                    "expected_prediction_array_sha256": "b" * 64,
                }
            ]
            * 3,
        }
        with self.assertRaisesRegex(RuntimeError, "same raw input"):
            aggregate.validate_cross_bundle_fixed_contract(records)

    def test_bundle_prediction_oracle_drift_still_fails_closed(self) -> None:
        records = {
            "m5": [
                {
                    "fixed_input_array_sha256": "1" * 64,
                    "expected_prediction_array_sha256": value * 64,
                }
                for value in ("a", "a", "c")
            ],
            "fp16": [
                {
                    "fixed_input_array_sha256": "1" * 64,
                    "expected_prediction_array_sha256": "b" * 64,
                }
            ]
            * 3,
        }
        with self.assertRaisesRegex(RuntimeError, "m5 expected prediction identity changed"):
            aggregate.validate_cross_bundle_fixed_contract(records)


class DockerAttestationTests(unittest.TestCase):
    def make_inspect(self, root: Path) -> tuple[Path, Path]:
        container_id = "a" * 64
        sequence = ("m5", "fp16")
        command = [
            "/tools/benchmark_bundle_cli_trial.py",
            "--bundle",
            "/bundle",
            "--variant",
            "m5",
            "--trial-index",
            "1",
            "--candidate-position",
            "1",
            "--sequence",
            ",".join(sequence),
            "--device",
            "0",
            "--output",
            "/run/result.json",
        ]
        row = {
            "Id": container_id,
            "Image": aggregate.IMAGE_ID,
            "Name": "/test",
            "State": {
                "Status": "exited",
                "ExitCode": 0,
                "StartedAt": "2026-08-18T00:00:00Z",
                "FinishedAt": "2026-08-18T00:01:00Z",
            },
            "Config": {
                "Image": aggregate.IMAGE_ID,
                "Hostname": container_id[:12],
                "Entrypoint": ["/usr/bin/python3"],
                "Cmd": command,
                "Env": [
                    f"PATH={aggregate.DOCKER_PATH}",
                    f"PHASE11_K100_IMAGE_ID={aggregate.IMAGE_ID}",
                ],
            },
            "HostConfig": {
                "Memory": aggregate.MEMORY_BYTES,
                "PidsLimit": aggregate.PIDS_LIMIT,
                "IpcMode": "host",
                "ShmSize": aggregate.SHM_BYTES,
                "GroupAdd": ["video"],
                "Devices": [
                    {
                        "PathOnHost": "/dev/kfd",
                        "PathInContainer": "/dev/kfd",
                        "CgroupPermissions": "rwm",
                    },
                    {
                        "PathOnHost": "/dev/dri",
                        "PathInContainer": "/dev/dri",
                        "CgroupPermissions": "rwm",
                    },
                ],
                "Binds": [
                    "/host/m5:/bundle:ro",
                    "/host/tools:/tools:ro",
                    "/host/run:/run:rw",
                    "/opt/hyhal:/opt/hyhal:ro",
                ],
            },
        }
        inspect_path = root / "inspect.json"
        inspect_path.write_text(json.dumps([row]), encoding="utf-8")
        id_path = root / "id.txt"
        id_path.write_text(container_id + "\n", encoding="utf-8")
        return inspect_path, id_path

    def test_exact_docker_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inspect_path, id_path = self.make_inspect(Path(temporary))
            result = aggregate.validate_container_inspect(
                inspect_path, id_path, "m5", 1, 1, ("m5", "fp16")
            )
            self.assertEqual(result["image_id"], aggregate.IMAGE_ID)

    def test_path_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inspect_path, id_path = self.make_inspect(Path(temporary))
            payload = json.loads(inspect_path.read_text(encoding="utf-8"))
            payload[0]["Config"]["Env"][0] = "PATH=/usr/bin"
            inspect_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                aggregate.validate_container_inspect(
                    inspect_path, id_path, "m5", 1, 1, ("m5", "fp16")
                )


class SourceBoundaryTests(unittest.TestCase):
    def test_trial_uses_locked_public_runner(self) -> None:
        source = (HERE / "benchmark_bundle_cli_trial.py").read_text(encoding="utf-8")
        self.assertIn("infer.K100BundleRunner(bundle, args.device, verify_payloads=True)", source)
        self.assertIn("_logits, prediction, elapsed = runner.run(raw)", source)
        self.assertIn('get_session_config_entry("session.disable_cpu_ep_fallback")', source)
        self.assertIn("all_130_predictions_fixed", source)

    def test_launcher_keeps_bundles_read_only_and_path_preimport(self) -> None:
        source = (HERE / "run_k100_2_bundle_cli_pair.sh").read_text(encoding="utf-8")
        self.assertIn('--volume "${bundle}:/bundle:ro"', source)
        self.assertIn('--volume "${evidence}:/run:rw"', source)
        self.assertIn('--env "PATH=${DOCKER_PATH}"', source)
        self.assertIn("[[ -z \"$(docker ps -q)\" ]]", source)
        self.assertNotIn("--skip-full-payload-hash", source)

    def test_claim_boundaries_are_explicit(self) -> None:
        source = (HERE / "aggregate_bundle_cli_pair.py").read_text(encoding="utf-8")
        self.assertIn('"native_int8_kernel_precision_proven_here": False', source)
        self.assertIn('"strict_logits_equivalence_proven_here": False', source)
        self.assertIn('"historical_strict_failure_overridden": False', source)
        self.assertIn("median_ratio <= M5_OVER_FP16_LIMIT", source)
        self.assertIn('"cross_bundle_expected_prediction_identity_required_equal": False', source)
        self.assertIn('"all_predictions_match_own_bundle_expected_sha": True', source)
        self.assertNotIn("len(prediction_hashes) != 1", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
