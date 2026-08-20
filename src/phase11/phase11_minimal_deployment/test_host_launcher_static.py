#!/usr/bin/env python3
"""CPU-only regression tests for host-owned formal smoke evidence roots."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "host_launcher", ROOT / "launch_cross_node_smoke_container.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load host launcher")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class HostLauncherTests(unittest.TestCase):
    image_id = "sha256:" + "a" * 64

    def fixture(self, base: Path) -> tuple[Path, str, Path]:
        bundle = base / "bundle"
        shim = bundle / "tools" / "bin" / "lsmod"
        shim.parent.mkdir(parents=True)
        shim.write_bytes(b"test shim")
        launcher.LSMOD_SIZE = shim.stat().st_size
        launcher.LSMOD_SHA256 = launcher.sha256(shim)
        manifest = {
            "bundle_id": "test-bundle",
            "runtime_contract": {"official_image_id": self.image_id},
        }
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        hyhal = base / "hyhal-real"
        hyhal.mkdir()
        return bundle, launcher.sha256(manifest_path), hyhal

    def invoke(
        self, bundle: Path, manifest_sha: str, hyhal: Path, output: Path, docker_ok: bool
    ) -> dict:
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            if argv[1:3] == ["image", "inspect"]:
                return CompletedProcess(argv, 0, stdout=self.image_id + "\n", stderr="")
            self.assertEqual(argv[1:3], ["run", "--rm"])
            # This is the regression gate: the invoking host owns/creates the
            # formal root before Docker starts.
            self.assertTrue(output.is_dir())
            self.assertFalse(output.is_symlink())
            self.assertIn(f"{output.resolve()}:/acceptance", argv)
            if not docker_ok:
                return CompletedProcess(argv, 2, stdout="", stderr="docker failed")
            child = output / "container_smoke"
            child.mkdir()
            (child / "result.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            (child / "target_runtime_fingerprint.json").write_text(
                json.dumps({"status": "captured"}), encoding="utf-8"
            )
            return CompletedProcess(argv, 0, stdout="passed", stderr="")

        argv = [
            "launch_cross_node_smoke_container.py",
            "--bundle",
            str(bundle),
            "--expected-manifest-sha256",
            manifest_sha,
            "--output-dir",
            str(output),
            "--official-image-id",
            self.image_id,
            "--host-hyhal",
            str(hyhal),
            "--node-label",
            "K100-3",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            launcher.subprocess, "run", side_effect=fake_run
        ), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            launcher.main()
        self.assertEqual(stopped.exception.code, 0 if docker_ok else 2)
        receipt = json.loads(
            (output / "container_launch_attestation.json").read_text(encoding="utf-8")
        )
        return receipt

    def test_host_precreates_root_and_container_writes_only_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle, manifest_sha, hyhal = self.fixture(base)
            output = base / "acceptance" / "smoke_k100_3"
            output.parent.mkdir()
            receipt = self.invoke(bundle, manifest_sha, hyhal, output, docker_ok=True)
            self.assertEqual(receipt["status"], "passed")
            self.assertTrue(
                receipt["host_output_ownership"]["created_exclusively_before_docker"]
            )
            self.assertEqual(receipt["container_output_dir"], "/acceptance/container_smoke")
            self.assertEqual(
                receipt["container_smoke_result"]["sha256"],
                receipt["smoke_result"]["sha256"],
            )

    def test_docker_failure_still_writes_failed_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle, manifest_sha, hyhal = self.fixture(base)
            output = base / "acceptance" / "smoke_k100_3_failed"
            output.parent.mkdir()
            receipt = self.invoke(bundle, manifest_sha, hyhal, output, docker_ok=False)
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["claims"]["formal_receipts_copied_without_byte_drift"])
            self.assertTrue((output / "container_launcher_stderr.log").is_file())

    def test_hyhal_symlink_is_resolved_before_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle, manifest_sha, hyhal = self.fixture(base)
            link = base / "hyhal-link"
            try:
                os.symlink(hyhal, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")
            output = base / "acceptance" / "smoke_k100_3_symlink"
            output.parent.mkdir()
            receipt = self.invoke(bundle, manifest_sha, link, output, docker_ok=True)
            mount = receipt["host_hyhal_mount"]
            self.assertTrue(mount["requested_source_was_symlink"])
            self.assertEqual(Path(mount["source"]), hyhal.resolve())
            self.assertIn(f"{hyhal.resolve()}:/opt/hyhal:ro", receipt["docker_argv"])


if __name__ == "__main__":
    unittest.main()
