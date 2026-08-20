#!/usr/bin/env python3
"""Runtime acceptance harness for an immutable Phase 11 K100 bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TELEMETRY_INTERVAL_SECONDS = 5.0
TELEMETRY_MIN_INTERVAL_SECONDS = 4.0
TELEMETRY_MAX_INTERVAL_SECONDS = 7.5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_infer_module():
    path = Path(__file__).resolve().parent / "infer_k100.py"
    spec = importlib.util.spec_from_file_location("phase11_bundle_infer", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import bundled inference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = load_infer_module()


def new_output_dir(path: Path, bundle: Path) -> Path:
    path = path.resolve(strict=False)
    bundle = bundle.resolve(strict=True)
    if path == bundle or bundle in path.parents:
        raise RuntimeError(f"acceptance output must be outside immutable bundle: {path}")
    if path.exists():
        raise RuntimeError(f"refusing to overwrite acceptance output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest_and_fixed(bundle: Path, expected_manifest_sha256: str) -> tuple[dict, Path]:
    bundle = bundle.resolve(strict=True)
    if not SHA256_PATTERN.fullmatch(expected_manifest_sha256):
        raise RuntimeError("expected manifest SHA256 must be 64 lowercase hex digits")
    observed = infer.sha256(bundle / "manifest.json")
    if observed != expected_manifest_sha256:
        raise RuntimeError(f"detached manifest SHA256 drift: {observed}")
    manifest, _ = infer.load_manifest(bundle, verify_all_payloads=False)
    relative = manifest["fixed_sample"]["input"]["path"]
    fixed = infer.resolve_payload(bundle, relative)
    return manifest, fixed


def require_image_attestation(manifest: dict) -> str:
    expected = str(manifest["runtime_contract"]["official_image_id"])
    observed = os.environ.get("PHASE11_K100_IMAGE_ID")
    if observed != expected:
        raise RuntimeError(
            "runtime acceptance requires PHASE11_K100_IMAGE_ID to equal the locked official image ID"
        )
    return observed


def write_result(output: Path, result: dict) -> None:
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


def run_infer_child(
    bundle: Path, fixed: Path, output: Path, device: int, expected_manifest_sha256: str
) -> tuple[int, dict | None]:
    cli = Path(__file__).resolve().parent / "infer_k100.py"
    output_npz = output / "output.npz"
    receipt = output / "receipt.json"
    command = [
        sys.executable,
        str(cli),
        "--bundle",
        str(bundle),
        "--input",
        str(fixed),
        "--output",
        str(output_npz),
        "--receipt",
        str(receipt),
        "--device",
        str(device),
        "--expected-manifest-sha256",
        expected_manifest_sha256,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - started
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    parsed = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else None
    if parsed is not None:
        parsed["fresh_process_wall_seconds"] = wall
    return completed.returncode, parsed


def cold_start(args) -> int:
    output = new_output_dir(args.output_dir, args.bundle)
    result = {
        "schema": "phase11_k100_cold_start_5_v1",
        "status": "failed",
        "started_at_utc": utc_now(),
        "claims": {"five_fresh_processes_passed": False, "deployment_ready": False},
    }
    passed = False
    try:
        manifest, fixed = manifest_and_fixed(args.bundle, args.expected_manifest_sha256)
        require_image_attestation(manifest)
        trials = []
        for index in range(1, 6):
            trial = output / f"trial_{index}"
            trial.mkdir()
            returncode, receipt = run_infer_child(
                args.bundle.resolve(strict=True), fixed, trial, args.device, args.expected_manifest_sha256
            )
            row = {"trial": index, "returncode": returncode, "receipt": receipt}
            trials.append(row)
            if returncode != 0 or receipt is None:
                raise RuntimeError(f"cold-start trial {index} failed")
            gate = receipt.get("fixed_sample_gate", {})
            if not gate.get("prediction_matches_expected"):
                raise RuntimeError(f"cold-start trial {index} fixed prediction drift")
            if not receipt.get("runtime", {}).get("image_identity_attested"):
                raise RuntimeError(f"cold-start trial {index} image identity was not attested")
        load_times = [float(row["receipt"]["timing"]["session_load_seconds"]) for row in trials]
        total_times = [float(row["receipt"]["timing"]["total_runner_load_seconds"]) for row in trials]
        passed = len(trials) == 5 and all(row["returncode"] == 0 for row in trials)
        result.update(
            {
                "status": "passed" if passed else "failed",
                "bundle_id": manifest["bundle_id"],
                "manifest_sha256": infer.sha256(args.bundle.resolve(strict=True) / "manifest.json"),
                "protocol": {
                    "fresh_python_processes": 5,
                    "full_payload_hash_each_process": True,
                    "fixed_prediction_sha_gate_each_process": True,
                },
                "trials": trials,
                "summary": {
                    "session_load_seconds_median": float(np.median(load_times)),
                    "session_load_seconds_min": min(load_times),
                    "session_load_seconds_max": max(load_times),
                    "total_runner_load_seconds_median": float(np.median(total_times)),
                },
            }
        )
        result["claims"]["five_fresh_processes_passed"] = passed
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    result["ended_at_utc"] = utc_now()
    write_result(output, result)
    return 0 if passed else 2


def read_optional_int(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="ascii").strip())
        return value if value >= 0 else None
    except (OSError, ValueError):
        return None


def telemetry_paths() -> dict[str, list[Path]]:
    return {
        "vram_used_bytes": sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")),
        "temperature_millicelsius": sorted(
            Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/temp1_input")
        ),
        "power_microwatts": sorted(
            list(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/power1_average"))
            + list(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/power1_input"))
        ),
    }


def _numbers(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text, flags=re.IGNORECASE)]


def parse_hy_smi(metrics_raw: str, vram_raw: str) -> dict[str, list[float]]:
    """Parse only the three explicitly required official hy-smi fields."""
    return {
        "temperature_c": _numbers(
            r"Temperature\s*\([^\r\n]*?\)\s*\(C\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            metrics_raw,
        ),
        "power_w": _numbers(
            r"Average\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            metrics_raw,
        ),
        "vram_used_bytes": [
            value * 1024.0 * 1024.0
            for value in _numbers(
                r"VRAM\s+USED\s*\(MiB\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                vram_raw,
            )
        ],
    }


def canonical_metrics(sysfs: dict, parsed_hy_smi: dict) -> dict:
    conversions = {
        "temperature_c": ("temperature_millicelsius", 1.0 / 1000.0),
        "power_w": ("power_microwatts", 1.0 / 1_000_000.0),
        "vram_used_bytes": ("vram_used_bytes", 1.0),
    }
    result = {}
    for target, (source_name, scale) in conversions.items():
        sysfs_values = [
            float(item["value"]) * scale
            for item in sysfs.get(source_name, [])
            if item.get("value") is not None
        ]
        hy_values = [float(value) for value in parsed_hy_smi.get(target, [])]
        result[target] = {
            "values": sysfs_values if sysfs_values else hy_values,
            "source": "sysfs" if sysfs_values else "hy-smi" if hy_values else None,
        }
    return result


def telemetry_coverage(rows: list[dict]) -> dict:
    required = ("temperature_c", "power_w", "vram_used_bytes")
    timestamps = [int(row["monotonic_ns"]) for row in rows]
    intervals = [
        (right - left) / 1.0e9 for left, right in zip(timestamps, timestamps[1:])
    ]
    coverage_counts = {
        name: sum(bool(row.get("metrics", {}).get(name, {}).get("values")) for row in rows)
        for name in required
    }
    all_three_each_sample = bool(rows) and all(
        all(row.get("metrics", {}).get(name, {}).get("values") for name in required)
        for row in rows
    )
    value_summary = {}
    for name in required:
        values = [
            float(value)
            for row in rows
            for value in row.get("metrics", {}).get(name, {}).get("values", [])
        ]
        value_summary[name] = {
            "value_count": len(values),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return {
        "samples": len(rows),
        "required_metrics": list(required),
        "coverage_sample_counts": coverage_counts,
        "all_three_metrics_present_in_every_sample": all_three_each_sample,
        "interval_count": len(intervals),
        "minimum_interval_seconds": min(intervals) if intervals else None,
        "maximum_interval_seconds": max(intervals) if intervals else None,
        "intervals_strictly_positive": bool(intervals) and all(value > 0 for value in intervals),
        "minimum_interval_ge_4_seconds": bool(intervals)
        and min(intervals) >= TELEMETRY_MIN_INTERVAL_SECONDS,
        "maximum_interval_le_7_5_seconds": bool(intervals)
        and max(intervals) <= TELEMETRY_MAX_INTERVAL_SECONDS,
        "values": value_summary,
    }


class TelemetrySampler:
    def __init__(self, path: Path, interval_seconds: float) -> None:
        if interval_seconds != TELEMETRY_INTERVAL_SECONDS:
            raise RuntimeError("formal stability telemetry interval is fixed at 5 seconds")
        self.path = path
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.failure: BaseException | None = None
        self.count = 0
        self.counter_lock = threading.Lock()
        self.runtime_counters = {
            "inferences": 0,
            "errors": 0,
            "nonfinite_outputs": 0,
            "fixed_prediction_drifts": 0,
        }
        self.sysfs = telemetry_paths()
        self.smi = Path("/usr/local/hyhal/bin/hy-smi")
        self.thread = threading.Thread(target=self._run, name="k100-telemetry", daemon=True)

    def _sample(self) -> dict:
        values: dict[str, list[dict]] = {}
        for name, paths in self.sysfs.items():
            values[name] = [
                {"path": str(path), "value": read_optional_int(path)} for path in paths
            ]
        metrics_raw = ""
        metrics_returncode = None
        vram_raw = ""
        vram_returncode = None
        if self.smi.is_file():
            completed = subprocess.run(
                [str(self.smi), "--showuse", "--showmemuse", "--showtemp", "--showpower"],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            metrics_returncode = completed.returncode
            metrics_raw = (completed.stdout + completed.stderr).strip()
            vram = subprocess.run(
                [str(self.smi), "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            vram_returncode = vram.returncode
            vram_raw = (vram.stdout + vram.stderr).strip()
        parsed = parse_hy_smi(
            metrics_raw if metrics_returncode == 0 else "",
            vram_raw if vram_returncode == 0 else "",
        )
        metrics = canonical_metrics(values, parsed)
        with self.counter_lock:
            counters = dict(self.runtime_counters)
        return {
            "utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "sysfs": values,
            "hy_smi": {
                "available": self.smi.is_file(),
                "metrics_returncode": metrics_returncode,
                "metrics_raw": metrics_raw,
                "vram_returncode": vram_returncode,
                "vram_raw": vram_raw,
                "parsed": parsed,
            },
            "metrics": metrics,
            "runtime_counters": counters,
        }

    def update_counters(
        self, inferences: int, errors: int, nonfinite_outputs: int, fixed_prediction_drifts: int
    ) -> None:
        with self.counter_lock:
            self.runtime_counters = {
                "inferences": int(inferences),
                "errors": int(errors),
                "nonfinite_outputs": int(nonfinite_outputs),
                "fixed_prediction_drifts": int(fixed_prediction_drifts),
            }

    def _run(self) -> None:
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    stream.write(json.dumps(self._sample(), ensure_ascii=False) + "\n")
                    stream.flush()
                    self.count += 1
                    remaining = max(0.0, self.interval - (time.monotonic() - started))
                    self.stop_event.wait(remaining)
        except BaseException as exc:
            self.failure = exc
            self.stop_event.set()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("telemetry thread did not stop")
        if self.failure is not None:
            raise RuntimeError(f"telemetry failed: {self.failure}") from self.failure


def summarize_telemetry(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return telemetry_coverage(rows)


def stability(args) -> int:
    output = new_output_dir(args.output_dir, args.bundle)
    result = {
        "schema": "phase11_k100_stability_60min_v1",
        "status": "failed",
        "started_at_utc": utc_now(),
        "claims": {"continuous_60min_passed": False, "deployment_ready": False},
    }
    sampler = None
    passed = False
    try:
        manifest, fixed = manifest_and_fixed(args.bundle, args.expected_manifest_sha256)
        require_image_attestation(manifest)
        raw = infer.load_input(fixed)
        runner = infer.K100BundleRunner(args.bundle, args.device, verify_payloads=True)
        telemetry_file = output / "telemetry_5s.jsonl"
        sampler = TelemetrySampler(telemetry_file, TELEMETRY_INTERVAL_SECONDS)
        sampler.start()
        deadline = time.monotonic() + args.duration_seconds
        iterations = 0
        errors = 0
        nonfinite = 0
        prediction_drifts = 0
        inference_seconds = []
        while time.monotonic() < deadline:
            try:
                logits, prediction, elapsed = runner.run(raw)
                iterations += 1
                inference_seconds.append(elapsed)
                if not np.isfinite(logits).all():
                    nonfinite += 1
                if not runner.fixed_prediction_gate(raw, prediction)["prediction_matches_expected"]:
                    prediction_drifts += 1
            except Exception:
                errors += 1
                with (output / "inference_failures.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"{utc_now()}\n{traceback.format_exc()}\n")
            sampler.update_counters(iterations, errors, nonfinite, prediction_drifts)
            if args.max_inferences and iterations >= args.max_inferences:
                break
        sampler.stop()
        sampler = None
        elapsed_wall = args.duration_seconds - max(0.0, deadline - time.monotonic())
        telemetry = summarize_telemetry(telemetry_file)
        minimum_samples = int(args.duration_seconds // TELEMETRY_INTERVAL_SECONDS)
        formal_duration = args.duration_seconds >= 3600 and args.max_inferences is None
        gates = {
            "duration_at_least_3600_seconds": formal_duration and elapsed_wall >= 3600,
            "at_least_one_inference": iterations > 0,
            "zero_inference_errors": errors == 0,
            "zero_nonfinite_outputs": nonfinite == 0,
            "zero_fixed_prediction_drifts": prediction_drifts == 0,
            "telemetry_interval_eq_5_seconds": TELEMETRY_INTERVAL_SECONDS == 5.0,
            "telemetry_sample_count_sufficient": telemetry["samples"] >= minimum_samples,
            "temperature_power_vram_present_every_sample": telemetry[
                "all_three_metrics_present_in_every_sample"
            ],
            "telemetry_intervals_strictly_positive": telemetry["intervals_strictly_positive"],
            "telemetry_minimum_interval_ge_4_seconds": telemetry[
                "minimum_interval_ge_4_seconds"
            ],
            "telemetry_maximum_interval_le_7_5_seconds": telemetry[
                "maximum_interval_le_7_5_seconds"
            ],
            "image_identity_attested": runner.runtime["image_identity_attested"],
        }
        passed = all(gates.values())
        result.update(
            {
                "status": "passed" if passed else "diagnostic_completed",
                "bundle_id": manifest["bundle_id"],
                "manifest_sha256": runner.manifest_identity["sha256"],
                "protocol": {
                    "requested_duration_seconds": args.duration_seconds,
                    "observed_duration_seconds": elapsed_wall,
                    "telemetry_interval_seconds": 5,
                    "telemetry_allowed_interval_seconds": [
                        TELEMETRY_MIN_INTERVAL_SECONDS,
                        TELEMETRY_MAX_INTERVAL_SECONDS,
                    ],
                    "max_inferences": args.max_inferences,
                },
                "measurements": {
                    "inferences": iterations,
                    "errors": errors,
                    "error_rate": float(errors / (iterations + errors)) if (iterations + errors) else None,
                    "nonfinite_outputs": nonfinite,
                    "fixed_prediction_drifts": prediction_drifts,
                    "inference_seconds_median": float(np.median(inference_seconds)) if inference_seconds else None,
                    "inference_seconds_p95": float(np.percentile(inference_seconds, 95)) if inference_seconds else None,
                    "telemetry": telemetry,
                },
                "gates": gates,
            }
        )
        result["claims"]["continuous_60min_passed"] = passed
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception as exc:
                result["telemetry_stop_failure"] = str(exc)
    result["ended_at_utc"] = utc_now()
    write_result(output, result)
    return 0 if passed else 2


def worker(args) -> int:
    manifest, fixed = manifest_and_fixed(args.bundle, args.expected_manifest_sha256)
    require_image_attestation(manifest)
    raw = infer.load_input(fixed)
    runner = infer.K100BundleRunner(args.bundle, args.device, verify_payloads=True)
    logits, prediction, _ = runner.run(raw)
    gate = runner.fixed_prediction_gate(raw, prediction)
    if not gate["prediction_matches_expected"] or not np.isfinite(logits).all():
        raise RuntimeError("recovery worker initial fixed-sample gate failed")
    args.ready.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"pid": os.getpid(), "ready_at_utc": utc_now(), "fixed_sample_gate": gate}) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.ready)
    while True:
        logits, prediction, _ = runner.run(raw)
        if not np.isfinite(logits).all() or not runner.fixed_prediction_gate(raw, prediction)[
            "prediction_matches_expected"
        ]:
            raise RuntimeError("recovery worker runtime drift")


def recovery(args) -> int:
    output = new_output_dir(args.output_dir, args.bundle)
    result = {
        "schema": "phase11_k100_recovery_3_v1",
        "status": "failed",
        "started_at_utc": utc_now(),
        "claims": {"three_active_termination_reloads_passed": False, "deployment_ready": False},
    }
    passed = False
    active_process: subprocess.Popen | None = None
    handles = []
    try:
        manifest, fixed = manifest_and_fixed(args.bundle, args.expected_manifest_sha256)
        require_image_attestation(manifest)
        cycles = []
        for index in range(1, 4):
            cycle = output / f"cycle_{index}"
            cycle.mkdir()
            ready = cycle / "worker_ready.json"
            stdout_handle = (cycle / "worker_stdout.log").open("x", encoding="utf-8")
            stderr_handle = (cycle / "worker_stderr.log").open("x", encoding="utf-8")
            handles = [stdout_handle, stderr_handle]
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--bundle",
                str(args.bundle.resolve(strict=True)),
                "--ready",
                str(ready),
                "--expected-manifest-sha256",
                args.expected_manifest_sha256,
                "--device",
                str(args.device),
            ]
            active_process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
            deadline = time.monotonic() + args.ready_timeout_seconds
            while not ready.is_file() and active_process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.25)
            if not ready.is_file():
                raise RuntimeError(f"recovery cycle {index} worker did not become ready")
            active_process.terminate()
            try:
                terminated_code = active_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                active_process.kill()
                terminated_code = active_process.wait(timeout=30)
            active_process = None
            for handle in handles:
                handle.close()
            handles = []
            recovery_dir = cycle / "reloaded"
            recovery_dir.mkdir()
            returncode, receipt = run_infer_child(
                args.bundle.resolve(strict=True),
                fixed,
                recovery_dir,
                args.device,
                args.expected_manifest_sha256,
            )
            fixed_ok = bool(receipt and receipt.get("fixed_sample_gate", {}).get("prediction_matches_expected"))
            cycles.append(
                {
                    "cycle": index,
                    "worker_ready": True,
                    "active_termination_returncode": terminated_code,
                    "reload_returncode": returncode,
                    "reload_fixed_prediction_passed": fixed_ok,
                    "reload_receipt": receipt,
                }
            )
            if returncode != 0 or not fixed_ok:
                raise RuntimeError(f"recovery cycle {index} reload failed")
        passed = len(cycles) == 3 and all(row["reload_fixed_prediction_passed"] for row in cycles)
        result.update(
            {
                "status": "passed" if passed else "failed",
                "bundle_id": manifest["bundle_id"],
                "manifest_sha256": infer.sha256(args.bundle.resolve(strict=True) / "manifest.json"),
                "protocol": {"active_termination_cycles": 3, "fresh_reload_process_each_cycle": True},
                "cycles": cycles,
            }
        )
        result["claims"]["three_active_termination_reloads_passed"] = passed
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if active_process is not None:
            active_process.kill()
            active_process.wait()
        for handle in handles:
            handle.close()
    result["ended_at_utc"] = utc_now()
    write_result(output, result)
    return 0 if passed else 2


def cross_node_smoke(args) -> int:
    output = new_output_dir(args.output_dir, args.bundle)
    result = {
        "schema": "phase11_k100_cross_node_smoke_v1",
        "status": "failed",
        "started_at_utc": utc_now(),
        "claims": {"target_cache_smoke_passed": False, "cache_portability_verified": False, "deployment_ready": False},
    }
    passed = False
    try:
        if args.node_label not in {"K100-2", "K100-3"}:
            raise RuntimeError("formal cross-node label must be K100-2 or K100-3")
        manifest, fixed = manifest_and_fixed(args.bundle, args.expected_manifest_sha256)
        image_id = require_image_attestation(manifest)
        fingerprint_path = output / "target_runtime_fingerprint.json"
        capture = Path(__file__).resolve().parent / "capture_k100_runtime_fingerprint.py"
        locked_lsmod = args.bundle.resolve(strict=True) / "tools/bin/lsmod"
        completed = subprocess.run(
            [
                sys.executable,
                str(capture),
                "--official-image-id",
                image_id,
                "--lsmod-shim",
                str(locked_lsmod),
                "--output",
                str(fingerprint_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (output / "fingerprint_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "fingerprint_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not fingerprint_path.is_file():
            raise RuntimeError("target runtime fingerprint capture failed")
        target = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        expected_signature = manifest["runtime_contract"]["portable_runtime_signature_sha256"]
        signature_match = target.get("portable_runtime_signature_sha256") == expected_signature
        if not signature_match:
            raise RuntimeError("target software runtime signature differs from bundle origin")
        infer_dir = output / "fixed_sample_inference"
        infer_dir.mkdir()
        returncode, receipt = run_infer_child(
            args.bundle.resolve(strict=True),
            fixed,
            infer_dir,
            args.device,
            args.expected_manifest_sha256,
        )
        fixed_ok = bool(receipt and receipt.get("fixed_sample_gate", {}).get("prediction_matches_expected"))
        passed = returncode == 0 and fixed_ok
        result.update(
            {
                "status": "passed" if passed else "failed",
                "bundle_id": manifest["bundle_id"],
                "manifest_sha256": infer.sha256(args.bundle.resolve(strict=True) / "manifest.json"),
                "node_label": args.node_label,
                "hostname": platform.node(),
                "runtime_signature_match": signature_match,
                "inference_returncode": returncode,
                "fixed_prediction_passed": fixed_ok,
                "inference_receipt": receipt,
                "failure_policy": (
                    "If cache loading fails, recompile on this target from locked ONNX, create a new bundle "
                    "manifest with the new MXR SHA, and rerun; never mutate the origin bundle."
                ),
            }
        )
        result["claims"]["target_cache_smoke_passed"] = passed
        result["claims"]["cache_portability_verified"] = passed
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    result["ended_at_utc"] = utc_now()
    write_result(output, result)
    return 0 if passed else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    cold = sub.add_parser("cold-start")
    cold.add_argument("--bundle", type=Path, required=True)
    cold.add_argument("--expected-manifest-sha256", required=True)
    cold.add_argument("--output-dir", type=Path, required=True)
    cold.add_argument("--device", type=int, default=0)
    cold.set_defaults(function=cold_start)

    stable = sub.add_parser("stability")
    stable.add_argument("--bundle", type=Path, required=True)
    stable.add_argument("--expected-manifest-sha256", required=True)
    stable.add_argument("--output-dir", type=Path, required=True)
    stable.add_argument("--device", type=int, default=0)
    stable.add_argument("--duration-seconds", type=int, default=3600)
    stable.add_argument("--max-inferences", type=int)
    stable.set_defaults(function=stability)

    recover = sub.add_parser("recovery")
    recover.add_argument("--bundle", type=Path, required=True)
    recover.add_argument("--expected-manifest-sha256", required=True)
    recover.add_argument("--output-dir", type=Path, required=True)
    recover.add_argument("--device", type=int, default=0)
    recover.add_argument("--ready-timeout-seconds", type=int, default=900)
    recover.set_defaults(function=recovery)

    smoke = sub.add_parser("cross-node-smoke")
    smoke.add_argument("--bundle", type=Path, required=True)
    smoke.add_argument("--expected-manifest-sha256", required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device", type=int, default=0)
    smoke.add_argument("--node-label", required=True)
    smoke.set_defaults(function=cross_node_smoke)

    hidden = sub.add_parser("_worker")
    hidden.add_argument("--bundle", type=Path, required=True)
    hidden.add_argument("--expected-manifest-sha256", required=True)
    hidden.add_argument("--ready", type=Path, required=True)
    hidden.add_argument("--device", type=int, default=0)
    hidden.set_defaults(function=worker)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "duration_seconds", 1) <= 0:
        raise RuntimeError("duration must be positive")
    if getattr(args, "max_inferences", None) is not None and args.max_inferences <= 0:
        raise RuntimeError("max-inferences must be positive")
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
