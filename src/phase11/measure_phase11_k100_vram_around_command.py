#!/usr/bin/env python3
"""Sample device-wide K100 VRAM around an explicitly untimed child command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns, sleep

import numpy as np


SCHEMA = "phase11_command_wrapped_k100_vram_v1"
FINGERPRINT_SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"
EXPECTED_FINGERPRINT_GENERATOR_SHA256 = "f38e24ea72c02a164ec311ce4521f119da1e0f397fc796ea380dc62bb2167577"
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
BASELINE_SAMPLES = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class Sampler:
    def __init__(self, path: Path, interval_ms: float) -> None:
        if interval_ms < 1:
            raise ValueError("sampling interval must be at least 1 ms")
        self.path = path.resolve(strict=True)
        self.interval = interval_ms / 1000.0
        self.values: list[tuple[int, int]] = []
        self.failure: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                value = int(self.path.read_text(encoding="ascii").strip())
                if value < 0:
                    raise RuntimeError("negative VRAM counter")
                self.values.append((perf_counter_ns(), value))
                self.stop_event.wait(self.interval)
        except BaseException as exc:
            self.failure = exc
            self.stop_event.set()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("sampler thread did not stop")
        if self.failure is not None:
            raise RuntimeError(f"sampler failed: {self.failure}") from self.failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument(
        "--vram-counter",
        type=Path,
        default=Path("/sys/class/drm/card1/device/mem_info_vram_used"),
    )
    parser.add_argument("--sampling-interval-ms", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise RuntimeError("a child command is required after --")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    started_ns = perf_counter_ns()
    result = {
        "schema": SCHEMA,
        "status": "failed",
        "label": args.label,
        "process": {"pid": os.getpid(), "started_at_utc": started_at},
        "claims": {
            "isolated_k100_device_vram_peak_measured": False,
            "wrapped_command_latency_claim_allowed": False,
            "deployment_ready": False,
        },
    }
    sampler = None
    try:
        if not IMAGE_ID_PATTERN.fullmatch(args.container_image_id):
            raise RuntimeError("container image ID must be a full sha256 digest")
        fingerprint_identity = identity(args.runtime_fingerprint)
        fingerprint = json.loads(args.runtime_fingerprint.read_text(encoding="utf-8"))
        if fingerprint.get("schema") != FINGERPRINT_SCHEMA:
            raise RuntimeError("runtime fingerprint schema drift")
        if fingerprint.get("generator", {}).get("sha256") != (
            EXPECTED_FINGERPRINT_GENERATOR_SHA256
        ):
            raise RuntimeError("runtime fingerprint generator drift")
        sampler = Sampler(args.vram_counter, args.sampling_interval_ms)
        sampler.start()
        while len(sampler.values) < BASELINE_SAMPLES and sampler.failure is None:
            sleep(args.sampling_interval_ms / 1000.0)
        if sampler.failure is not None:
            raise RuntimeError(f"baseline sampling failed: {sampler.failure}")
        baseline = [value for _, value in sampler.values[:BASELINE_SAMPLES]]
        stdout_path = args.output_dir / "wrapped_command.stdout"
        stderr_path = args.output_dir / "wrapped_command.stderr"
        child_started_at = utc_now()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            child = subprocess.Popen(args.command, stdout=stdout, stderr=stderr)
            child_pid = child.pid
            child_returncode = child.wait()
        child_ended_at = utc_now()
        sampler.stop()
        samples = list(sampler.values)
        sampler = None
        if child_returncode != 0:
            raise RuntimeError(f"wrapped command failed with exit code {child_returncode}")
        array = np.asarray(samples, dtype=np.uint64)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < BASELINE_SAMPLES:
            raise RuntimeError("insufficient VRAM samples")
        samples_path = args.output_dir / "vram_samples_perf_ns_bytes.npy"
        np.save(samples_path, array, allow_pickle=False)
        used = array[:, 1]
        baseline_bytes = int(np.median(np.asarray(baseline, dtype=np.uint64)))
        peak_bytes = int(np.max(used))
        result.update(
            {
                "status": "passed",
                "identities": {
                    "measurement_script": identity(Path(__file__)),
                    "runtime_fingerprint": fingerprint_identity,
                    "vram_samples": identity(samples_path),
                    "wrapped_stdout": identity(stdout_path),
                    "wrapped_stderr": identity(stderr_path),
                },
                "runtime": {
                    "container_image_id": args.container_image_id,
                    "runtime_fingerprint_schema": fingerprint["schema"],
                },
                "wrapped_command": {
                    "argv": args.command,
                    "pid": child_pid,
                    "started_at_utc": child_started_at,
                    "ended_at_utc": child_ended_at,
                    "exit_code": child_returncode,
                },
                "protocol": {
                    "scope": (
                        "device-wide K100 VRAM from pre-child baseline through the entire child; "
                        "the child is executed only to exercise its resident sessions"
                    ),
                    "counter": str(args.vram_counter.resolve(strict=True)),
                    "sampling_interval_target_ms": args.sampling_interval_ms,
                    "baseline_sample_count": BASELINE_SAMPLES,
                    "latency_timed": False,
                    "device_isolation_precondition": (
                        "launcher must establish no competing K100 workload immediately before run"
                    ),
                },
                "measurements": {
                    "sample_count": int(array.shape[0]),
                    "baseline_vram_used_bytes": baseline_bytes,
                    "minimum_vram_used_bytes": int(np.min(used)),
                    "peak_vram_used_bytes": peak_bytes,
                    "incremental_peak_over_baseline_bytes": max(0, peak_bytes - baseline_bytes),
                },
                "evidence_boundary": {
                    "separate_from_formal_timed_trials": True,
                    "all_latency_outputs_from_wrapped_command_are_invalid_for_claims": True,
                    "device_wide_counter_not_process_attribution": True,
                    "incremental_value_requires_launcher_device_isolation": True,
                },
            }
        )
        result["claims"]["isolated_k100_device_vram_peak_measured"] = True
    except Exception as exc:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception as stop_exc:
                result["sampler_stop_failure"] = str(stop_exc)
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result["process"].update(
        {
            "ended_at_utc": utc_now(),
            "elapsed_seconds": (perf_counter_ns() - started_ns) / 1.0e9,
        }
    )
    dump(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
