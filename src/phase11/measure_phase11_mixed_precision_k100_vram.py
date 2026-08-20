#!/usr/bin/env python3
"""Measure isolated K100 device-VRAM peak for one admitted mixed candidate.

This is deliberately a separate, untimed process.  The sampler observes the
device-wide sysfs counter while all 25 sessions are loaded, output buffers are
allocated, and 30+100 model-only inferences execute.  Its result must never be
merged into or interpreted as latency timing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, perf_counter_ns, sleep

import numpy as np

import phase11_mixed_precision_common as common


SCHEMA = "phase11_mixed_precision_k100_vram_v1"
WARMUP = 30
EXERCISE = 100
BASELINE_SAMPLES = 20
EXPECTED_TEST90_AGGREGATE_SCRIPT_SHA256 = "717aa6c5e10b889c082acb4626105ad708c9006acf70380793f0b93c71fbd159"
EXPECTED_COMMON_MODULE_SHA256 = "f445092c61a731105ba47d1128c12b52e8b658b8b9364f5afef4522ca6a5e5a8"
EXPECTED_RUNTIME_FINGERPRINT_GENERATOR_SHA256 = "f38e24ea72c02a164ec311ce4521f119da1e0f397fc796ea380dc62bb2167577"
RUNTIME_FINGERPRINT_SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VramSampler:
    def __init__(self, counter_path: Path, interval_ms: float) -> None:
        if interval_ms < 1.0:
            raise ValueError("VRAM sampling interval must be at least 1 ms")
        self.counter_path = counter_path.resolve(strict=True)
        self.interval_seconds = interval_ms / 1000.0
        self.samples: list[tuple[int, int]] = []
        self.failure: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="k100-vram-sampler", daemon=True)

    def _read(self) -> int:
        value = int(self.counter_path.read_text(encoding="ascii").strip())
        if value < 0:
            raise RuntimeError("negative K100 VRAM counter")
        return value

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.samples.append((perf_counter_ns(), self._read()))
                self.stop_event.wait(self.interval_seconds)
        except BaseException as exc:  # retained and raised on the main thread
            self.failure = exc
            self.stop_event.set()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("VRAM sampler thread did not stop")
        if self.failure is not None:
            raise RuntimeError(f"VRAM sampler failed: {self.failure}") from self.failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test90-three-run-summary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--runtime-fingerprint", type=Path, required=True)
    parser.add_argument(
        "--vram-counter",
        type=Path,
        default=Path("/sys/class/drm/card1/device/mem_info_vram_used"),
    )
    parser.add_argument("--sampling-interval-ms", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    process_started_at_utc = utc_now()
    process_started_ns = perf_counter_ns()
    result = {
        "schema": SCHEMA,
        "status": "failed",
        "claims": {
            "isolated_k100_device_vram_peak_measured": False,
            "latency_measured": False,
            "strict_numeric_equivalence_confirmed": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
        "process": {"pid": os.getpid(), "started_at_utc": process_started_at_utc},
    }
    sampler: VramSampler | None = None
    passed = False
    try:
        ort = common.require_runtime()
        if not IMAGE_ID_PATTERN.fullmatch(args.container_image_id):
            raise RuntimeError("container image ID must be a full sha256 digest")
        fingerprint_identity = common.identity(args.runtime_fingerprint)
        fingerprint = common.load_json(args.runtime_fingerprint)
        if fingerprint.get("schema") != RUNTIME_FINGERPRINT_SCHEMA:
            raise RuntimeError("runtime fingerprint schema drift")
        if fingerprint.get("onnxruntime") != ort.__version__:
            raise RuntimeError("runtime fingerprint ORT version drift")
        if fingerprint.get("generator", {}).get("sha256") != (
            EXPECTED_RUNTIME_FINGERPRINT_GENERATOR_SHA256
        ):
            raise RuntimeError("runtime fingerprint generator identity drift")
        manifest, manifest_identity, rows = common.load_candidate(args.bundle, args.manifest)
        common.validate_onnx_contracts(rows)
        admission_identity = common.identity(args.test90_three_run_summary)
        admission = common.load_json(args.test90_three_run_summary)
        if admission.get("schema") != "phase11_mixed_precision_test90_three_run_summary_v1":
            raise RuntimeError("test90 three-run prerequisite schema drift")
        if admission.get("status") != "passed" or not admission.get("claims", {}).get(
            "task_equivalence_performance_track_may_proceed"
        ):
            raise RuntimeError("test90 three-run task admission did not pass")
        if admission.get("identities", {}).get("aggregate_script", {}).get("sha256") != (
            EXPECTED_TEST90_AGGREGATE_SCRIPT_SHA256
        ):
            raise RuntimeError("test90 aggregate script identity drift")
        if admission.get("identities", {}).get("common_module", {}).get("sha256") != (
            EXPECTED_COMMON_MODULE_SHA256
        ):
            raise RuntimeError("test90 aggregate common module identity drift")
        common.validate_result_candidate(admission, manifest_identity, manifest["candidate_id"])

        raw = common.load_raw_sample(args.sample)
        sampler = VramSampler(args.vram_counter, args.sampling_interval_ms)
        sampler.start()
        while len(sampler.samples) < BASELINE_SAMPLES and sampler.failure is None:
            sleep(args.sampling_interval_ms / 1000.0)
        if sampler.failure is not None:
            raise RuntimeError(f"VRAM baseline sampling failed: {sampler.failure}")
        baseline_values = [value for _, value in sampler.samples[:BASELINE_SAMPLES]]

        load_started = perf_counter()
        sessions = [common.create_migraphx_session(ort, row) for row in rows]
        session_load_seconds = perf_counter() - load_started
        if len(sessions) != 25:
            raise RuntimeError("all 25 sessions are not concurrently resident")

        device_input = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        allocated_outputs: list[dict] = []
        retained = {}
        current = device_input
        for index, session in enumerate(sessions):
            binding = session.io_binding()
            inputs = session.get_inputs()
            if index < 24:
                binding.bind_ortvalue_input(inputs[0].name, current)
            else:
                if {item.name for item in inputs} != set(retained):
                    raise RuntimeError("head retained-feature contract drift")
                for item in inputs:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                binding.bind_output(name, "cuda", 0)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            values = binding.get_outputs()
            if not all(value.device_name() == "cuda" for value in values):
                raise RuntimeError(f"output allocation left device at segment {index}")
            output_map = dict(zip(output_names, values, strict=True))
            allocated_outputs.append(output_map)
            current = common.choose_primary_output(index, output_map)
            if index in common.RETAIN_AFTER:
                retained[output_names[0]] = current

        fixed_bindings = []
        retained = {}
        current = device_input
        for index, (session, output_map) in enumerate(
            zip(sessions, allocated_outputs, strict=True)
        ):
            binding = session.io_binding()
            inputs = session.get_inputs()
            if index < 24:
                binding.bind_ortvalue_input(inputs[0].name, current)
            else:
                for item in inputs:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            for name, output_value in output_map.items():
                binding.bind_ortvalue_output(name, output_value)
            binding.synchronize_inputs()
            fixed_bindings.append(binding)
            current = common.choose_primary_output(index, output_map)
            if index in common.RETAIN_AFTER:
                retained[next(iter(output_map))] = current

        def run_model_only() -> None:
            for session, binding in zip(sessions, fixed_bindings, strict=True):
                session.run_with_iobinding(binding)
            fixed_bindings[-1].synchronize_outputs()

        def run_end_to_end() -> np.ndarray:
            fresh_input = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
            first_binding = sessions[0].io_binding()
            first_binding.bind_ortvalue_input(sessions[0].get_inputs()[0].name, fresh_input)
            for name, output_value in allocated_outputs[0].items():
                first_binding.bind_ortvalue_output(name, output_value)
            first_binding.synchronize_inputs()
            sessions[0].run_with_iobinding(first_binding)
            for index in range(1, 25):
                sessions[index].run_with_iobinding(fixed_bindings[index])
            fixed_bindings[-1].synchronize_outputs()
            return np.ascontiguousarray(
                common.choose_primary_output(24, allocated_outputs[-1]).numpy(),
                dtype=np.float32,
            )

        for _ in range(WARMUP):
            run_model_only()
        for _ in range(EXERCISE):
            run_model_only()
        logits = None
        for _ in range(WARMUP):
            logits = run_end_to_end()
        for _ in range(EXERCISE):
            logits = run_end_to_end()
        if logits is None:
            raise RuntimeError("end-to-end VRAM exercise did not execute")
        output_finite = logits.shape == (1, 2, 224, 224) and np.isfinite(logits).all()
        if not output_finite:
            raise RuntimeError("VRAM exercise final logits contract failure")
        sampler.stop()
        samples = list(sampler.samples)
        sampler = None
        if len(samples) < BASELINE_SAMPLES:
            raise RuntimeError("insufficient K100 VRAM samples")
        sample_array = np.asarray(samples, dtype=np.uint64)
        samples_path = args.output_dir / "vram_samples_perf_ns_bytes.npy"
        np.save(samples_path, sample_array, allow_pickle=False)
        used_values = sample_array[:, 1].astype(np.uint64, copy=False)
        baseline_vram_used_bytes = int(np.median(np.asarray(baseline_values, dtype=np.uint64)))
        minimum_vram_used_bytes = int(np.min(used_values))
        peak_vram_used_bytes = int(np.max(used_values))
        incremental_peak_bytes = max(0, peak_vram_used_bytes - baseline_vram_used_bytes)
        gates = {
            "task90_three_run_admission_passed": True,
            "session_count_eq_25": len(sessions) == 25,
            "all_intersegment_allocations_on_device": True,
            "warmup_eq_30": WARMUP == 30,
            "exercise_iterations_per_scope_eq_100": EXERCISE == 100,
            "baseline_samples_ge_20": len(baseline_values) >= BASELINE_SAMPLES,
            "vram_samples_nonnegative": bool(np.all(used_values >= 0)),
            "peak_ge_baseline": peak_vram_used_bytes >= baseline_vram_used_bytes,
            "final_output_finite": bool(output_finite),
        }
        passed = all(gates.values())
        result.update(
            {
                "status": "passed" if passed else "failed",
                "candidate_lineage": {
                    "candidate_id": manifest["candidate_id"],
                    "manifest": manifest_identity,
                    "bundle": str(args.bundle.resolve(strict=True)),
                    "fp16_backbone_blocks": manifest["fp16_backbone_blocks"],
                    "int8_backbone_blocks": manifest["int8_backbone_blocks"],
                },
                "identities": {
                    "measurement_script": common.identity(Path(__file__)),
                    "common_module": common.identity(Path(common.__file__)),
                    "sample": common.identity(args.sample),
                    "test90_three_run_summary": admission_identity,
                    "runtime_fingerprint": fingerprint_identity,
                    "vram_samples": common.identity(samples_path),
                },
                "runtime": {
                    "onnxruntime": ort.__version__,
                    "container_image_id": args.container_image_id,
                    "runtime_fingerprint_schema": fingerprint["schema"],
                    "session_load_seconds_descriptive": session_load_seconds,
                },
                "protocol": {
                    "scope": (
                        "device-wide K100 VRAM used counter from pre-session baseline through "
                        "25 resident MIGraphX sessions, fixed device OrtValues, 30 warmups, "
                        "100 untimed model-only exercises, then 30 warmups and 100 untimed "
                        "host-input-to-logits exercises"
                    ),
                    "counter": str(args.vram_counter.resolve(strict=True)),
                    "sampling_interval_target_ms": args.sampling_interval_ms,
                    "baseline_sample_count": BASELINE_SAMPLES,
                    "warmup_per_scope": WARMUP,
                    "exercise_iterations_per_scope": EXERCISE,
                    "scope_order": ["model_only", "end_to_end_logits"],
                    "latency_timed": False,
                    "fresh_process": True,
                    "device_isolation_precondition": (
                        "launcher must establish no competing K100 workload immediately before run"
                    ),
                },
                "measurements": {
                    "sample_count": int(sample_array.shape[0]),
                    "baseline_vram_used_bytes": baseline_vram_used_bytes,
                    "minimum_vram_used_bytes": minimum_vram_used_bytes,
                    "peak_vram_used_bytes": peak_vram_used_bytes,
                    "incremental_peak_over_baseline_bytes": incremental_peak_bytes,
                },
                "gates": gates,
                "evidence_boundary": {
                    "separate_from_all_timed_trials": True,
                    "device_wide_counter_not_process_attribution": True,
                    "incremental_value_requires_launcher_device_isolation": True,
                    "sampling_may_perturb_this_untimed_exercise_only": True,
                    "latency_claim_allowed_from_this_result": False,
                },
            }
        )
        result["claims"]["isolated_k100_device_vram_peak_measured"] = passed
    except Exception as exc:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception as sampler_exc:
                result["sampler_stop_failure"] = str(sampler_exc)
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["process"].update(
            {
                "ended_at_utc": utc_now(),
                "elapsed_seconds": (perf_counter_ns() - process_started_ns) / 1.0e9,
            }
        )
    common.json_dump(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
