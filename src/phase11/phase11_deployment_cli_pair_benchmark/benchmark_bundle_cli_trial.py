#!/usr/bin/env python3
"""One fail-closed, steady-state K100 deployment-bundle benchmark trial.

The script is executed once per candidate in a fresh container.  It imports the
SHA-locked ``infer_k100.py`` from that candidate's immutable bundle and times
only the third value returned by ``K100BundleRunner.run``.  That value covers
the strict float32 input-to-synchronized-float32-logits path; bundle hashing,
session/cache loading, prediction hashing, and JSON writing are outside it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_ID = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
M5_MANIFEST_SHA256 = "f4dc433595c43332d4e29c2351363b3245f214273f058551788d50be29c388da"
FP16_MANIFEST_SHA256 = "1f433a917aed5fead6dd95597a5a03f500a93ce564067e9dad4d25f4682bad25"
INFER_K100_SHA256 = "fc338d90f7592f6a75cfa28191702598777b21914415e7178cef932de60571b7"
LSMOD_SIZE = 819_664
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"
MGX = "MIGraphXExecutionProvider"
DOCKER_PATH = "/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
WARMUP = 30
EXERCISES = 100

VARIANTS = {
    "m5": {
        "manifest_sha256": M5_MANIFEST_SHA256,
        "bundle_id": "prithvi-k100-m5-v1",
        "execution_kind": "segment25",
        "source_candidate_id": "M5",
    },
    "fp16": {
        "manifest_sha256": FP16_MANIFEST_SHA256,
        "bundle_id": "prithvi-k100-fp16-full-v1",
        "execution_kind": "fp16_full",
        "source_candidate_id": None,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def expected_sequence(trial_index: int) -> tuple[str, str]:
    if trial_index not in (1, 2, 3):
        raise RuntimeError("trial index must be 1, 2, or 3")
    return ("m5", "fp16") if trial_index % 2 else ("fp16", "m5")


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("cannot summarize an empty timing vector")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(seconds: list[float]) -> dict[str, Any]:
    if len(seconds) != EXERCISES or any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise RuntimeError("timing vector must contain exactly 100 finite positive values")
    milliseconds = [value * 1000.0 for value in seconds]
    return {
        "count": len(seconds),
        "median_ms": statistics.median(milliseconds),
        "p95_ms": percentile(milliseconds, 0.95),
        "p99_ms": percentile(milliseconds, 0.99),
        "mean_ms": statistics.fmean(milliseconds),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "throughput_samples_per_second": len(seconds) / sum(seconds),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve(strict=False)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite trial result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def import_infer(bundle: Path):
    infer_candidate = bundle / "tools" / "infer_k100.py"
    if infer_candidate.is_symlink():
        raise RuntimeError("infer_k100.py must be a regular non-symlink bundle payload")
    infer_path = infer_candidate.resolve(strict=True)
    if not infer_path.is_file():
        raise RuntimeError("infer_k100.py must be a regular non-symlink bundle payload")
    observed = sha256(infer_path)
    if observed != INFER_K100_SHA256:
        raise RuntimeError(f"infer_k100.py identity drift: {observed}")
    spec = importlib.util.spec_from_file_location("phase11_locked_infer_k100", infer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to create locked infer_k100 import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, identity(infer_path)


def check_preimport_contract(bundle: Path, variant: str) -> tuple[dict, dict]:
    if os.environ.get("PATH") != DOCKER_PATH:
        raise RuntimeError(f"Docker-level PATH drift: {os.environ.get('PATH')!r}")
    if os.environ.get("PHASE11_K100_IMAGE_ID") != IMAGE_ID:
        raise RuntimeError("container image environment attestation drift")
    if platform.node() in ("", "machine2"):
        raise RuntimeError("benchmark must run inside a fresh Docker container with default hostname")
    if os.getpid() != 1:
        raise RuntimeError(f"benchmark Python must be container PID 1; got {os.getpid()}")

    manifest_candidate = bundle / "manifest.json"
    if manifest_candidate.is_symlink():
        raise RuntimeError("manifest.json must not be a symlink")
    manifest_path = manifest_candidate.resolve(strict=True)
    expected = VARIANTS[variant]["manifest_sha256"]
    observed = sha256(manifest_path)
    if observed != expected:
        raise RuntimeError(f"detached {variant} manifest SHA drift: {observed}; expected={expected}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != VARIANTS[variant]["bundle_id"]:
        raise RuntimeError(f"{variant} bundle_id drift")
    execution = manifest.get("execution", {})
    if execution.get("kind") != VARIANTS[variant]["execution_kind"]:
        raise RuntimeError(f"{variant} execution kind drift")
    source_id = VARIANTS[variant]["source_candidate_id"]
    if source_id is not None and execution.get("source_candidate_id") != source_id:
        raise RuntimeError("M5 source candidate identity drift")
    runtime = manifest.get("runtime_contract", {})
    required_runtime = {
        "official_image_id": IMAGE_ID,
        "onnxruntime": "1.19.2",
        "provider": MGX,
        "cpu_ep_fallback_disabled": True,
        "load_compiled_cache": True,
        "save_compiled_cache": False,
    }
    if any(runtime.get(key) != value for key, value in required_runtime.items()):
        raise RuntimeError(f"{variant} manifest runtime contract drift")
    shim_candidate = bundle / "tools" / "bin" / "lsmod"
    if shim_candidate.is_symlink():
        raise RuntimeError("locked lsmod must be executable, regular, and non-symlink")
    shim = shim_candidate.resolve(strict=True)
    shim_row = identity(shim)
    if not shim.is_file() or not os.access(shim, os.X_OK):
        raise RuntimeError("locked lsmod must be executable, regular, and non-symlink")
    if (shim_row["size_bytes"], shim_row["sha256"]) != (LSMOD_SIZE, LSMOD_SHA256):
        raise RuntimeError("locked static lsmod identity drift")
    return manifest, {"manifest": identity(manifest_path), "static_lsmod": shim_row}


def session_fallback_contract(runner: Any) -> dict[str, Any]:
    sessions = runner.sessions if runner.kind == "segment25" else [runner.session]
    rows = []
    for index, session in enumerate(sessions):
        providers = list(session.get_providers())
        options = session.get_session_options()
        try:
            disable_cpu = options.get_session_config_entry("session.disable_cpu_ep_fallback")
        except Exception as exc:  # fail closed if this runtime cannot expose the setting
            raise RuntimeError(f"cannot inspect CPU fallback setting for session {index}") from exc
        row = {
            "session_index": index,
            "providers": providers,
            "selected_provider": providers[0] if providers else None,
            "session_disable_cpu_ep_fallback": disable_cpu,
        }
        if not providers or providers[0] != MGX or disable_cpu != "1":
            raise RuntimeError(f"provider/fallback contract failed for session {index}: {row}")
        rows.append(row)
    return {
        "session_count": len(rows),
        "all_selected_provider_migraphx": True,
        "all_session_disable_cpu_ep_fallback_eq_1": True,
        "successful_execution_cannot_use_cpu_fallback_under_locked_session_config": True,
        "sessions": rows,
    }


def require_fixed_prediction(runner: Any, raw: Any, prediction: Any, phase: str, index: int) -> str:
    gate = runner.fixed_prediction_gate(raw, prediction)
    if not all(
        (
            gate.get("input_matches_bundled_fixed_sample") is True,
            gate.get("prediction_matches_expected") is True,
            isinstance(gate.get("observed_prediction_array_sha256"), str),
        )
    ):
        raise RuntimeError(f"fixed prediction gate failed during {phase} {index}: {gate}")
    return str(gate["observed_prediction_array_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--trial-index", type=int, required=True)
    parser.add_argument("--candidate-position", type=int, choices=(1, 2), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sequence = tuple(args.sequence.split(","))
    expected = expected_sequence(args.trial_index)
    if sequence != expected:
        raise RuntimeError(f"trial sequence drift: {sequence}; expected={expected}")
    if sequence[args.candidate_position - 1] != args.variant:
        raise RuntimeError("variant/candidate-position mismatch")
    if args.device != 0:
        raise RuntimeError("formal K100-2 paired benchmark is locked to device 0")

    bundle = args.bundle.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output == bundle or bundle in output.parents:
        raise RuntimeError("trial output must remain outside immutable bundle")

    process_started_utc = datetime.now(timezone.utc).isoformat()
    process_started_monotonic_ns = time.monotonic_ns()
    manifest, preimport = check_preimport_contract(bundle, args.variant)
    infer, infer_identity = import_infer(bundle)
    if infer.MGX != MGX or infer.EXPECTED_ORT != "1.19.2":
        raise RuntimeError("imported inference runtime constants drift")

    fixed_row = manifest.get("fixed_sample", {}).get("input", {})
    fixed_path = infer.resolve_payload(bundle, fixed_row.get("path"))
    infer.verify_identity(fixed_path, fixed_row)
    raw = infer.load_input(fixed_path)
    raw_sha = infer.array_sha256(raw)
    if raw_sha != manifest.get("fixed_sample", {}).get("input_array_sha256"):
        raise RuntimeError("fixed input array identity drift")

    runner = infer.K100BundleRunner(bundle, args.device, verify_payloads=True)
    if runner.manifest_identity["sha256"] != VARIANTS[args.variant]["manifest_sha256"]:
        raise RuntimeError("runner manifest identity drift")
    if not runner.payloads_fully_verified:
        raise RuntimeError("all bundle payload SHA256 values were not verified")
    if runner.runtime.get("image_identity_attested") is not True:
        raise RuntimeError("runner image identity was not attested")
    fallback = session_fallback_contract(runner)

    expected_prediction_sha = str(
        manifest.get("fixed_sample", {}).get("expected_prediction_array_sha256", "")
    )
    if len(expected_prediction_sha) != 64:
        raise RuntimeError("manifest expected prediction SHA is missing")

    warmup_prediction_hashes: set[str] = set()
    for index in range(1, WARMUP + 1):
        _logits, prediction, elapsed = runner.run(raw)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid warmup timing at index {index}")
        warmup_prediction_hashes.add(require_fixed_prediction(runner, raw, prediction, "warmup", index))

    measurements = []
    seconds = []
    measured_prediction_hashes: set[str] = set()
    for index in range(1, EXERCISES + 1):
        _logits, prediction, elapsed = runner.run(raw)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid measured timing at index {index}")
        observed_prediction_sha = require_fixed_prediction(
            runner, raw, prediction, "measurement", index
        )
        seconds.append(elapsed)
        measured_prediction_hashes.add(observed_prediction_sha)
        measurements.append(
            {
                "index": index,
                "input_to_synchronized_logits_seconds": elapsed,
                "prediction_array_sha256": observed_prediction_sha,
            }
        )

    if warmup_prediction_hashes != {expected_prediction_sha} or measured_prediction_hashes != {
        expected_prediction_sha
    }:
        raise RuntimeError("prediction hash instability detected")
    if os.environ.get("ORT_MIGRAPHX_LOAD_COMPILED_MODEL") != "1" or os.environ.get(
        "ORT_MIGRAPHX_SAVE_COMPILED_MODEL"
    ) != "0":
        raise RuntimeError("compiled-cache load/save environment drift")

    result = {
        "schema": "phase11_k100_deployment_cli_pair_trial_v1",
        "status": "passed",
        "protocol": {
            "node_label": "K100-2",
            "batch": 1,
            "warmup": WARMUP,
            "exercises": EXERCISES,
            "fresh_container_process_required": True,
            "trial_index": args.trial_index,
            "candidate_position": args.candidate_position,
            "trial_sequence": list(sequence),
            "timing_source": "third return value from locked K100BundleRunner.run",
            "timing_scope": (
                "steady float32[1,6,224,224] host input to synchronized float32 "
                "logits; includes H2D, ORT/MIGraphX dispatch, device synchronization, "
                "and logits D2H; excludes payload hashing, session/cache load, argmax, "
                "prediction hash, and receipt I/O"
            ),
            "same_public_runner_interface": "infer_k100.py::K100BundleRunner.run",
        },
        "variant": args.variant,
        "bundle": {
            "bundle_id": runner.manifest["bundle_id"],
            "manifest": runner.manifest_identity,
            "payload_file_count": runner.manifest["payload_file_count"],
            "payload_total_bytes": runner.manifest["payload_total_bytes"],
            "all_manifest_payload_sha256_verified": True,
            "execution_kind": runner.kind,
        },
        "identities": {
            "benchmark_script": identity(Path(__file__)),
            "infer_k100": infer_identity,
            "static_lsmod_preimport": preimport["static_lsmod"],
            "manifest_preimport": preimport["manifest"],
            "fixed_input_file": identity(fixed_path),
        },
        "runtime": {
            **runner.runtime,
            "container_hostname": platform.node(),
            "container_pid": os.getpid(),
            "process_started_utc": process_started_utc,
            "process_started_monotonic_ns": process_started_monotonic_ns,
            "docker_level_path_before_infer_import": DOCKER_PATH,
            "compiled_cache_load": os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"],
            "compiled_cache_save": os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"],
        },
        "provider_and_fallback": fallback,
        "fixed_sample": {
            "input_array_sha256": raw_sha,
            "expected_prediction_array_sha256": expected_prediction_sha,
            "warmup_prediction_hashes": sorted(warmup_prediction_hashes),
            "measured_prediction_hashes": sorted(measured_prediction_hashes),
            "all_130_predictions_fixed": True,
        },
        "load_seconds_not_timed": {
            "integrity_validation": runner.integrity_validation_seconds,
            "session_cache_load": runner.session_load_seconds,
            "total_runner_load": runner.total_load_seconds,
        },
        "measurements": measurements,
        "summary": timing_summary(seconds),
        "gates": {
            "exact_image_attested": True,
            "docker_level_path_locked_before_infer_import": True,
            "static_lsmod_locked": True,
            "detached_manifest_locked": True,
            "all_bundle_payload_sha256_verified": True,
            "batch1_fixed_input": True,
            "warmup_eq_30": True,
            "measurements_eq_100": True,
            "migraphx_selected_for_all_sessions": True,
            "cpu_ep_fallback_disabled_for_all_sessions": True,
            "all_outputs_finite": True,
            "fixed_prediction_all_runs": True,
        },
        "claim_boundaries": {
            "this_trial_is_not_a_kernel_precision_proof": True,
            "this_trial_does_not_override_strict_logits_failure": True,
            "latency_upgrade_decision_requires_three_fresh_trials_and_cv_gate": True,
        },
    }
    if not all(result["gates"].values()):
        raise RuntimeError("one or more formal trial gates failed")
    atomic_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
