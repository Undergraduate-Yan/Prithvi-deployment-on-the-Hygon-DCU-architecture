#!/usr/bin/env python3
"""Validate and aggregate six fresh-container K100 deployment CLI trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_ID = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
M5_MANIFEST_SHA256 = "f4dc433595c43332d4e29c2351363b3245f214273f058551788d50be29c388da"
FP16_MANIFEST_SHA256 = "1f433a917aed5fead6dd95597a5a03f500a93ce564067e9dad4d25f4682bad25"
INFER_K100_SHA256 = "fc338d90f7592f6a75cfa28191702598777b21914415e7178cef932de60571b7"
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"
DOCKER_PATH = "/bundle/tools/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
WARMUP = 30
EXERCISES = 100
MEMORY_BYTES = 54 * 1024**3
SHM_BYTES = 16 * 1024**3
PIDS_LIMIT = 2048
CV_LIMIT_PERCENT = 5.0
M5_OVER_FP16_LIMIT = 1.05
INPUT_SHAPE = [1, 6, 224, 224]
INPUT_DTYPE = "float32"

VARIANTS = {
    "m5": {
        "manifest_sha256": M5_MANIFEST_SHA256,
        "bundle_id": "prithvi-k100-m5-v1",
        "execution_kind": "segment25",
        "session_count": 25,
    },
    "fp16": {
        "manifest_sha256": FP16_MANIFEST_SHA256,
        "bundle_id": "prithvi-k100-fp16-full-v1",
        "execution_kind": "fp16_full",
        "session_count": 1,
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
    if not seconds or any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise RuntimeError("timing vector contains invalid values")
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


def near(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def validate_summary(observed: dict, expected: dict, context: str) -> None:
    if set(observed) != set(expected):
        raise RuntimeError(f"{context} timing summary keys drift")
    for key, value in expected.items():
        if key == "count":
            if int(observed[key]) != int(value):
                raise RuntimeError(f"{context} count drift")
        elif not near(observed[key], value):
            raise RuntimeError(f"{context} {key} drift: {observed[key]} != {value}")


def env_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or key in result:
            raise RuntimeError("container environment contains malformed or duplicate entries")
        result[key] = value
    return result


def bind_map(values: list[str]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for value in values:
        parts = value.rsplit(":", 2)
        if len(parts) != 3:
            raise RuntimeError(f"malformed Docker bind: {value}")
        source, target, mode = parts
        if target in result:
            raise RuntimeError(f"duplicate Docker bind target: {target}")
        result[target] = (source, mode)
    return result


def validate_container_inspect(
    path: Path,
    container_id_path: Path,
    variant: str,
    trial_index: int,
    position: int,
    sequence: tuple[str, str],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"Docker inspect must contain exactly one object: {path}")
    row = payload[0]
    container_id = container_id_path.read_text(encoding="utf-8").strip()
    if len(container_id) != 64 or row.get("Id") != container_id:
        raise RuntimeError("container ID evidence drift")
    if row.get("Image") != IMAGE_ID or row.get("Config", {}).get("Image") != IMAGE_ID:
        raise RuntimeError("container image digest drift")
    state = row.get("State", {})
    if state.get("Status") != "exited" or int(state.get("ExitCode", -1)) != 0:
        raise RuntimeError("benchmark container did not exit successfully")
    if row.get("Config", {}).get("Hostname") != container_id[:12]:
        raise RuntimeError("default Docker hostname/container-ID relation drift")
    if row.get("Config", {}).get("Entrypoint") != ["/usr/bin/python3"]:
        raise RuntimeError("container entrypoint drift")
    expected_cmd = [
        "/tools/benchmark_bundle_cli_trial.py",
        "--bundle",
        "/bundle",
        "--variant",
        variant,
        "--trial-index",
        str(trial_index),
        "--candidate-position",
        str(position),
        "--sequence",
        ",".join(sequence),
        "--device",
        "0",
        "--output",
        "/run/result.json",
    ]
    if row.get("Config", {}).get("Cmd") != expected_cmd:
        raise RuntimeError("container benchmark command drift")
    environment = env_map(row.get("Config", {}).get("Env", []))
    if environment.get("PATH") != DOCKER_PATH or environment.get(
        "PHASE11_K100_IMAGE_ID"
    ) != IMAGE_ID:
        raise RuntimeError("Docker-level PATH/image environment drift")

    host = row.get("HostConfig", {})
    if int(host.get("Memory", -1)) != MEMORY_BYTES:
        raise RuntimeError("container memory limit drift")
    if int(host.get("PidsLimit", -1)) != PIDS_LIMIT:
        raise RuntimeError("container PID limit drift")
    if host.get("IpcMode") != "host" or int(host.get("ShmSize", -1)) != SHM_BYTES:
        raise RuntimeError("container IPC/shm contract drift")
    if "video" not in host.get("GroupAdd", []):
        raise RuntimeError("container video group contract drift")
    devices = {
        (item.get("PathOnHost"), item.get("PathInContainer"), item.get("CgroupPermissions"))
        for item in host.get("Devices", [])
    }
    if devices != {("/dev/kfd", "/dev/kfd", "rwm"), ("/dev/dri", "/dev/dri", "rwm")}:
        raise RuntimeError(f"container device contract drift: {devices}")
    binds = bind_map(host.get("Binds", []))
    if set(binds) != {"/bundle", "/tools", "/run", "/opt/hyhal"}:
        raise RuntimeError(f"container bind target set drift: {set(binds)}")
    if binds["/bundle"][1] != "ro" or binds["/tools"][1] != "ro" or binds[
        "/opt/hyhal"
    ] != ("/opt/hyhal", "ro") or binds["/run"][1] != "rw":
        raise RuntimeError("container bind mode/source contract drift")
    return {
        "container_id": container_id,
        "container_name": row.get("Name"),
        "container_hostname": row["Config"]["Hostname"],
        "image_id": row["Image"],
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "docker_path": environment["PATH"],
        "bundle_host_path": binds["/bundle"][0],
        "tools_host_path": binds["/tools"][0],
        "run_host_path": binds["/run"][0],
    }


def validate_trial(
    directory: Path,
    variant: str,
    trial_index: int,
    position: int,
    expected_trial_script_sha256: str,
) -> tuple[dict[str, Any], list[float]]:
    sequence = expected_sequence(trial_index)
    result_path = directory / "result.json"
    inspect_path = directory / "container_inspect_exited.json"
    id_path = directory / "container_id.txt"
    exit_path = directory / "container.exit"
    for path in (result_path, inspect_path, id_path, exit_path, directory / "container.log"):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or non-regular trial evidence: {path}")
    if exit_path.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError(f"nonzero container exit evidence: {directory}")
    inspect = validate_container_inspect(
        inspect_path, id_path, variant, trial_index, position, sequence
    )
    if Path(inspect["run_host_path"]).resolve(strict=True) != directory.resolve(strict=True):
        raise RuntimeError("Docker /run bind does not resolve to this trial evidence directory")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "phase11_k100_deployment_cli_pair_trial_v1" or result.get(
        "status"
    ) != "passed":
        raise RuntimeError("trial schema/status drift")
    protocol = result.get("protocol", {})
    required_protocol = {
        "node_label": "K100-2",
        "batch": 1,
        "warmup": WARMUP,
        "exercises": EXERCISES,
        "fresh_container_process_required": True,
        "trial_index": trial_index,
        "candidate_position": position,
        "trial_sequence": list(sequence),
        "same_public_runner_interface": "infer_k100.py::K100BundleRunner.run",
    }
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise RuntimeError("trial protocol drift")
    if result.get("variant") != variant:
        raise RuntimeError("trial variant drift")
    spec = VARIANTS[variant]
    bundle = result.get("bundle", {})
    if (
        bundle.get("bundle_id") != spec["bundle_id"]
        or bundle.get("manifest", {}).get("sha256") != spec["manifest_sha256"]
        or bundle.get("execution_kind") != spec["execution_kind"]
        or bundle.get("all_manifest_payload_sha256_verified") is not True
    ):
        raise RuntimeError("trial bundle identity/full-hash contract drift")
    identities = result.get("identities", {})
    if identities.get("benchmark_script", {}).get("sha256") != expected_trial_script_sha256:
        raise RuntimeError("trial benchmark-script identity drift")
    if identities.get("infer_k100", {}).get("sha256") != INFER_K100_SHA256:
        raise RuntimeError("trial infer_k100 identity drift")
    static_lsmod = identities.get("static_lsmod_preimport", {})
    if (int(static_lsmod.get("size_bytes", -1)), static_lsmod.get("sha256")) != (
        819_664,
        LSMOD_SHA256,
    ):
        raise RuntimeError("trial static lsmod identity drift")
    runtime = result.get("runtime", {})
    if (
        runtime.get("container_hostname") != inspect["container_hostname"]
        or int(runtime.get("container_pid", -1)) != 1
        or runtime.get("attested_image_id") != IMAGE_ID
        or runtime.get("image_identity_attested") is not True
        or runtime.get("docker_level_path_before_infer_import") != DOCKER_PATH
        or runtime.get("compiled_cache_load") != "1"
        or runtime.get("compiled_cache_save") != "0"
    ):
        raise RuntimeError("trial runtime attestation drift")
    fallback = result.get("provider_and_fallback", {})
    if (
        int(fallback.get("session_count", -1)) != spec["session_count"]
        or fallback.get("all_selected_provider_migraphx") is not True
        or fallback.get("all_session_disable_cpu_ep_fallback_eq_1") is not True
        or fallback.get(
            "successful_execution_cannot_use_cpu_fallback_under_locked_session_config"
        )
        is not True
    ):
        raise RuntimeError("trial provider/fallback gate drift")
    sessions = fallback.get("sessions", [])
    if len(sessions) != spec["session_count"] or any(
        row.get("selected_provider") != "MIGraphXExecutionProvider"
        or row.get("session_disable_cpu_ep_fallback") != "1"
        for row in sessions
    ):
        raise RuntimeError("trial session-level provider/fallback evidence drift")
    fixed = result.get("fixed_sample", {})
    expected_prediction = fixed.get("expected_prediction_array_sha256")
    if (
        len(str(fixed.get("input_array_sha256", ""))) != 64
        or len(str(expected_prediction or "")) != 64
        or fixed.get("warmup_prediction_hashes") != [expected_prediction]
        or fixed.get("measured_prediction_hashes") != [expected_prediction]
        or fixed.get("all_130_predictions_fixed") is not True
    ):
        raise RuntimeError("trial fixed input/prediction contract drift")
    gates = result.get("gates", {})
    if not gates or any(value is not True for value in gates.values()):
        raise RuntimeError("trial contains a failed gate")
    measurements = result.get("measurements", [])
    if len(measurements) != EXERCISES or [row.get("index") for row in measurements] != list(
        range(1, EXERCISES + 1)
    ):
        raise RuntimeError("trial measurement count/order drift")
    seconds = [float(row.get("input_to_synchronized_logits_seconds", math.nan)) for row in measurements]
    if any(
        row.get("prediction_array_sha256") != expected_prediction for row in measurements
    ):
        raise RuntimeError("measured prediction SHA drift")
    recomputed = timing_summary(seconds)
    validate_summary(result.get("summary", {}), recomputed, f"trial {trial_index} {variant}")
    return {
        "trial_index": trial_index,
        "candidate_position": position,
        "sequence": list(sequence),
        "result": identity(result_path),
        "container_inspect": identity(inspect_path),
        "container_log": identity(directory / "container.log"),
        "container": inspect,
        "summary": recomputed,
        "fixed_input_array_sha256": fixed["input_array_sha256"],
        "expected_prediction_array_sha256": expected_prediction,
    }, seconds


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve(strict=False)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite aggregate output: {path}")
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


def validate_cross_bundle_fixed_contract(
    records: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, str]]:
    """Require one raw input while preserving bundle-specific output oracles.

    The locked trial tool and locked infer_k100 implementation enforce exact
    float32[1,6,224,224] before every run. The SHA is the value-derived raw
    array SHA, so equality across all six trials proves identical input values.
    Expected predictions are intentionally bundle-specific: each must remain
    stable across its own three trials, but M5 and FP16 need not share one.
    """
    input_hashes = {
        row["fixed_input_array_sha256"] for rows in records.values() for row in rows
    }
    if len(input_hashes) != 1:
        raise RuntimeError("M5 and FP16 did not use the same raw input array SHA256")
    raw_input_sha256 = next(iter(input_hashes))
    if len(raw_input_sha256) != 64:
        raise RuntimeError("shared raw input array SHA256 is malformed")
    int(raw_input_sha256, 16)

    expected_predictions: dict[str, str] = {}
    for variant, rows in records.items():
        hashes = {row["expected_prediction_array_sha256"] for row in rows}
        if len(hashes) != 1:
            raise RuntimeError(
                f"{variant} expected prediction identity changed across its three trials"
            )
        expected = next(iter(hashes))
        if len(expected) != 64:
            raise RuntimeError(f"{variant} expected prediction SHA256 is malformed")
        int(expected, 16)
        expected_predictions[variant] = expected
    if set(expected_predictions) != set(VARIANTS):
        raise RuntimeError("bundle-specific expected prediction contracts are incomplete")
    return raw_input_sha256, expected_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-trial-script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_trial_script_sha256) != 64:
        raise RuntimeError("expected trial-script SHA256 must contain 64 hex characters")
    int(args.expected_trial_script_sha256, 16)
    root = args.run_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if root not in output.parents:
        raise RuntimeError("aggregate summary must be written under the new run root")

    records: dict[str, list[dict[str, Any]]] = {"m5": [], "fp16": []}
    all_seconds: dict[str, list[float]] = {"m5": [], "fp16": []}
    container_ids: set[str] = set()
    position_counts = {variant: {1: 0, 2: 0} for variant in VARIANTS}
    for trial_index in (1, 2, 3):
        sequence = expected_sequence(trial_index)
        for position, variant in enumerate(sequence, start=1):
            directory = root / f"trial_{trial_index:02d}" / f"position_{position:02d}_{variant}"
            record, seconds = validate_trial(
                directory,
                variant,
                trial_index,
                position,
                args.expected_trial_script_sha256,
            )
            container_id = record["container"]["container_id"]
            if container_id in container_ids:
                raise RuntimeError("Docker container ID reused across fresh-process trials")
            container_ids.add(container_id)
            records[variant].append(record)
            all_seconds[variant].extend(seconds)
            position_counts[variant][position] += 1

    if len(container_ids) != 6 or any(len(rows) != 3 for rows in records.values()):
        raise RuntimeError("formal paired benchmark requires six unique containers")
    for variant, rows in records.items():
        if len({row["container"]["bundle_host_path"] for row in rows}) != 1:
            raise RuntimeError(f"{variant} bundle host path changed across fresh trials")
    if len(
        {
            row["container"]["tools_host_path"]
            for rows in records.values()
            for row in rows
        }
    ) != 1:
        raise RuntimeError("benchmark tool host path changed across fresh trials")
    if any(abs(counts[1] - counts[2]) > 1 for counts in position_counts.values()):
        raise RuntimeError("three-trial candidate positions are not maximally balanced")

    raw_input_sha256, expected_predictions = validate_cross_bundle_fixed_contract(records)

    variants = {}
    for variant, rows in records.items():
        trial_medians = [row["summary"]["median_ms"] for row in rows]
        cv_percent = statistics.stdev(trial_medians) / statistics.fmean(trial_medians) * 100.0
        variants[variant] = {
            "bundle_id": VARIANTS[variant]["bundle_id"],
            "manifest_sha256": VARIANTS[variant]["manifest_sha256"],
            "trial_count": 3,
            "fresh_container_count": 3,
            "warmup_per_trial": WARMUP,
            "measurements_per_trial": EXERCISES,
            "aggregate": timing_summary(all_seconds[variant]),
            "trial_median_ms": trial_medians,
            "trial_median_cv_percent": cv_percent,
            "trial_median_cv_le_5_percent": cv_percent <= CV_LIMIT_PERCENT,
            "candidate_position_counts": {
                "position_1": position_counts[variant][1],
                "position_2": position_counts[variant][2],
            },
            "trials": rows,
        }

    m5_median = variants["m5"]["aggregate"]["median_ms"]
    fp16_median = variants["fp16"]["aggregate"]["median_ms"]
    median_ratio = m5_median / fp16_median
    paired_trial_ratios = [
        records["m5"][index]["summary"]["median_ms"]
        / records["fp16"][index]["summary"]["median_ms"]
        for index in range(3)
    ]
    stable = all(row["trial_median_cv_le_5_percent"] for row in variants.values())
    latency_gate = stable and median_ratio <= M5_OVER_FP16_LIMIT
    status = (
        "completed_formal_latency_gate_passed"
        if latency_gate
        else "completed_formal_latency_gate_failed"
        if stable
        else "completed_diagnostic_only_cv_gate_failed"
    )
    result = {
        "schema": "phase11_k100_deployment_cli_pair_summary_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "node_label": "K100-2",
            "image_id": IMAGE_ID,
            "same_locked_interface": "infer_k100.py::K100BundleRunner.run",
            "infer_k100_sha256": INFER_K100_SHA256,
            "fixed_batch": 1,
            "warmup_per_candidate_per_trial": WARMUP,
            "measurements_per_candidate_per_trial": EXERCISES,
            "trials_per_candidate": 3,
            "fresh_container_processes_total": 6,
            "candidate_order": [
                {"trial_index": index, "sequence": list(expected_sequence(index))}
                for index in (1, 2, 3)
            ],
            "position_balance_rule": "per-candidate position-count difference <= 1",
            "timing_scope": (
                "steady float32 input to synchronized float32 logits through the public bundle "
                "runner; excludes integrity validation, cache/session load, argmax/hash and I/O"
            ),
            "trial_median_cv_limit_percent": CV_LIMIT_PERCENT,
            "m5_over_fp16_median_latency_limit": M5_OVER_FP16_LIMIT,
            "aggregation_is_read_only_over_six_existing_trial_directories": True,
            "gpu_inference_is_not_executed_by_this_aggregator": True,
        },
        "identities": {
            "aggregator": identity(Path(__file__)),
            "expected_trial_script_sha256": args.expected_trial_script_sha256,
            "m5_manifest_sha256": M5_MANIFEST_SHA256,
            "fp16_manifest_sha256": FP16_MANIFEST_SHA256,
            "static_lsmod_sha256": LSMOD_SHA256,
        },
        "fixed_sample": {
            "raw_input": {
                "array_sha256": raw_input_sha256,
                "shape": INPUT_SHAPE,
                "dtype": INPUT_DTYPE,
                "same_values_shape_and_dtype_for_both_bundles_and_all_six_trials": True,
                "shape_dtype_enforced_by_locked_trial_and_infer_k100_tools": True,
            },
            "bundle_specific_expected_prediction_array_sha256": expected_predictions,
            "cross_bundle_expected_prediction_identity_required_equal": False,
            "each_bundle_expected_prediction_stable_across_its_three_trials": True,
            "all_780_predictions_matched_their_own_bundle_oracle": True,
        },
        "variants": variants,
        "comparison": {
            "m5_median_ms": m5_median,
            "fp16_median_ms": fp16_median,
            "m5_over_fp16_median_latency_ratio": median_ratio,
            "fp16_over_m5_speed_ratio": fp16_median / m5_median,
            "m5_latency_delta_vs_fp16_percent": (median_ratio - 1.0) * 100.0,
            "paired_trial_m5_over_fp16_median_ratios": paired_trial_ratios,
            "m5_not_slower_than_fp16_by_more_than_5_percent": latency_gate,
        },
        "gates": {
            "exact_image_all_six_containers": True,
            "six_unique_fresh_containers": True,
            "docker_level_path_locked": True,
            "bundles_read_only_and_receipts_separate": True,
            "all_bundle_payload_sha256_verified_each_trial": True,
            "same_raw_input_array_sha_shape_dtype": True,
            "each_bundle_prediction_oracle_stable_across_three_trials": True,
            "all_predictions_match_own_bundle_expected_sha": True,
            "migraphx_selected_all_sessions": True,
            "cpu_ep_fallback_disabled_all_sessions": True,
            "three_trials_each": True,
            "thirty_warmups_each_trial": True,
            "one_hundred_measurements_each_trial": True,
            "candidate_order_maximally_balanced": True,
            "m5_trial_median_cv_le_5_percent": variants["m5"][
                "trial_median_cv_le_5_percent"
            ],
            "fp16_trial_median_cv_le_5_percent": variants["fp16"][
                "trial_median_cv_le_5_percent"
            ],
            "m5_over_fp16_median_latency_le_1_05": median_ratio <= M5_OVER_FP16_LIMIT,
        },
        "claims": {
            "formal_same_cli_latency_comparison_allowed": stable,
            "m5_latency_upgrade_gate_passed": latency_gate,
            "deployment_recommendation_may_change_from_this_latency_gate_alone": False,
            "same_raw_input_values_shape_dtype_confirmed": True,
            "bundle_specific_prediction_oracles_preserved": True,
            "existing_six_passed_trials_reused_without_gpu_rerun": True,
            "native_int8_kernel_precision_proven_here": False,
            "strict_logits_equivalence_proven_here": False,
            "historical_strict_failure_overridden": False,
        },
        "decision": (
            "M5 passes the deployment-plan latency condition (<=5% slower than FP16 full)."
            if latency_gate
            else "M5 fails the deployment-plan latency condition (>5% slower than FP16 full)."
            if stable
            else "No formal M5/FP16 latency decision: at least one trial-median CV exceeds 5%."
        ),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
