#!/usr/bin/env python3
'Research implementation: diagnose blocks.'

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


EXPECTED_ORT_VERSION = "1.19.2"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
BLOCK_LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24))
FIELDS = (
    "sample_id",
    "block_index",
    "block_label",
    "L_mae",
    "L_max_abs",
    "L_relative_l2",
    "C_mae",
    "C_max_abs",
    "C_relative_l2",
    "P_mae",
    "P_max_abs",
    "P_relative_l2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if expected is not None and (
        item["size_bytes"] != int(expected["size_bytes"])
        or item["sha256"] != str(expected["sha256"])
    ):
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def comparison(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise RuntimeError(f"shape mismatch: {reference64.shape} != {candidate64.shape}")
    if not np.isfinite(reference64).all() or not np.isfinite(candidate64).all():
        raise RuntimeError("non-finite tensor in block comparison")
    difference = candidate64 - reference64
    absolute = np.abs(difference)
    denominator = max(float(np.linalg.norm(reference64.ravel())), np.finfo(np.float64).tiny)
    return (
        float(absolute.mean()),
        float(absolute.max()),
        float(np.linalg.norm(difference.ravel()) / denominator),
    )


def options(ort, *, strict: bool, profile_prefix: Path | None = None):
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    value.intra_op_num_threads = 4
    value.inter_op_num_threads = 1
    if strict:
        value.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        value.enable_profiling = True
        value.profile_file_prefix = str(profile_prefix)
    return value


def parse_profile(path: Path) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    result = {
        "provider_event_counts": dict(sorted(counts.items())),
        "migraphx_events_positive": int(counts.get(MGX, 0)) > 0,
        "cpu_events_zero": int(counts.get(CPU, 0)) == 0,
    }
    result["passed"] = result["migraphx_events_positive"] and result["cpu_events_zero"]
    return result


def set_cache(cache: Path | None) -> None:
    if cache is None:
        os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
        os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
        os.environ.pop("ORT_MIGRAPHX_LOAD_COMPILE_PATH", None)
        os.environ.pop("ORT_MIGRAPHX_SAVE_COMPILE_PATH", None)
        return
    cache = cache.resolve(strict=True)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache)


def find_cache(cache_dir: Path, block_index: int) -> Path:
    candidates = []
    for path in cache_dir.rglob("*.mxr"):
        normalized = str(path).replace("-", "_").lower()
        markers = (f"segment_{block_index:02d}", f"block_{block_index:02d}")
        if any(marker in normalized for marker in markers):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"block {block_index}: expected exactly one cache under {cache_dir}, found {candidates}"
        )
    path = candidates[0].resolve(strict=True)
    result_path = (path.parent / "output" / "result.json").resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = result.get("compiled_cache")
    if not isinstance(expected, dict) or "size_bytes" not in expected or "sha256" not in expected:
        raise RuntimeError(f"block {block_index}: compiled-cache receipt is incomplete")
    identity(path, expected)
    return path


def load_inputs(pack_path: Path, manifest: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    identity(pack_path, manifest["output_pack"])
    with np.load(pack_path, allow_pickle=False) as pack:
        if set(pack.files) != {"inputs", "targets", "sample_ids"}:
            raise RuntimeError(f"unexpected input-pack keys: {pack.files}")
        inputs = np.ascontiguousarray(pack["inputs"], dtype=np.float32)
        sample_ids = pack["sample_ids"].tolist()
    selection = manifest["configuration_validation"]
    if inputs.shape != (64, 6, 224, 224) or len(sample_ids) != 64:
        raise RuntimeError("input pack is not float32[64,6,224,224]")
    if sample_ids != selection["sample_ids"]:
        raise RuntimeError("input pack sample IDs/order drift")
    if hashlib.sha256(inputs.tobytes()).hexdigest() != selection["inputs_sha256"]:
        raise RuntimeError("input tensor SHA-256 drift")
    if hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest() != selection[
        "sample_ids_sha256"
    ]:
        raise RuntimeError("input sample-ID digest drift")
    return inputs, sample_ids


def validate_input_role(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "journal_stage1_input_pack_v1" or manifest.get("status") != "passed":
        raise RuntimeError("input manifest is not a passed Stage-1 pack")
    if manifest.get("selection_role") != "configuration-validation only; precision-map selection permitted":
        raise RuntimeError("input pack is not configuration-validation only")
    gates = manifest.get("leakage_gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise RuntimeError("not every input-pack leakage gate is true")
    if manifest.get("configuration_validation", {}).get("sample_count") != 64:
        raise RuntimeError("configuration-validation count is not 64")
    if manifest.get("frozen_test_replay", {}).get("expected_samples") != 90:
        raise RuntimeError("frozen-test identity audit is missing")


def load_models(report_path: Path, models_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("segments", [])
    if report.get("status") != "created_static_pass" or len(rows) != 25:
        raise RuntimeError("INT8 backbone segment report status/count drift")
    if report.get("variant") != "int8_backbone_compat_25_static_batch1_onnx_segments":
        raise RuntimeError(f"unexpected segment variant: {report.get('variant')!r}")
    if [row.get("label") for row in rows[:24]] != list(BLOCK_LABELS):
        raise RuntimeError("encoder-block order drift")
    paths = []
    for row in rows[:24]:
        path = models_dir / Path(row["path"]).name
        identity(path, row)
        paths.append(path.resolve())
    return paths, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pack", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--segment-report", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "run_report.json"
    result: dict[str, Any] = {
        "schema": "journal_stage1_block_diagnostics_run_v1",
        "status": "failed",
        "selection_split": "configuration-validation only",
        "formal_test_used_for_selection": False,
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != EXPECTED_ORT_VERSION:
            raise RuntimeError(f"ORT version drift: {ort.__version__} != {EXPECTED_ORT_VERSION}")
        available = ort.get_available_providers()
        if MGX not in available or CPU not in available:
            raise RuntimeError(f"required providers unavailable: {available}")

        input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        validate_input_role(input_manifest)
        inputs, sample_ids = load_inputs(args.input_pack, input_manifest)
        model_paths, segment_report = load_models(args.segment_report, args.models_dir.resolve(strict=True))
        cache_paths = (
            [find_cache(args.cache_dir.resolve(strict=True), index) for index in range(24)]
            if args.cache_dir is not None
            else [None] * 24
        )

        cpu_current = [np.ascontiguousarray(inputs[index : index + 1]) for index in range(64)]
        mgx_current = [value.copy() for value in cpu_current]
        rows: list[dict[str, Any]] = []
        profiles: list[dict[str, Any]] = []
        block_seconds: list[dict[str, Any]] = []

        for block_index, model_path in enumerate(model_paths):
            cpu_session = ort.InferenceSession(
                str(model_path), sess_options=options(ort, strict=False), providers=[CPU]
            )
            set_cache(cache_paths[block_index])
            profile_prefix = args.output_dir / f"block_{block_index:02d}_profile"
            mgx_session = ort.InferenceSession(
                str(model_path),
                sess_options=options(ort, strict=True, profile_prefix=profile_prefix),
                providers=[(MGX, {"device_id": 0})],
            )
            mgx_session.disable_fallback()
            if not mgx_session.get_providers() or mgx_session.get_providers()[0] != MGX:
                raise RuntimeError(f"block {block_index}: provider priority drift")
            if len(cpu_session.get_inputs()) != 1 or len(cpu_session.get_outputs()) != 1:
                raise RuntimeError(f"block {block_index}: expected one input and one output")
            input_name = cpu_session.get_inputs()[0].name
            output_name = cpu_session.get_outputs()[0].name
            if input_name != mgx_session.get_inputs()[0].name or output_name != mgx_session.get_outputs()[0].name:
                raise RuntimeError(f"block {block_index}: CPU/MIGraphX I/O contract drift")

            next_cpu: list[np.ndarray] = []
            next_mgx: list[np.ndarray] = []
            started = perf_counter()
            for scene_index, sample_id in enumerate(sample_ids):
                cpu_input = cpu_current[scene_index]
                mgx_input = mgx_current[scene_index]
                cpu_output = np.ascontiguousarray(
                    cpu_session.run([output_name], {input_name: cpu_input})[0]
                )
                mgx_local = np.ascontiguousarray(
                    mgx_session.run([output_name], {input_name: cpu_input})[0]
                )
                if block_index == 0:
                    mgx_cumulative = mgx_local.copy()
                    propagated_cpu = cpu_output.copy()
                else:
                    mgx_cumulative = np.ascontiguousarray(
                        mgx_session.run([output_name], {input_name: mgx_input})[0]
                    )
                    propagated_cpu = np.ascontiguousarray(
                        cpu_session.run([output_name], {input_name: mgx_input})[0]
                    )
                local = comparison(cpu_output, mgx_local)
                cumulative = comparison(cpu_output, mgx_cumulative)
                upstream = comparison(cpu_output, propagated_cpu)
                rows.append(
                    dict(
                        zip(
                            FIELDS,
                            (
                                sample_id,
                                block_index,
                                BLOCK_LABELS[block_index],
                                *local,
                                *cumulative,
                                *upstream,
                            ),
                        )
                    )
                )
                next_cpu.append(cpu_output)
                next_mgx.append(mgx_cumulative)
            elapsed = perf_counter() - started

            profile_path = Path(mgx_session.end_profiling()).resolve(strict=True)
            placement = parse_profile(profile_path)
            if not placement["passed"]:
                raise RuntimeError(f"block {block_index}: strict placement failed: {placement}")
            profiles.append(
                {
                    "block_index": block_index,
                    "block_label": BLOCK_LABELS[block_index],
                    **identity(profile_path),
                    **placement,
                }
            )
            block_seconds.append({"block_index": block_index, "diagnostic_seconds": elapsed})
            cpu_current, mgx_current = next_cpu, next_mgx
            del cpu_session, mgx_session
            gc.collect()
            print(
                f"block {block_index:02d}/23 complete: 64 scenes, {elapsed:.3f}s, "
                f"MIGraphX events={placement['provider_event_counts'].get(MGX, 0)}",
                flush=True,
            )

        expected_pairs = {(sample_id, index) for sample_id in sample_ids for index in range(24)}
        observed_pairs = {(row["sample_id"], row["block_index"]) for row in rows}
        if len(rows) != 1536 or observed_pairs != expected_pairs:
            raise RuntimeError("per-scene/per-block result grid is incomplete or duplicated")

        csv_path = args.output_dir / "per_scene_per_block_diagnostics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        result.update(
            {
                "status": "passed",
                "definitions": {
                    "L": "CPU and strict MIGraphX block outputs on the identical CPU-reference input",
                    "C": "CPU-reference chain output versus cumulative strict-MIGraphX chain output",
                    "P": "CPU block output on CPU-reference upstream input versus CPU block output on cumulative-MIGraphX upstream input",
                },
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "onnxruntime": ort.__version__,
                    "available_providers": available,
                },
                "input_pack": identity(args.input_pack, input_manifest["output_pack"]),
                "input_manifest": identity(args.input_manifest),
                "segment_report": identity(args.segment_report),
                "segment_variant": segment_report["variant"],
                "models": [identity(path) for path in model_paths],
                "caches": [identity(path) if path is not None else None for path in cache_paths],
                "sample_count": 64,
                "block_count": 24,
                "row_count": 1536,
                "sample_ids_sha256": input_manifest["configuration_validation"]["sample_ids_sha256"],
                "profiles": profiles,
                "block_seconds": block_seconds,
                "gates": {
                    "configuration_validation_only": True,
                    "exact_64_by_24_grid": True,
                    "all_values_finite": True,
                    "all_profiles_migraphx_positive_cpu_zero": all(item["passed"] for item in profiles),
                    "formal_test_not_used_for_selection": True,
                },
                "per_scene_per_block_diagnostics": identity(csv_path),
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
