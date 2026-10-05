#!/usr/bin/env python3
'Trace a selected compiled flood region.'

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify(record: dict[str, Any]) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    actual = identity(path)
    if (actual["size_bytes"], actual["sha256"]) != (int(record["size_bytes"]), str(record["sha256"])):
        raise RuntimeError(f"identity drift: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prepass-result", required=True, type=Path)
    parser.add_argument("--session-ordinal", required=True, type=int)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--expected-start-block", required=True, type=int)
    parser.add_argument("--expected-end-block", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.session_ordinal < 0 or not 1 <= args.repetitions <= 1000:
        raise RuntimeError("trace selection drift")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema": "journal_phase2d_rcs21_direct_region_trace_target_v1",
        "status": "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "hardware": "海光 K100 AI 加速卡",
        "formal_90_image_test_used": False,
    }
    try:
        manifest_path = args.manifest.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prepass_path = args.prepass_result.resolve(strict=True)
        prepass = json.loads(prepass_path.read_text(encoding="utf-8"))
        if not str(manifest.get("schema", "")).startswith("journal_phase2d_rcs"):
            raise RuntimeError("admitted RCS candidate manifest required")
        if prepass.get("status") != "passed" or prepass.get("formal_90_image_test_used") is not False:
            raise RuntimeError("prepass is not admitted")
        if prepass["manifest"]["sha256"] != identity(manifest_path)["sha256"]:
            raise RuntimeError("prepass manifest lineage drift")
        row = manifest["candidate_sessions"][args.session_ordinal]
        if (
            int(row["start_block"]) != args.expected_start_block
            or int(row["end_block"]) != args.expected_end_block
        ):
            raise RuntimeError("selected interval disagrees with explicit block contract")
        boundary_row = next(
            item for item in prepass["boundaries"]
            if int(item["session_ordinal"]) == args.session_ordinal
        )
        model = verify(row["model"])
        cache = verify(row["cache"])
        boundary_path = verify(boundary_row["boundary"])
        if boundary_row["model"]["sha256"] != identity(model)["sha256"] or boundary_row["cache"]["sha256"] != identity(cache)["sha256"]:
            raise RuntimeError("boundary target identity drift")
        if boundary_row.get("boundary_format") == "named_npz" or boundary_path.suffix == ".npz":
            with np.load(boundary_path, allow_pickle=False) as packed:
                boundaries = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
        else:
            boundaries = {
                str(boundary_row["input_name"]): np.ascontiguousarray(
                    np.load(boundary_path, allow_pickle=False)
                )
            }
        expected_hashes = boundary_row.get("boundary_tensor_sha256_by_name") or {
            str(boundary_row["input_name"]): boundary_row["boundary_tensor_sha256"]
        }
        if set(boundaries) != set(expected_hashes):
            raise RuntimeError("boundary input-name set drift")
        for name, boundary in boundaries.items():
            expected_shape = (1, 6, 224, 224) if name == "image" else (1, 197, 1024)
            if boundary.dtype != np.float32 or boundary.shape != expected_shape or not np.isfinite(boundary).all():
                raise RuntimeError(f"boundary contract drift: {name}")
            if hashlib.sha256(boundary.tobytes()).hexdigest() != expected_hashes[name]:
                raise RuntimeError(f"boundary tensor hash drift: {name}")
        os.environ.update({
            "ORT_MIGRAPHX_SAVE_COMPILED_MODEL": "0",
            "ORT_MIGRAPHX_LOAD_COMPILED_MODEL": "1",
            "ORT_MIGRAPHX_SAVE_COMPILE_PATH": str(cache),
            "ORT_MIGRAPHX_LOAD_COMPILE_PATH": str(cache),
            "MIGRAPHX_GPU_COMPILE_PARALLEL": "1",
            "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
        })
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("runtime drift")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output_dir / "ort_profile")
        cache_before = identity(cache)
        session = ort.InferenceSession(str(model), sess_options=options, providers=[(MGX, {"device_id": 0})])
        session.disable_fallback()
        inputs = [item.name for item in session.get_inputs()]
        outputs = [item.name for item in session.get_outputs()]
        expected_inputs = boundary_row.get("input_names") or [boundary_row["input_name"]]
        if inputs != expected_inputs or set(inputs) != set(boundaries):
            raise RuntimeError("target input name drift")
        device_inputs = {
            name: ort.OrtValue.ortvalue_from_numpy(boundaries[name], "cuda", 0)
            for name in inputs
        }
        binding = session.io_binding()
        for name in inputs:
            binding.bind_ortvalue_input(name, device_inputs[name])
        for name in outputs:
            binding.bind_output(name, "cuda", 0)
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        output_values = binding.get_outputs()
        fixed = session.io_binding()
        for name in inputs:
            fixed.bind_ortvalue_input(name, device_inputs[name])
        for name, value in zip(outputs, output_values):
            fixed.bind_ortvalue_output(name, value)
        fixed.synchronize_inputs()
        for _ in range(args.repetitions - 1):
            session.run_with_iobinding(fixed)
            fixed.synchronize_outputs()
        primary_name = row.get("primary_output") or next(
            (name for name in outputs if f"blocks.{row['end_block']}/Add_1_output_0" in name),
            outputs[-1],
        )
        primary = output_values[outputs.index(primary_name)].numpy()
        if not np.isfinite(primary).all():
            raise RuntimeError("non-finite target output")
        profile = Path(session.end_profiling()).resolve(strict=True)
        events = json.loads(profile.read_text(encoding="utf-8"))
        counts = Counter(
            str(event["args"]["provider"])
            for event in events if event.get("args", {}).get("provider")
        )
        gates = {
            "migraphx_events_positive": counts[MGX] > 0,
            "cpu_events_zero": counts[CPU] == 0,
            "input_and_outputs_device_resident": all(
                value.device_name() == "cuda" for value in device_inputs.values()
            ) and all(
                value.device_name() == "cuda" for value in output_values
            ),
            "exact_repetitions": args.repetitions >= 1,
            "cache_identity_unchanged": cache_before == identity(cache),
            "load_only_no_compile_requested": True,
            "selected_region_only_no_upstream_sessions": True,
            "formal_90_image_test_not_used": True,
        }
        result.update({
            "status": "target_passed_pending_external_hipprof_parse" if all(gates.values()) else "failed",
            "manifest": identity(manifest_path),
            "prepass_result": identity(prepass_path),
            "selection": {
                "session_ordinal": args.session_ordinal,
                "session_id": row["session_id"],
                "start_block": row["start_block"],
                "end_block": row["end_block"],
                "precision": row["precision"],
                "repetitions": args.repetitions,
            },
            "model": identity(model),
            "cache": identity(cache),
            "boundary": identity(boundary_path),
            "ort_profile": identity(profile),
            "provider_event_counts": dict(counts),
            "output": {
                "name": primary_name,
                "shape": list(primary.shape),
                "dtype": str(primary.dtype),
                "finite": True,
                "sha256": hashlib.sha256(np.ascontiguousarray(primary).tobytes()).hexdigest(),
            },
            "gates": gates,
            "trace_contract": {
                "mode": "direct_outer_hipprof_no_dynamic_session",
                "sessions_created": 1,
                "upstream_sessions_in_trace": 0,
                "target_region_calls": args.repetitions,
            },
        })
    except BaseException as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "target_passed_pending_external_hipprof_parse" else 2


if __name__ == "__main__":
    raise SystemExit(main())
