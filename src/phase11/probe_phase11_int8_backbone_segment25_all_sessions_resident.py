#!/usr/bin/env python3
"""Load all 25 cached MIGraphX sessions concurrently and run one request."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


REPORT = (21_342, "e43daaa5dd8914c3632d0548b3f41f3d8c7d3b28f7be0bf391949b686532bb1e")
SAMPLE = (4_820_344, "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62")
DEVICE_SINGLE = (40_245, "7a7907e7e90a08ea58badbccd9302aa0fbfc814d3a3e33dcc2839a03e8c7d881")
DEVICE_SINGLE_LOGITS = (1_943_271, "535cee1507c33759e0b2366d02be16ed6ab963c90710be77e2d41742a2718bb1")
RAW_SHA = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS = tuple(f"encoder_block_{i:02d}" for i in range(24)) + ("upernet_decoder_head",)
RETAIN_AFTER = (5, 11, 17, 23)
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lock(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_raw(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def visit(value):
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(obj)
    raw = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if hashlib.sha256(raw.tobytes()).hexdigest() != RAW_SHA:
        raise RuntimeError("raw input identity drift")
    return raw


def options(ort, prefix: Path):
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    value.intra_op_num_threads = 4
    value.inter_op_num_threads = 1
    value.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    value.enable_profiling = True
    value.profile_file_prefix = str(prefix)
    return value


def parse_profile(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return {"provider_event_counts": dict(counts), "passed": counts[MGX] > 0 and counts[CPU] == 0}


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    p0, p1 = np.argmax(reference, axis=1), np.argmax(candidate, axis=1)
    return {
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "pixel_class_agreement": float(np.mean(p0 == p1)),
        "changed_pixels": int(np.count_nonzero(p0 != p1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "variant": "int8_backbone_25_cached_sessions_concurrently_resident_single",
        "claims": {
            "all_25_sessions_concurrently_resident": False,
            "device_resident_single_request_pipeline": False,
            "task_utility_test90": False,
            "strict_numeric_admission": False,
            "performance": False,
            "native_int8_kernel_verified": False,
            "deployment_ready": False,
        },
    }
    sessions = []
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("runtime identity drift")
        segment_root = args.root / "segment25_static_build_cpu"
        report_path = segment_root / "segment_build_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report["segments"]
        if len(rows) != 25 or [row["label"] for row in rows] != list(LABELS):
            raise RuntimeError("segment manifest/order drift")
        device_single_root = args.root / "segment25_iobinding_end_to_end_single" / "output"
        identities = {
            "segment_report": lock(report_path, REPORT),
            "sample": lock(args.root / "sample_and_logits.pt", SAMPLE),
            "passed_sequential_session_device_single": lock(device_single_root / "result.json", DEVICE_SINGLE),
            "sequential_session_device_logits": lock(
                device_single_root / "cpu_host_staged_and_device_logits.npz", DEVICE_SINGLE_LOGITS
            ),
        }
        frozen_single = json.loads((device_single_root / "result.json").read_text(encoding="utf-8"))
        if frozen_single.get("status") != "passed" or not frozen_single.get("claims", {}).get(
            "device_resident_intersegment_io_single_sample"
        ):
            raise RuntimeError("sequential-session device prerequisite failed")
        frozen_arrays = np.load(
            device_single_root / "cpu_host_staged_and_device_logits.npz", allow_pickle=False
        )
        reference_logits = np.ascontiguousarray(
            frozen_arrays["device_resident_migraphx_logits"], dtype=np.float32
        )

        models, caches, frozen_cache_evidence = [], [], []
        for index, row in enumerate(rows):
            model = segment_root / "models" / Path(row["path"]).name
            identities[row["label"]] = lock(model, (int(row["size_bytes"]), row["sha256"]))
            base = (
                args.root / "segment25_static_cache54g_segment0"
                if index == 0
                else args.root / "segment25_static_remaining_caches" / f"segment_{index:02d}"
            )
            frozen_path = base / "output/result.json"
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            cache_meta = frozen["compiled_cache"]
            cache = base / f"segment_{index:02d}.mxr"
            lock(cache, (int(cache_meta["size_bytes"]), cache_meta["sha256"]))
            models.append(model)
            caches.append(cache)
            frozen_cache_evidence.append({"index": index, "result": lock(frozen_path), "cache": cache_meta})

        load_started = perf_counter()
        load_rows = []
        for index, (model, cache) in enumerate(zip(models, caches, strict=True)):
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve())
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve())
            started = perf_counter()
            session = ort.InferenceSession(
                str(model),
                sess_options=options(ort, args.output_dir / f"segment_{index:02d}_resident_profile"),
                providers=[(MGX, {"device_id": 0})],
            )
            session.disable_fallback()
            if session.get_providers()[0] != MGX:
                raise RuntimeError(f"provider priority drift at segment {index}")
            sessions.append(session)
            load_rows.append({
                "index": index,
                "label": LABELS[index],
                "creation_seconds": perf_counter() - started,
                "process_max_rss_kib_after_load": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            })
            print(f"resident load {index + 1:02d}/25 {LABELS[index]} passed", flush=True)
        load_seconds = perf_counter() - load_started
        if len(sessions) != 25:
            raise RuntimeError("not all sessions remained resident")

        raw = load_raw(args.root / "sample_and_logits.pt")
        current = ort.OrtValue.ortvalue_from_numpy(raw, "cuda", 0)
        retained = {}
        inference_rows = []
        infer_started = perf_counter()
        for index, session in enumerate(sessions):
            binding = session.io_binding()
            if index < 24:
                binding.bind_ortvalue_input(session.get_inputs()[0].name, current)
            else:
                for item in session.get_inputs():
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_name = session.get_outputs()[0].name
            binding.bind_output(output_name, "cuda", 0)
            binding.synchronize_inputs()
            started = perf_counter()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            inference_rows.append({"index": index, "seconds_diagnostic_only": perf_counter() - started})
            outputs = binding.get_outputs()
            if len(outputs) != 1 or outputs[0].device_name() != "cuda":
                raise RuntimeError(f"device output contract drift at segment {index}")
            current = outputs[0]
            if index in RETAIN_AFTER:
                retained[output_name] = current
            del binding
        inference_seconds = perf_counter() - infer_started
        logits = np.ascontiguousarray(current.numpy(), dtype=np.float32)

        profiles, placement_ok = [], True
        for index, session in enumerate(sessions):
            profile_path = Path(session.end_profiling()).resolve(strict=True)
            placement = parse_profile(profile_path)
            placement_ok = placement_ok and placement["passed"]
            profiles.append({"index": index, **lock(profile_path), **placement})
        aligned = compare(reference_logits, logits)
        gates = {
            "all_25_sessions_concurrently_loaded": len(sessions) == 25,
            "all_25_profiles_migraphx_positive_cpu_zero": placement_ok,
            "final_output_device_cuda_alias": current.device_name() == "cuda",
            "resident_vs_sequential_device_mae_le_1e_6": aligned["mae"] <= 1e-6,
            "resident_vs_sequential_device_max_abs_le_1e_5": aligned["max_abs"] <= 1e-5,
            "resident_vs_sequential_device_predictions_exact": aligned["pixel_class_agreement"] == 1.0,
        }
        passed = all(gates.values())
        paired_path = args.output_dir / "sequential_and_all_resident_device_logits.npz"
        np.savez_compressed(
            paired_path,
            raw=raw,
            sequential_session_device_logits=reference_logits,
            all_sessions_resident_device_logits=logits,
        )
        result.update({
            "status": "passed" if passed else "failed",
            "identities": identities,
            "frozen_cache_evidence": frozen_cache_evidence,
            "runtime": {
                "onnxruntime": ort.__version__,
                "resident_session_count": len(sessions),
                "session_load_seconds_diagnostic_only": load_seconds,
                "single_request_chain_seconds_unwarmed_diagnostic_only": inference_seconds,
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "intersegment_transport": "direct_OrtValue_device_binding",
            },
            "session_loads": load_rows,
            "segment_inference": inference_rows,
            "profiles": profiles,
            "comparison_vs_sequential_session_device_pipeline": aligned,
            "gates": gates,
            "artifacts": {"paired_logits": lock(paired_path)},
            "evidence_boundary": {
                "only_one_unwarmed_request_tested": True,
                "formal_latency_repetitions_not_run": True,
                "strict_logits_numeric_gate_remains_failed": True,
                "all_resident_sessions_are_not_a_single_monolithic_model": True,
                "provider_profiles_do_not_prove_native_int8_kernels": True,
            },
        })
        result["claims"]["all_25_sessions_concurrently_resident"] = len(sessions) == 25
        result["claims"]["device_resident_single_request_pipeline"] = passed
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        for session in sessions:
            try:
                session.end_profiling()
            except Exception:
                pass

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
