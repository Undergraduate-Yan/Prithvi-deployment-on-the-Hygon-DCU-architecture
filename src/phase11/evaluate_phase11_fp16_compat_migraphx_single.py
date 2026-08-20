#!/usr/bin/env python3
"""Compile/cache and diagnose the Phase-11 FP16 compatibility candidate."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch


SOURCE_SIZE = 638_894_819
SOURCE_SHA256 = "10df534d4dbaabf8336e4195a60d2ebd719b14c7a3e34362d48a0609425e6dbd"
CANDIDATE_SIZE = 638_970_735
CANDIDATE_SHA256 = "8ad6b71482be31ebcc1d5322a9671cc15d44113d0d2597ed715a98e0dc1089f7"
BUILD_REPORT_SIZE = 15_591
BUILD_REPORT_SHA256 = "93293dc0909f5158f983ce9dc5b7953cf4956a116f73e54aab7e86e2a7cd3fba"
SAMPLE_SIZE = 4_820_344
SAMPLE_SHA256 = "8ba78a29f324ab0bf3c188c8cb105e76b06925f22abb8c2da104a2fd87376e4b"
IMAGE_ID = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def lock(path: Path, size: int, digest: str, label: str) -> dict:
    item = record(path)
    if item["size_bytes"] != size or item["sha256"] != digest:
        raise RuntimeError(f"{label} identity mismatch: {item}")
    return item


def raw_image(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def visit(value):
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)
    if not found:
        raise RuntimeError("sample image not found")
    return np.ascontiguousarray(found[0][:1].cpu().numpy(), dtype=np.float32)


def options(profile_prefix: Path | None = None, strict: bool = False):
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


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    pa, pb = np.argmax(a, axis=1), np.argmax(b, axis=1)
    return {
        "mae": float(delta.mean()),
        "max_abs": float(delta.max()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "p99_abs": float(np.percentile(delta, 99)),
        "pixel_class_agreement": float(np.mean(pa == pb)),
        "changed_pixels": int(np.count_nonzero(pa != pb)),
    }


def release() -> None:
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def provider_counts(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return dict(sorted(counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--build-report", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_file = lock(args.source, SOURCE_SIZE, SOURCE_SHA256, "source")
    candidate_file = lock(args.candidate, CANDIDATE_SIZE, CANDIDATE_SHA256, "candidate")
    report_file = lock(args.build_report, BUILD_REPORT_SIZE, BUILD_REPORT_SHA256, "build report")
    sample_file = lock(args.sample, SAMPLE_SIZE, SAMPLE_SHA256, "sample")
    report = json.loads(args.build_report.read_text(encoding="utf-8"))
    if report.get("status") != "created_static_pass" or report["candidate"]["sha256"] != CANDIDATE_SHA256:
        raise RuntimeError("build report semantics mismatch")
    if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
        raise RuntimeError("official ORT/MIGraphX runtime mismatch")
    image = raw_image(args.sample)

    started = perf_counter()
    cpu_source = ort.InferenceSession(str(args.source), sess_options=options(), providers=[CPU])
    logits_source = np.asarray(cpu_source.run(["logits"], {"image": image})[0])
    source_seconds = perf_counter() - started
    del cpu_source
    release()
    started = perf_counter()
    cpu_candidate = ort.InferenceSession(str(args.candidate), sess_options=options(), providers=[CPU])
    logits_cpu = np.asarray(cpu_candidate.run(["logits"], {"image": image})[0])
    candidate_cpu_seconds = perf_counter() - started
    del cpu_candidate
    release()

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists():
        raise RuntimeError("cache output already exists")
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(args.cache.resolve())
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(args.cache.resolve())
    profile_prefix = args.output_dir / "fp16_compat_migraphx_profile"
    started = perf_counter()
    mgx = ort.InferenceSession(
        str(args.candidate),
        sess_options=options(profile_prefix, strict=True),
        providers=[(MGX, {"device_id": 0})],
    )
    mgx.disable_fallback()
    creation_seconds = perf_counter() - started
    registered = mgx.get_providers()
    started = perf_counter()
    logits_mgx = np.asarray(mgx.run(["logits"], {"image": image})[0])
    inference_seconds = perf_counter() - started
    profile_path = Path(mgx.end_profiling()).resolve(strict=True)
    counts = provider_counts(profile_path)
    if not args.cache.is_file() or args.cache.stat().st_size <= 0:
        raise RuntimeError("MIGraphX compiled cache was not created")

    transform = compare(logits_source, logits_cpu)
    runtime = compare(logits_cpu, logits_mgx)
    finite = bool(np.isfinite(logits_source).all() and np.isfinite(logits_cpu).all() and np.isfinite(logits_mgx).all())
    formal_exact_gates = {
        "mae_le_1e_6": transform["mae"] <= 1e-6,
        "max_abs_le_1e_5": transform["max_abs"] <= 1e-5,
        "agreement_eq_100pct": transform["pixel_class_agreement"] == 1.0,
    }
    downstream_semantic_gates = {
        "finite": finite,
        "cpu_transform_mae_le_1e_3": transform["mae"] <= 1e-3,
        "cpu_transform_max_le_1e_2": transform["max_abs"] <= 1e-2,
        "cpu_transform_agreement_eq_100pct": transform["pixel_class_agreement"] == 1.0,
        "migraphx_vs_cpu_mae_le_1e_3": runtime["mae"] <= 1e-3,
        "migraphx_vs_cpu_max_le_5e_2": runtime["max_abs"] <= 5e-2,
        "migraphx_vs_cpu_agreement_ge_99_9pct": runtime["pixel_class_agreement"] >= 0.999,
        "migraphx_events_positive": counts.get(MGX, 0) > 0,
        "cpu_events_zero": counts.get(CPU, 0) == 0,
    }
    npz = args.output_dir / "paired_logits.npz"
    np.savez_compressed(npz, image=image, source_cpu=logits_source, candidate_cpu=logits_cpu, candidate_migraphx=logits_mgx)
    accepted_for_90 = all(downstream_semantic_gates.values())
    result = {
        "status": "strict_migraphx_diagnostic_passed" if accepted_for_90 else "failed",
        "inputs": {"source": source_file, "candidate": candidate_file, "build_report": report_file, "sample": sample_file},
        "runtime": {
            "onnxruntime": ort.__version__, "container_image_id": IMAGE_ID,
            "registered_providers": registered, "source_cpu_seconds": source_seconds,
            "candidate_cpu_seconds": candidate_cpu_seconds,
            "migraphx_compile_and_session_seconds": creation_seconds,
            "first_inference_seconds_diagnostic_only": inference_seconds,
        },
        "comparisons": {"candidate_cpu_vs_source_cpu": transform, "candidate_migraphx_vs_candidate_cpu": runtime},
        "formal_exact_transform_gate": {"passed": all(formal_exact_gates.values()), "gates": formal_exact_gates},
        "downstream_task_evaluation_gate": {"passed": accepted_for_90, "gates": downstream_semantic_gates},
        "profile": {**record(profile_path), "provider_event_counts": counts},
        "compiled_cache": record(args.cache),
        "artifacts": {"paired_logits": record(npz)},
        "claims": {
            "strict_migraphx_single_sample_placement": counts.get(MGX, 0) > 0 and counts.get(CPU, 0) == 0,
            "formal_exact_transform_equivalence": all(formal_exact_gates.values()),
            "eligible_for_90_sample_task_evaluation": accepted_for_90,
            "task_accuracy_90": False, "performance": False, "deployment_complete": False,
        },
    }
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if accepted_for_90 else 1)


if __name__ == "__main__":
    main()
