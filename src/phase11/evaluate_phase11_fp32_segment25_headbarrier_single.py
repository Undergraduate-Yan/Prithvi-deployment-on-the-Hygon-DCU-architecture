#!/usr/bin/env python3
"""Single-sample K100 admission for the FP32 25-segment fpn4-barrier candidate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SEGMENT_REPORT = (24_959, "c94692d130a0eb924290b3d56f790e166535fa3049fd9f3e72ef1fe16085c90f")
CPU_PARITY = (14_840, "58c076e974ea7116a3f943bce6ca653fb0803993ac77af18c0b4cdfb837c2e72")
SAMPLE = (4_820_344, "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62")
PAIRED_INPUTS = (3_653_541, "98fe1142953a05944244f8934141b2500d787bfe9d549a2f3c34d7c643b49ab9")
RAW_SHA = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
LABELS = tuple(f"encoder_block_{index:02d}" for index in range(24)) + ("upernet_decoder_head_fpn4barrier",)
RETAIN_AFTER = (5, 11, 17, 23)
BARRIER = "/task/model/decoder/fpn4/fpn4.0/MaxPool_output_0"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock(path: Path, expected: tuple[int, str] | None = None) -> dict:
    path = path.resolve(strict=True)
    item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected is not None and (item["size_bytes"], item["sha256"]) != expected:
        raise RuntimeError(f"artifact identity drift: {item}")
    return item


def load_image(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    found: list[torch.Tensor] = []

    def visit(value) -> None:
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224):
            found.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(obj)
    if not found:
        raise RuntimeError("sample image missing")
    image = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if hashlib.sha256(image.tobytes(order="C")).hexdigest() != RAW_SHA:
        raise RuntimeError("raw input identity drift")
    return image


def options(ort, profile: Path | None = None, strict: bool = False):
    value = ort.SessionOptions()
    value.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    value.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    value.intra_op_num_threads = 4
    value.inter_op_num_threads = 1
    if strict:
        value.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile is not None:
        value.enable_profiling = True
        value.profile_file_prefix = str(profile)
    return value


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    left = np.argmax(reference, axis=1)
    right = np.argmax(candidate, axis=1)
    return {
        "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "pixel_class_agreement": float(np.mean(left == right)),
        "changed_pixels": int(np.count_nonzero(left != right)),
        "all_finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def parse_profile(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        str(event["args"]["provider"])
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    return {
        "provider_event_counts": dict(counts),
        "passed": int(counts.get(MGX, 0)) > 0 and int(counts.get(CPU, 0)) == 0,
    }


def validate_build(candidate: Path, report_path: Path) -> tuple[dict, dict]:
    report_identity = lock(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "phase11_fp32_segment25_head_fpn4barrier_build_v1":
        raise RuntimeError("candidate build schema drift")
    if report.get("status") != "passed" or not all(report.get("gates", {}).values()):
        raise RuntimeError("candidate CPU/build prerequisite failed")
    meta = report.get("candidate", {})
    candidate_identity = lock(candidate, (int(meta["size_bytes"]), str(meta["sha256"])))
    source = report.get("identities", {}).get("source_head", {})
    if (int(source.get("size_bytes", -1)), str(source.get("sha256"))) != (
        60_551_311,
        "2644c864768639151a062efcd0c69e657cdb0b7455c9b3561f08411397af2f6e",
    ):
        raise RuntimeError("candidate source lineage drift")
    return candidate_identity, report_identity


def cache_paths(root: Path, index: int) -> tuple[Path, Path]:
    base = root / "cache_ln_segment00" if index == 0 else root / "cache_ln_remaining" / f"segment_{index:02d}"
    return base / f"segment_{index:02d}.mxr", base / "output" / "result.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--head-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.head_cache.exists():
        raise RuntimeError("refusing to overwrite an existing head cache")
    result = {
        "schema": "phase11_fp32_segment25_headbarrier_single_v1",
        "status": "failed",
        "claims": {
            "candidate_cpu_semantics_exact": False,
            "head_local_strict_numeric_admission": False,
            "all_25_segments_strict_migraphx": False,
            "device_resident_intersegment_io": False,
            "strict_end_to_end_numeric_admission": False,
            "task90_eligible": False,
            "performance": False,
        },
    }
    try:
        import onnxruntime as ort

        if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
            raise RuntimeError("runtime identity drift")
        report_path = args.root / "build_ln" / "segment25_report_deconv.json"
        parity_path = args.root / "build_ln" / "cpu_sequential_parity_deconv.json"
        paired_path = args.root / "cache_ln_deconv_head" / "output" / "cpu_and_migraphx_output.npz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report.get("segments", [])
        if len(rows) != 25 or [row.get("label") for row in rows[:24]] != list(LABELS[:24]):
            raise RuntimeError("segment topology drift")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if parity.get("status") != "passed" or not parity.get("claims", {}).get("cpu_sequential_parity"):
            raise RuntimeError("frozen CPU parity prerequisite failed")
        candidate_identity, build_identity = validate_build(args.candidate, args.build_report)
        identities = {
            "segment_report": lock(report_path, SEGMENT_REPORT),
            "cpu_sequential_parity": lock(parity_path, CPU_PARITY),
            "sample": lock(args.sample, SAMPLE),
            "paired_head_inputs": lock(paired_path, PAIRED_INPUTS),
            "candidate": candidate_identity,
            "candidate_build_report": build_identity,
        }

        models: list[Path] = []
        caches: list[Path] = []
        frozen_cache_evidence = []
        for index, row in enumerate(rows[:24]):
            model = args.root / "build_ln" / "models" / Path(row["path"]).name
            identities[f"model_{index:02d}"] = lock(model, (int(row["size_bytes"]), str(row["sha256"])))
            cache, cache_result_path = cache_paths(args.root, index)
            cache_result = json.loads(cache_result_path.read_text(encoding="utf-8"))
            cache_meta = cache_result.get("compiled_cache", {})
            cache_identity = lock(cache, (int(cache_meta["size_bytes"]), str(cache_meta["sha256"])))
            counts = cache_result.get("profile", {}).get("provider_event_counts", {})
            if int(counts.get(MGX, 0)) <= 0 or int(counts.get(CPU, 0)) != 0:
                raise RuntimeError(f"frozen cache placement prerequisite failed at segment {index}")
            models.append(model)
            caches.append(cache)
            frozen_cache_evidence.append({"index": index, "cache": cache_identity, "result": lock(cache_result_path)})

        packed = np.load(paired_path, allow_pickle=False)
        cpu_reference = np.ascontiguousarray(packed["cpu_output"], dtype=np.float32)
        cpu_session = ort.InferenceSession(
            str(args.candidate), sess_options=options(ort), providers=["CPUExecutionProvider"]
        )
        head_feeds = {
            item.name: np.ascontiguousarray(packed[f"input_{index}"], dtype=np.float32)
            for index, item in enumerate(cpu_session.get_inputs())
        }
        cpu_candidate, cpu_barrier = cpu_session.run(["logits", BARRIER], head_feeds)
        cpu_candidate = np.ascontiguousarray(cpu_candidate, dtype=np.float32)
        cpu_semantics_exact = bool(np.array_equal(cpu_reference, cpu_candidate))
        if not cpu_semantics_exact:
            raise RuntimeError("candidate CPU logits do not exactly reproduce frozen source head")
        del cpu_session
        gc.collect()

        args.head_cache.parent.mkdir(parents=True, exist_ok=True)
        os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "1"
        os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "0"
        os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(args.head_cache.resolve())
        os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(args.head_cache.resolve())
        head_session = ort.InferenceSession(
            str(args.candidate),
            sess_options=options(ort, args.output_dir / "head_local_profile", strict=True),
            providers=[(MGX, {"device_id": 0})],
        )
        head_session.disable_fallback()
        binding = head_session.io_binding()
        for item in head_session.get_inputs():
            binding.bind_ortvalue_input(
                item.name, ort.OrtValue.ortvalue_from_numpy(head_feeds[item.name], "cuda", 0)
            )
        for name in ("logits", BARRIER):
            binding.bind_output(name, "cuda", 0)
        binding.synchronize_inputs()
        head_session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        head_values = binding.get_outputs()
        if len(head_values) != 2 or not all(value.device_name() == "cuda" for value in head_values):
            raise RuntimeError("head local device output contract drift")
        head_logits = np.ascontiguousarray(head_values[0].numpy(), dtype=np.float32)
        head_barrier = np.ascontiguousarray(head_values[1].numpy(), dtype=np.float32)
        head_profile_path = Path(head_session.end_profiling()).resolve(strict=True)
        head_placement = parse_profile(head_profile_path)
        del binding, head_session
        gc.collect()
        if not head_placement["passed"]:
            raise RuntimeError(f"head local strict placement failed: {head_placement}")
        head_cache_identity = lock(args.head_cache)
        head_comparison = compare(cpu_reference, head_logits)
        head_gates = {
            "migraphx_events_positive_cpu_zero": head_placement["passed"],
            "all_outputs_device": True,
            "mae_le_1e_3": head_comparison["mae"] <= 1e-3,
            "max_abs_le_5e_2": head_comparison["max_abs"] <= 5e-2,
            "pixel_class_agreement_ge_99_9pct": head_comparison["pixel_class_agreement"] >= 0.999,
            "all_finite": bool(head_comparison["all_finite"] and np.isfinite(head_barrier).all()),
        }
        if not all(head_gates.values()):
            raise RuntimeError(f"repaired head local numeric admission failed: {head_gates}")

        current = ort.OrtValue.ortvalue_from_numpy(load_image(args.sample), "cuda", 0)
        retained: dict[str, object] = {}
        profiles = []
        runtime_rows = []
        for index in range(25):
            model = models[index] if index < 24 else args.candidate
            cache = caches[index] if index < 24 else args.head_cache
            os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
            os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
            os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache.resolve())
            os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache.resolve())
            session = ort.InferenceSession(
                str(model),
                sess_options=options(ort, args.output_dir / f"segment_{index:02d}_profile", strict=True),
                providers=[(MGX, {"device_id": 0})],
            )
            session.disable_fallback()
            run_binding = session.io_binding()
            if index < 24:
                run_binding.bind_ortvalue_input(session.get_inputs()[0].name, current)
            else:
                if set(item.name for item in session.get_inputs()) != set(retained):
                    raise RuntimeError("head retained-feature contract drift")
                for item in session.get_inputs():
                    run_binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                run_binding.bind_output(name, "cuda", 0)
            run_binding.synchronize_inputs()
            session.run_with_iobinding(run_binding)
            run_binding.synchronize_outputs()
            output_values = run_binding.get_outputs()
            if not output_values or not all(value.device_name() == "cuda" for value in output_values):
                raise RuntimeError(f"device output contract drift at segment {index}")
            output_map = dict(zip(output_names, output_values, strict=True))
            current = output_map[output_names[0] if index < 24 else "logits"]
            if index in RETAIN_AFTER:
                retained[output_names[0]] = current
            profile_path = Path(session.end_profiling()).resolve(strict=True)
            placement = parse_profile(profile_path)
            if not placement["passed"]:
                raise RuntimeError(f"strict placement failed at segment {index}: {placement}")
            profiles.append({"index": index, **lock(profile_path), **placement})
            runtime_rows.append({
                "index": index,
                "label": LABELS[index],
                "outputs": output_names,
                "all_outputs_device": all(value.device_name() == "cuda" for value in output_values),
            })
            del run_binding, session
            gc.collect()
            print(f"FP32 headbarrier single segment {index + 1:02d}/25 passed", flush=True)

        device_logits = np.ascontiguousarray(current.numpy(), dtype=np.float32)
        end_to_end = compare(cpu_reference, device_logits)
        end_to_end_gates = {
            "all_25_profiles_migraphx_positive_cpu_zero": len(profiles) == 25,
            "all_outputs_device": all(row["all_outputs_device"] for row in runtime_rows),
            "mae_le_1e_3": end_to_end["mae"] <= 1e-3,
            "max_abs_le_5e_2": end_to_end["max_abs"] <= 5e-2,
            "pixel_class_agreement_ge_99_9pct": end_to_end["pixel_class_agreement"] >= 0.999,
            "all_finite": end_to_end["all_finite"],
        }
        task90_eligible = bool(
            end_to_end_gates["all_25_profiles_migraphx_positive_cpu_zero"]
            and end_to_end_gates["all_outputs_device"]
            and end_to_end_gates["all_finite"]
            and all(head_gates.values())
            and cpu_semantics_exact
        )
        paired_out = args.output_dir / "cpu_headlocal_and_device_logits.npz"
        np.savez_compressed(
            paired_out,
            cpu_reference=cpu_reference,
            head_local_migraphx_logits=head_logits,
            device_pipeline_logits=device_logits,
            head_local_fpn4_maxpool=head_barrier,
        )
        result.update({
            "status": "diagnostic_completed" if task90_eligible else "failed",
            "runtime": {"onnxruntime": ort.__version__, "intersegment_transport": "device_OrtValue"},
            "identities": identities,
            "frozen_cache_evidence": frozen_cache_evidence,
            "new_head_cache": head_cache_identity,
            "head_local_profile": {**lock(head_profile_path), **head_placement},
            "head_local_comparison": head_comparison,
            "head_local_gates": head_gates,
            "segments": runtime_rows,
            "profiles": profiles,
            "end_to_end_comparison": end_to_end,
            "strict_end_to_end_gates": end_to_end_gates,
            "artifacts": {"paired_outputs": lock(paired_out)},
            "evidence_boundary": {
                "strict_logits_result_is_recorded_but_not_task90_hard_prerequisite": True,
                "timings_not_measured": True,
                "performance_claim": False,
                "monolithic_deployment_ready": False,
            },
        })
        result["claims"].update({
            "candidate_cpu_semantics_exact": cpu_semantics_exact,
            "head_local_strict_numeric_admission": all(head_gates.values()),
            "all_25_segments_strict_migraphx": end_to_end_gates["all_25_profiles_migraphx_positive_cpu_zero"],
            "device_resident_intersegment_io": end_to_end_gates["all_outputs_device"],
            "strict_end_to_end_numeric_admission": all(end_to_end_gates.values()),
            "task90_eligible": task90_eligible,
        })
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "diagnostic_completed" else 2)


if __name__ == "__main__":
    main()
