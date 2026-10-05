#!/usr/bin/env python3
'Build contiguous FP16 region probes.'

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper


FULL_REGIONS = (
    {"id": "fp16_14_15", "start": 14, "end": 15},
    {"id": "fp16_17_20", "start": 17, "end": 20},
)
FALLBACK_PAIR_REGIONS = (
    {"id": "fp16_17_18", "start": 17, "end": 18},
    {"id": "fp16_19_20", "start": 19, "end": 20},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def verify_identity(row: dict[str, Any]) -> Path:
    path = Path(row["path"]).resolve(strict=True)
    if (path.stat().st_size, sha256(path)) != (int(row["size_bytes"]), str(row["sha256"])):
        raise RuntimeError(f"source model identity drift: {path}")
    return path


def array_record(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def inventory(model: onnx.ModelProto) -> dict[str, Any]:
    counts = Counter(node.op_type for node in model.graph.node)
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_type_counts": dict(sorted(counts.items())),
        "quantize_linear_count": counts["QuantizeLinear"],
        "dequantize_linear_count": counts["DequantizeLinear"],
        "cast_count": counts["Cast"],
    }


def merge_region(models: list[onnx.ModelProto], name: str) -> onnx.ModelProto:
    for left, right in zip(models, models[1:]):
        if [value.name for value in left.graph.output] != [value.name for value in right.graph.input]:
            raise RuntimeError(f"region boundary contract mismatch for {name}")
    initializers = [copy.deepcopy(value) for model in models for value in model.graph.initializer]
    names = [value.name for value in initializers]
    if len(names) != len(set(names)):
        duplicates = [key for key, count in Counter(names).items() if count > 1]
        raise RuntimeError(f"duplicate region initializers: {duplicates[:10]}")
    graph = helper.make_graph(
        [copy.deepcopy(node) for model in models for node in model.graph.node],
        name,
        [copy.deepcopy(value) for value in models[0].graph.input],
        [copy.deepcopy(value) for value in models[-1].graph.output],
        initializer=initializers,
    )
    merged = helper.make_model(graph)
    merged.ir_version = models[0].ir_version
    del merged.opset_import[:]
    merged.opset_import.extend(copy.deepcopy(models[0].opset_import))
    merged.producer_name = "journal_phase2d_lossless_contiguous_fp16_merge"
    onnx.checker.check_model(merged)
    return merged


def cpu_options(ort: Any, optimization_level: str) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimization_level == "all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-manifest", required=True, type=Path)
    parser.add_argument("--precision-map", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--profile", choices=("full", "fallback-pairs"), default="full")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    args.output_root.mkdir(parents=True)
    regions = FULL_REGIONS if args.profile == "full" else FALLBACK_PAIR_REGIONS

    b3 = json.loads(args.b3_manifest.resolve(strict=True).read_text(encoding="utf-8"))
    precision_map = json.loads(args.precision_map.resolve(strict=True).read_text(encoding="utf-8"))
    rows = b3["execution"]["segments"]
    if b3.get("variant") != "B3" or len(rows) != 25:
        raise RuntimeError("B3 source topology drift")
    if precision_map.get("name") != "M5-R" or precision_map.get("status") != "selected_and_configuration_validation_evaluated":
        raise RuntimeError("frozen MP-R/M5-R identity or status drift")
    frozen_fp16 = set(int(index) for index in precision_map["backbone"]["fp16_blocks"])
    selected_blocks = {index for region in regions for index in range(region["start"], region["end"] + 1)}
    if not selected_blocks <= frozen_fp16:
        raise RuntimeError(f"representative region is not FP16 in frozen M5-R: {selected_blocks - frozen_fp16}")
    for index in selected_blocks:
        if rows[index]["precision"] != "fp16":
            raise RuntimeError(f"B3 source block {index} is not the required FP16 graph")
    paths = [verify_identity(row["model"]) for row in rows]

    raw = np.load(args.input.resolve(strict=True), allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != (1, 6, 224, 224) or not np.isfinite(raw).all():
        raise RuntimeError("configuration-validation input contract drift")
    import onnxruntime as ort
    if ort.__version__ != "1.19.2":
        raise RuntimeError(f"ONNX Runtime drift: {ort.__version__}")

    models_dir = args.output_root / "models"
    feeds_dir = args.output_root / "feeds"
    models_dir.mkdir()
    feeds_dir.mkdir()
    built: dict[str, dict[str, Any]] = {}
    for region in regions:
        source_models = [onnx.load(str(paths[index]), load_external_data=False) for index in range(region["start"], region["end"] + 1)]
        for model in source_models:
            onnx.checker.check_model(model)
        merged = merge_region(source_models, f"M5_R_{region['id']}")
        model_path = models_dir / f"{region['id']}.onnx"
        onnx.save(merged, str(model_path), save_as_external_data=False)
        reloaded = onnx.load(str(model_path), load_external_data=False)
        onnx.checker.check_model(reloaded)
        built[region["id"]] = {
            "spec": region,
            "model": identity(model_path),
            "inventory": inventory(reloaded),
            "input_name": reloaded.graph.input[0].name,
            "output_name": reloaded.graph.output[0].name,
            "source_models": [identity(paths[index]) for index in range(region["start"], region["end"] + 1)],
        }
        del source_models, merged, reloaded
        gc.collect()

    starts = {region["start"]: region for region in regions}
    ends = {region["end"]: region for region in regions}
    current = np.ascontiguousarray(raw)
    references: dict[str, np.ndarray] = {}
    for index in range(max(ends) + 1):
        if index in starts:
            region = starts[index]
            input_name = str(rows[index]["inputs"][0]["name"])
            feed_path = feeds_dir / f"{region['id']}.npz"
            np.savez_compressed(feed_path, **{input_name: current})
            built[region["id"]]["feed"] = identity(feed_path)
            built[region["id"]]["feed_array"] = array_record(current)
        session = ort.InferenceSession(
            str(paths[index]), sess_options=cpu_options(ort, optimization_level="basic"),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        current = np.ascontiguousarray(session.run([output_name], {input_name: current})[0])
        if index in ends:
            references[ends[index]["id"]] = current.copy()
        del session
        gc.collect()

    region_rows = []
    all_contracts_pass = True
    for region in regions:
        row = built[region["id"]]
        with np.load(row["feed"]["path"], allow_pickle=False) as packed:
            feeds = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
        session = ort.InferenceSession(
            row["model"]["path"], sess_options=cpu_options(ort, optimization_level="basic"),
            providers=["CPUExecutionProvider"],
        )
        candidate = np.ascontiguousarray(session.run([row["output_name"]], feeds)[0])
        reference = references[region["id"]]
        difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
        semantic_denominator = float(
            np.linalg.norm(reference.astype(np.float64).reshape(-1))
        )
        semantic_comparison = {
            "mae": float(difference.mean()), "max_abs": float(difference.max()),
            "relative_l2": float(
                np.linalg.norm(
                    candidate.astype(np.float64).reshape(-1)
                    - reference.astype(np.float64).reshape(-1)
                ) / max(semantic_denominator, 1e-12)
            ),
            "array_equal": bool(np.array_equal(candidate, reference)),
            "allclose_rtol_1e_5_atol_1e_6": bool(np.allclose(candidate, reference, rtol=1e-5, atol=1e-6)),
            "candidate": array_record(candidate), "sequential_reference": array_record(reference),
        }
        optimized_current = next(iter(feeds.values()))
        for index in range(region["start"], region["end"] + 1):
            optimized_source = ort.InferenceSession(
                str(paths[index]), sess_options=cpu_options(ort, optimization_level="all"),
                providers=["CPUExecutionProvider"],
            )
            source_input = optimized_source.get_inputs()[0].name
            source_output = optimized_source.get_outputs()[0].name
            optimized_current = np.ascontiguousarray(
                optimized_source.run([source_output], {source_input: optimized_current})[0]
            )
            del optimized_source
            gc.collect()
        optimized_merged = ort.InferenceSession(
            row["model"]["path"], sess_options=cpu_options(ort, optimization_level="all"),
            providers=["CPUExecutionProvider"],
        )
        optimized_candidate = np.ascontiguousarray(
            optimized_merged.run([row["output_name"]], feeds)[0]
        )
        optimized_difference = np.abs(
            optimized_candidate.astype(np.float64) - optimized_current.astype(np.float64)
        )
        optimized_denominator = float(
            np.linalg.norm(optimized_current.astype(np.float64).reshape(-1))
        )
        optimized_comparison = {
            "mae": float(optimized_difference.mean()),
            "max_abs": float(optimized_difference.max()),
            "relative_l2": float(
                np.linalg.norm(
                    optimized_candidate.astype(np.float64).reshape(-1)
                    - optimized_current.astype(np.float64).reshape(-1)
                ) / max(optimized_denominator, 1e-12)
            ),
            "all_outputs_finite": bool(
                np.isfinite(optimized_candidate).all() and np.isfinite(optimized_current).all()
            ),
            "candidate": array_record(optimized_candidate),
            "sequential_reference": array_record(optimized_current),
            "interpretation": "reported optimizer/fusion-path drift; not used to prove graph semantic equivalence",
        }
        gates = {
            "all_blocks_fp16_in_frozen_M5_R": True,
            "onnx_checker_passed": True,
            "configuration_validation_input_only": True,
            "formal_90_image_test_not_used": True,
            "basic_optimized_cpu_mae_le_1e_3": semantic_comparison["mae"] <= 1e-3,
            "basic_optimized_cpu_max_abs_le_5e_2": semantic_comparison["max_abs"] <= 5e-2,
            "basic_optimized_cpu_relative_l2_le_1e_3": semantic_comparison["relative_l2"] <= 1e-3,
            "all_outputs_finite": bool(np.isfinite(candidate).all() and np.isfinite(reference).all()),
        }
        all_contracts_pass = all_contracts_pass and all(gates.values())
        region_rows.append({
            **row,
            "semantic_comparison_minimum_required_basic_optimization": semantic_comparison,
            "optimized_runtime_comparison": optimized_comparison,
            "gates": gates,
        })
        del session, optimized_merged
        gc.collect()

    report = {
        "schema": "journal_phase2d_contiguous_fp16_build_v1",
        "status": "passed" if all_contracts_pass else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_target": "海光 K100 AI 加速卡",
        "operation": "lossless_merge_of_each_nontrivial_contiguous_frozen_FP16_region",
        "profile": args.profile,
        "scope": "structural compile-feasibility probes; not an end-to-end M5-R task-accuracy result",
        "cpu_validation_note": "These frozen FP16 exports contain InsertedPrecisionFreeCast nodes that fail ORT type initialization with all graph optimizations disabled. ORT_ENABLE_BASIC is therefore the minimum executable semantic check; ORT_ENABLE_ALL drift is reported separately.",
        "source_manifest": identity(args.b3_manifest),
        "frozen_precision_map": identity(args.precision_map),
        "benchmark_input": identity(args.input),
        "selection_role": "configuration-validation structural probe only",
        "formal_90_image_test_used": False,
        "regions": region_rows,
        "gates": {
            "exact_requested_fp16_profile": (
                selected_blocks == {14, 15, 17, 18, 19, 20}
                if args.profile == "full"
                else selected_blocks == {17, 18, 19, 20}
            ),
            "all_selected_blocks_fp16_in_frozen_M5_R": selected_blocks <= frozen_fp16,
            "all_region_contracts_passed": all_contracts_pass,
            "formal_90_image_test_not_used": True,
        },
    }
    (args.output_root / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "regions": [row["spec"] for row in region_rows], "gates": report["gates"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" and all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
