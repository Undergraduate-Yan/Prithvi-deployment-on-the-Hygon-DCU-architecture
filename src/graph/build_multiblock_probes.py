#!/usr/bin/env python3
'Build contiguous multi-block graph-merging probes.'

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify(row: dict[str, Any]) -> Path:
    path = Path(row["path"]).resolve(strict=True)
    actual = identity(path)
    if (actual["size_bytes"], actual["sha256"]) != (int(row["size_bytes"]), str(row["sha256"])):
        raise RuntimeError(f"identity drift: {path}")
    return path


def array_record(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
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


def merge_region(
    models: list[onnx.ModelProto], name: str, retain_offsets: list[int]
) -> onnx.ModelProto:
    if len(models) < 3:
        raise RuntimeError("multi-block probe requires at least three source graphs")
    for left, right in zip(models, models[1:], strict=False):
        left_outputs = [value.name for value in left.graph.output]
        right_inputs = [value.name for value in right.graph.input]
        if left_outputs != right_inputs or len(left_outputs) != 1:
            raise RuntimeError(f"region boundary contract mismatch for {name}: {left_outputs} != {right_inputs}")
    opsets = [[(item.domain, item.version) for item in model.opset_import] for model in models]
    if any(row != opsets[0] for row in opsets[1:]):
        raise RuntimeError(f"opset drift for {name}")
    initializers = [copy.deepcopy(value) for model in models for value in model.graph.initializer]
    names = [value.name for value in initializers]
    if len(names) != len(set(names)):
        duplicates = [key for key, count in Counter(names).items() if count > 1]
        raise RuntimeError(f"duplicate initializers for {name}: {duplicates[:10]}")
    output_values = [copy.deepcopy(models[offset].graph.output[0]) for offset in retain_offsets]
    output_values.append(copy.deepcopy(models[-1].graph.output[0]))
    output_names = [value.name for value in output_values]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError(f"duplicate retained/final outputs for {name}")
    graph = helper.make_graph(
        [copy.deepcopy(node) for model in models for node in model.graph.node],
        name,
        [copy.deepcopy(value) for value in models[0].graph.input],
        output_values,
        initializer=initializers,
    )
    merged = helper.make_model(graph)
    merged.ir_version = models[0].ir_version
    del merged.opset_import[:]
    merged.opset_import.extend(copy.deepcopy(models[0].opset_import))
    merged.producer_name = "journal_phase2d_controlled_rcs_contiguous_multiblock_merge"
    onnx.checker.check_model(merged)
    return merged


def cpu_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--regions-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "prepared_for_cache_load_validation"
        or not all(manifest.get("gates", {}).values())
        or manifest.get("selection_boundary", {}).get("formal_90_image_test_used") is not False
    ):
        raise RuntimeError("source RCS manifest is not admitted")
    reference = manifest["reference_sessions"]
    if len(reference) != 25:
        raise RuntimeError("exact 25-session corrected M5-R reference required")
    source_by_block = {int(row["start_block"]): row for row in reference}
    if set(source_by_block) != set(range(25)) or any(
        int(row["start_block"]) != int(row["end_block"]) for row in reference
    ):
        raise RuntimeError("reference must contain exact singleton sessions")

    regions_path = args.regions_json.resolve(strict=True)
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    required = {"id", "role", "start", "end", "expected_precisions"}
    allowed = required | {"retain_outputs"}
    if not isinstance(regions, list) or not regions or any(
        not required <= set(row) or not set(row) <= allowed for row in regions
    ):
        raise RuntimeError("declared region schema drift")
    occupied: list[int] = []
    for row in regions:
        start, end = int(row["start"]), int(row["end"])
        if not 0 <= start < end <= 23 or end - start + 1 < 3:
            raise RuntimeError(f"only contiguous multi-block backbone regions are allowed: {row}")
        expected = [source_by_block[index]["precision"] for index in range(start, end + 1)]
        if row["expected_precisions"] != expected:
            raise RuntimeError(f"precision declaration drift for {row['id']}: {row['expected_precisions']} != {expected}")
        retain_outputs = [int(index) for index in row.get("retain_outputs", [])]
        if len(retain_outputs) != len(set(retain_outputs)) or any(
            not start <= index < end for index in retain_outputs
        ):
            raise RuntimeError(f"invalid retained output declaration for {row['id']}: {retain_outputs}")
        occupied.extend(range(start, end + 1))
    if len(occupied) != len(set(occupied)):
        raise RuntimeError("regions in one compile tier must be non-overlapping")

    paths = [verify(source_by_block[index]["model"]) for index in range(25)]
    raw = np.load(verify(manifest["input"]), allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != (1, 6, 224, 224) or not np.isfinite(raw).all():
        raise RuntimeError("configuration-validation input contract drift")
    import onnxruntime as ort
    if ort.__version__ != "1.19.2":
        raise RuntimeError(f"ONNX Runtime drift: {ort.__version__}")

    args.output_root.mkdir(parents=True)
    models_dir = args.output_root / "models"
    feeds_dir = args.output_root / "feeds"
    models_dir.mkdir()
    feeds_dir.mkdir()
    built: dict[str, dict[str, Any]] = {}
    for region in regions:
        start, end = int(region["start"]), int(region["end"])
        source_models = [onnx.load(str(paths[index]), load_external_data=False) for index in range(start, end + 1)]
        for model in source_models:
            onnx.checker.check_model(model)
        retain_outputs = [int(index) for index in region.get("retain_outputs", [])]
        merged = merge_region(
            source_models,
            f"M5_R_{region['id']}",
            [index - start for index in retain_outputs],
        )
        model_path = models_dir / f"{region['id']}.onnx"
        onnx.save(merged, str(model_path), save_as_external_data=False)
        reloaded = onnx.load(str(model_path), load_external_data=False)
        onnx.checker.check_model(reloaded)
        built[region["id"]] = {
            "spec": region,
            "model": identity(model_path),
            "inventory": inventory(reloaded),
            "input_name": reloaded.graph.input[0].name,
            "outputs": [value.name for value in reloaded.graph.output],
            "primary_output": source_models[-1].graph.output[0].name,
            "retained_outputs": {
                str(index): source_models[index - start].graph.output[0].name
                for index in retain_outputs
            },
            "source_models": [identity(paths[index]) for index in range(start, end + 1)],
        }
        del source_models, merged, reloaded
        gc.collect()

    starts = {int(row["start"]): row for row in regions}
    ends = {int(row["end"]): row for row in regions}
    current = np.ascontiguousarray(raw)
    reference_indexes = set(ends) | {
        int(index) for row in regions for index in row.get("retain_outputs", [])
    }
    references: dict[int, np.ndarray] = {}
    for index in range(max(ends) + 1):
        if index in starts:
            region = starts[index]
            model = onnx.load(str(paths[index]), load_external_data=False)
            input_name = model.graph.input[0].name
            del model
            feed_path = feeds_dir / f"{region['id']}.npz"
            np.savez_compressed(feed_path, **{input_name: current})
            built[region["id"]]["feed"] = identity(feed_path)
            built[region["id"]]["feed_array"] = array_record(current)
        session = ort.InferenceSession(str(paths[index]), sess_options=cpu_options(ort), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        current = np.ascontiguousarray(session.run([output_name], {input_name: current})[0])
        if index in reference_indexes:
            references[index] = current.copy()
        del session
        gc.collect()

    rows = []
    for region in regions:
        record = built[region["id"]]
        with np.load(record["feed"]["path"], allow_pickle=False) as packed:
            feeds = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
        session = ort.InferenceSession(record["model"]["path"], sess_options=cpu_options(ort), providers=["CPUExecutionProvider"])
        candidate_values = [
            np.ascontiguousarray(value) for value in session.run(record["outputs"], feeds)
        ]
        candidates = dict(zip(record["outputs"], candidate_values, strict=True))
        output_blocks = {
            **{name: int(index) for index, name in record["retained_outputs"].items()},
            record["primary_output"]: int(region["end"]),
        }
        comparisons = {}
        for output_name, block_index in output_blocks.items():
            candidate = candidates[output_name]
            reference_output = references[block_index]
            difference = np.abs(candidate.astype(np.float64) - reference_output.astype(np.float64))
            comparisons[output_name] = {
                "source_block": block_index,
                "mae": float(difference.mean()),
                "max_abs": float(difference.max()),
                "allclose_rtol_1e_5_atol_1e_6": bool(
                    np.allclose(candidate, reference_output, rtol=1e-5, atol=1e-6)
                ),
                "candidate": array_record(candidate),
                "sequential_reference": array_record(reference_output),
            }
        comparison = comparisons[record["primary_output"]]
        gates = {
            "exact_declared_contiguous_source_blocks": len(record["source_models"]) == int(region["end"]) - int(region["start"]) + 1,
            "declared_precision_sequence_matches_corrected_M5_R": True,
            "onnx_checker_passed": True,
            "cpu_region_matches_sequential_source": all(
                row["allclose_rtol_1e_5_atol_1e_6"] for row in comparisons.values()
            ),
            "all_declared_retained_outputs_preserved": len(record["retained_outputs"])
            == len(region.get("retain_outputs", [])),
            "all_outputs_finite": all(
                value["candidate"]["finite"] and value["sequential_reference"]["finite"]
                for value in comparisons.values()
            ),
            "configuration_validation_input_only": True,
            "formal_90_image_test_not_used": True,
        }
        rows.append({
            **record,
            "comparison": comparison,
            "comparisons": comparisons,
            "gates": gates,
        })
        del session
        gc.collect()

    report = {
        "schema": "journal_phase2d_controlled_rcs_multiblock_build_v1",
        "status": "passed" if all(all(row["gates"].values()) for row in rows) else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_target": "海光 K100 AI 加速卡",
        "source_manifest": identity(manifest_path),
        "benchmark_input": identity(Path(manifest["input"]["path"])),
        "declared_regions": identity(regions_path),
        "operation": "lossless_merge_of_explicit_contiguous_corrected_M5_R_backbone_graphs",
        "selection_role": "configuration-validation structural compile probe only",
        "regions": rows,
        "gates": {
            "all_region_contracts_passed": all(all(row["gates"].values()) for row in rows),
            "corrected_M5_R_reference_used": True,
            "formal_90_image_test_not_used": True,
        },
        "formal_90_image_test_used": False,
    }
    (args.output_root / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" and all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
