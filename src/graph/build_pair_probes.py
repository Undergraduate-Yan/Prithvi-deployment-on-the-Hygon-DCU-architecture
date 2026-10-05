#!/usr/bin/env python3
'Build adjacent block graph-merging probes.'

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
    if (actual["size_bytes"], actual["sha256"]) != (
        int(row["size_bytes"]),
        str(row["sha256"]),
    ):
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


def merge_pair(first: onnx.ModelProto, second: onnx.ModelProto, name: str) -> onnx.ModelProto:
    first_outputs = [value.name for value in first.graph.output]
    second_inputs = [value.name for value in second.graph.input]
    if first_outputs != second_inputs or len(first_outputs) != 1:
        raise RuntimeError(f"pair boundary contract mismatch for {name}: {first_outputs} != {second_inputs}")
    if [(item.domain, item.version) for item in first.opset_import] != [
        (item.domain, item.version) for item in second.opset_import
    ]:
        raise RuntimeError(f"opset drift for {name}")
    initializers = [
        copy.deepcopy(value)
        for model in (first, second)
        for value in model.graph.initializer
    ]
    names = [value.name for value in initializers]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate initializers for {name}")
    graph = helper.make_graph(
        [copy.deepcopy(node) for model in (first, second) for node in model.graph.node],
        name,
        [copy.deepcopy(value) for value in first.graph.input],
        [copy.deepcopy(value) for value in second.graph.output],
        initializer=initializers,
    )
    merged = helper.make_model(graph)
    merged.ir_version = first.ir_version
    del merged.opset_import[:]
    merged.opset_import.extend(copy.deepcopy(first.opset_import))
    merged.producer_name = "journal_phase2d_controlled_rcs_adjacent_pair_merge"
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
    if not isinstance(regions, list) or not regions or any(set(row) != required for row in regions):
        raise RuntimeError("declared region schema drift")
    flattened = []
    for row in regions:
        start, end = int(row["start"]), int(row["end"])
        if end != start + 1 or not 0 <= start < end <= 23:
            raise RuntimeError(f"only adjacent backbone pairs are allowed: {row}")
        expected = [source_by_block[index]["precision"] for index in (start, end)]
        if row["expected_precisions"] != expected:
            raise RuntimeError(f"precision declaration drift for {row['id']}: {row['expected_precisions']} != {expected}")
        flattened.extend((start, end))
    if len(flattened) != len(set(flattened)):
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
        first = onnx.load(str(paths[start]), load_external_data=False)
        second = onnx.load(str(paths[end]), load_external_data=False)
        onnx.checker.check_model(first)
        onnx.checker.check_model(second)
        merged = merge_pair(first, second, f"M5_R_{region['id']}")
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
            "source_models": [identity(paths[start]), identity(paths[end])],
        }
        del first, second, merged, reloaded
        gc.collect()

    starts = {int(row["start"]): row for row in regions}
    ends = {int(row["end"]): row for row in regions}
    current = np.ascontiguousarray(raw)
    references: dict[str, np.ndarray] = {}
    for index in range(24):
        if index in starts:
            region = starts[index]
            input_name = str(source_by_block[index]["model"].get("input_name", ""))
            model = onnx.load(str(paths[index]), load_external_data=False)
            input_name = model.graph.input[0].name
            del model
            feed_path = feeds_dir / f"{region['id']}.npz"
            np.savez_compressed(feed_path, **{input_name: current})
            built[region["id"]]["feed"] = identity(feed_path)
            built[region["id"]]["feed_array"] = array_record(current)
        session = ort.InferenceSession(
            str(paths[index]), sess_options=cpu_options(ort), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        current = np.ascontiguousarray(session.run([output_name], {input_name: current})[0])
        if index in ends:
            references[ends[index]["id"]] = current.copy()
        del session
        gc.collect()

    rows = []
    for region in regions:
        record = built[region["id"]]
        with np.load(record["feed"]["path"], allow_pickle=False) as packed:
            feeds = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
        session = ort.InferenceSession(
            record["model"]["path"], sess_options=cpu_options(ort), providers=["CPUExecutionProvider"]
        )
        candidate = np.ascontiguousarray(session.run([record["output_name"]], feeds)[0])
        reference_output = references[region["id"]]
        difference = np.abs(candidate.astype(np.float64) - reference_output.astype(np.float64))
        comparison = {
            "mae": float(difference.mean()),
            "max_abs": float(difference.max()),
            "allclose_rtol_1e_5_atol_1e_6": bool(
                np.allclose(candidate, reference_output, rtol=1e-5, atol=1e-6)
            ),
            "candidate": array_record(candidate),
            "sequential_reference": array_record(reference_output),
        }
        gates = {
            "exact_two_adjacent_source_blocks": True,
            "declared_precision_pair_matches_corrected_M5_R": True,
            "onnx_checker_passed": True,
            "cpu_pair_matches_sequential_source": comparison["allclose_rtol_1e_5_atol_1e_6"],
            "all_outputs_finite": bool(np.isfinite(candidate).all() and np.isfinite(reference_output).all()),
            "configuration_validation_input_only": True,
            "formal_90_image_test_not_used": True,
        }
        rows.append({**record, "comparison": comparison, "gates": gates})
        del session
        gc.collect()

    report = {
        "schema": "journal_phase2d_controlled_rcs_pair_build_v1",
        "status": "passed" if all(all(row["gates"].values()) for row in rows) else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_target": "海光 K100 AI 加速卡",
        "source_manifest": identity(manifest_path),
        "benchmark_input": identity(Path(manifest["input"]["path"])),
        "declared_regions": identity(regions_path),
        "operation": "lossless_merge_of_explicit_adjacent_corrected_M5_R_single_block_graphs",
        "selection_role": "configuration-validation structural compile probe only",
        "regions": rows,
        "gates": {
            "all_region_contracts_passed": all(all(row["gates"].values()) for row in rows),
            "corrected_M5_R_reference_used": True,
            "formal_90_image_test_not_used": True,
        },
        "formal_90_image_test_used": False,
    }
    (args.output_root / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" and all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
