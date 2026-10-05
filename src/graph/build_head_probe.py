#!/usr/bin/env python3
'Build a backbone-tail and decoder graph probe.'

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
        int(row["size_bytes"]), str(row["sha256"])
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


def cpu_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return options


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--start-block", type=int, default=23)
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
    start = args.start_block
    if (
        len(reference) != 25
        or not 0 <= start <= 23
        or reference[24]["precision"] != "fp32_compat_barrier"
    ):
        raise RuntimeError("corrected backbone/head precision identity drift")
    paths = [verify(row["model"]) for row in reference]
    backbone = [
        onnx.load(str(paths[index]), load_external_data=False)
        for index in range(start, 24)
    ]
    head = onnx.load(str(paths[24]), load_external_data=False)
    for block in backbone:
        onnx.checker.check_model(block)
    onnx.checker.check_model(head)
    opsets = [
        [(item.domain, item.version) for item in model.opset_import]
        for model in (*backbone, head)
    ]
    if any(row != opsets[0] for row in opsets[1:]):
        raise RuntimeError("backbone/head opset drift")
    for left, right in zip(backbone, backbone[1:], strict=False):
        if [value.name for value in left.graph.output] != [value.name for value in right.graph.input]:
            raise RuntimeError("backbone suffix boundary mismatch")
    block_outputs = [value.name for value in backbone[-1].graph.output]
    head_inputs = [value.name for value in head.graph.input]
    if len(block_outputs) != 1 or block_outputs[0] not in head_inputs:
        raise RuntimeError("final backbone output is not an exact head input")
    internal = block_outputs[0]
    merged_inputs = [copy.deepcopy(value) for value in backbone[0].graph.input] + [
        copy.deepcopy(value) for value in head.graph.input if value.name != internal
    ]
    input_names = [value.name for value in merged_inputs]
    if len(input_names) != len(set(input_names)):
        raise RuntimeError("duplicate merged external inputs")
    initializers = [
        copy.deepcopy(value)
        for model in (*backbone, head)
        for value in model.graph.initializer
    ]
    initializer_names = [value.name for value in initializers]
    if len(initializer_names) != len(set(initializer_names)):
        raise RuntimeError("duplicate backbone-suffix/head initializers")
    graph = helper.make_graph(
        [copy.deepcopy(node) for model in (*backbone, head) for node in model.graph.node],
        f"M5_R_tierH_{start}_head",
        merged_inputs,
        [copy.deepcopy(value) for value in head.graph.output],
        initializer=initializers,
    )
    merged = helper.make_model(graph)
    merged.ir_version = backbone[0].ir_version
    del merged.opset_import[:]
    merged.opset_import.extend(copy.deepcopy(backbone[0].opset_import))
    merged.producer_name = "journal_phase2d_controlled_backbone_suffix_multi_input_head_merge"
    onnx.checker.check_model(merged)

    args.output_root.mkdir(parents=True)
    models = args.output_root / "models"
    feeds_dir = args.output_root / "feeds"
    models.mkdir()
    feeds_dir.mkdir()
    region_id = f"tierH_{start:02d}_head"
    model_path = models / f"{region_id}.onnx"
    onnx.save(merged, str(model_path), save_as_external_data=False)
    reloaded = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(reloaded)

    raw = np.load(verify(manifest["input"]), allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != (1, 6, 224, 224) or not np.isfinite(raw).all():
        raise RuntimeError("configuration-validation input contract drift")
    import onnxruntime as ort
    if ort.__version__ != "1.19.2":
        raise RuntimeError("ONNX Runtime drift")
    values: dict[str, np.ndarray] = {"image": np.ascontiguousarray(raw)}
    for index in range(start):
        session = ort.InferenceSession(
            str(paths[index]), sess_options=cpu_options(ort), providers=["CPUExecutionProvider"]
        )
        names = [item.name for item in session.get_inputs()]
        outputs = [item.name for item in session.get_outputs()]
        result = session.run(outputs, {name: values[name] for name in names})
        values.update({name: np.ascontiguousarray(value) for name, value in zip(outputs, result)})
        del session
        gc.collect()
    feeds = {name: np.ascontiguousarray(values[name]) for name in input_names}
    feed_path = feeds_dir / f"{region_id}.npz"
    np.savez_compressed(feed_path, **feeds)

    for index in range(start, 24):
        block_session = ort.InferenceSession(
            str(paths[index]), sess_options=cpu_options(ort), providers=["CPUExecutionProvider"]
        )
        block_input = block_session.get_inputs()[0].name
        block_output = block_session.get_outputs()[0].name
        values[block_output] = np.ascontiguousarray(
            block_session.run([block_output], {block_input: values[block_input]})[0]
        )
        del block_session
        gc.collect()
    head_session = ort.InferenceSession(
        str(paths[24]), sess_options=cpu_options(ort), providers=["CPUExecutionProvider"]
    )
    head_output_names = [item.name for item in head_session.get_outputs()]
    reference_outputs = [
        np.ascontiguousarray(value)
        for value in head_session.run(
            head_output_names,
            {item.name: values[item.name] for item in head_session.get_inputs()},
        )
    ]
    merged_session = ort.InferenceSession(
        str(model_path), sess_options=cpu_options(ort), providers=["CPUExecutionProvider"]
    )
    merged_output_names = [item.name for item in merged_session.get_outputs()]
    if merged_output_names != head_output_names:
        raise RuntimeError("merged head output interface drift")
    candidate_outputs = [
        np.ascontiguousarray(value)
        for value in merged_session.run(
            merged_output_names,
            {item.name: feeds[item.name] for item in merged_session.get_inputs()},
        )
    ]
    comparisons = {}
    allclose = True
    for name, candidate, reference_output in zip(
        merged_output_names, candidate_outputs, reference_outputs
    ):
        difference = np.abs(candidate.astype(np.float64) - reference_output.astype(np.float64))
        close = bool(np.allclose(candidate, reference_output, rtol=1e-5, atol=1e-6))
        allclose = allclose and close
        comparisons[name] = {
            "mae": float(difference.mean()),
            "max_abs": float(difference.max()),
            "allclose_rtol_1e_5_atol_1e_6": close,
            "candidate": array_record(candidate),
            "sequential_reference": array_record(reference_output),
        }
    gates = {
        "exact_contiguous_backbone_suffix_and_head_edges": True,
        "three_retained_head_inputs_remain_external": len(input_names) == 4,
        "head_output_contract_preserved": merged_output_names == head_output_names,
        "onnx_checker_passed": True,
        "cpu_merged_matches_sequential_backbone_suffix_then_head": allclose,
        "all_outputs_finite": all(np.isfinite(value).all() for value in candidate_outputs + reference_outputs),
        "configuration_validation_input_only": True,
        "formal_90_image_test_not_used": True,
    }
    region = {
        "spec": {
            "id": region_id,
            "role": "merge_backbone_suffix_with_fp32_compatibility_head",
            "start": start,
            "end": 24,
            "expected_precisions": [reference[index]["precision"] for index in range(start, 25)],
        },
        "model": identity(model_path),
        "feed": identity(feed_path),
        "feed_arrays": {name: array_record(value) for name, value in feeds.items()},
        "inventory": inventory(reloaded),
        "inputs": input_names,
        "outputs": merged_output_names,
        "source_models": [identity(paths[index]) for index in range(start, 25)],
        "comparison": comparisons,
        "gates": gates,
    }
    report = {
        "schema": "journal_phase2d_controlled_backbone_suffix_head_build_v2",
        "status": "passed" if all(gates.values()) else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_target": "海光 K100 AI 加速卡",
        "source_manifest": identity(manifest_path),
        "benchmark_input": identity(Path(manifest["input"]["path"])),
        "operation": "lossless_backbone_suffix_internalization_into_multi_input_head",
        "selection_role": "configuration-validation structural compile probe only",
        "regions": [region],
        "gates": {
            "all_region_contracts_passed": all(gates.values()),
            "corrected_M5_R_reference_used": True,
            "formal_90_image_test_not_used": True,
        },
        "formal_90_image_test_used": False,
    }
    (args.output_root / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
