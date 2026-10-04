#!/usr/bin/env python3
'Research implementation: partition cloud.'
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx


FLOOD_TOPOLOGY_SHA256 = "3d32a0c5ee4aadd2be522ea1fa598182743dd61eb03d7c6c3ed71c0350639835"
BASE_BUILDER_SHA256 = "44943ff7dbd02e917b1cf53a19bdd34df36f316f22e52008fbc02e1110921f4c"
VARIANTS = ("Cloud-RCS13-FP32", "Cloud-RCS13-FP16-Transfer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_base_builder(path: Path) -> Any:
    if sha256_file(path) != BASE_BUILDER_SHA256:
        raise RuntimeError("frozen RCS13 base builder identity mismatch")
    spec = importlib.util.spec_from_file_location("frozen_rcs13_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen RCS13 base builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cloud_name(name: str) -> str:
    if name == "image":
        return "input"
    if name.startswith("/task/model/"):
        return "/model/" + name[len("/task/model/") :]
    return name


def cloud_value(item: dict[str, Any]) -> dict[str, Any]:
    mapped = copy.deepcopy(item)
    mapped["name"] = cloud_name(mapped["name"])
    if mapped["name"] == "logits":
        mapped["shape"] = [1, 4, 224, 224]
    mapped["dtype"] = "float32"
    return mapped


def map_flood_topology(path: Path) -> dict[str, Any]:
    if sha256_file(path) != FLOOD_TOPOLOGY_SHA256:
        raise RuntimeError("frozen flood RCS13 topology identity mismatch")
    flood = json.loads(path.read_text(encoding="utf-8"))
    if flood.get("status") != "passed" or flood.get("session_count") != 13:
        raise RuntimeError("frozen flood topology gate is not passed")
    sessions = []
    for expected_ordinal, row in enumerate(flood["topology"]["sessions"]):
        if int(row["ordinal"]) != expected_ordinal:
            raise RuntimeError("flood topology ordinal drift")
        sessions.append(
            {
                "ordinal": expected_ordinal,
                "session_id": f"Cloud-RCS13_{expected_ordinal:02d}",
                "start_block": row["start_block"],
                "end_block": row["end_block"],
                "logical_role": row["logical_role"],
                "inputs": [cloud_value(item) for item in row["inputs"]],
                "outputs": [cloud_value(item) for item in row["outputs"]],
                "multiscale_tap_outputs": [
                    cloud_name(name) for name in row["multiscale_tap_outputs"]
                ],
            }
        )
    topology = {
        "schema": "phase7f_cloud_rcs13_topology_hypothesis_v1",
        "status": "MAPPED_FROM_FROZEN_FLOOD_BOUNDARIES",
        "session_count": 13,
        "source_flood_topology": identity(path),
        "transfer_scope": "logical Session boundaries only; no flood model, weights, MXR, Head node list, feeds or predictions",
        "external_contract": {
            "input": "FP32[1,6,224,224]",
            "output": "FP32[1,4,224,224] logits",
        },
        "topology": {"sessions": sessions},
        "data_firewall": {
            "calibration_payload_access_count": 0,
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
        },
    }
    if sessions[0]["inputs"] != [
        {"dtype": "float32", "name": "input", "shape": [1, 6, 224, 224]}
    ]:
        raise RuntimeError("cloud first-Session contract drift")
    if not any(item["name"] == "logits" for item in sessions[-1]["outputs"]):
        raise RuntimeError("cloud final logits boundary is absent")
    return topology


def graph_tensor_names(model: onnx.ModelProto) -> set[str]:
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        names.update(value.name for value in collection)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--flood-topology", required=True, type=Path)
    parser.add_argument("--base-builder", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    source = args.source_model.resolve(strict=True)
    if sha256_file(source) != args.source_sha256:
        raise RuntimeError("cloud monolithic source identity drift")
    base = load_base_builder(args.base_builder.resolve(strict=True))
    topology = map_flood_topology(args.flood_topology.resolve(strict=True))

    args.output_root.mkdir(parents=True)
    topology_path = args.output_root / "cloud_rcs13_topology.json"
    write_json(topology_path, topology)
    models_root = args.output_root / "models"
    feeds_root = args.output_root / "compile_feeds"
    models_root.mkdir()
    feeds_root.mkdir()

    print(f"loading {source}", flush=True)
    source_model = onnx.load_model(str(source), load_external_data=True)
    onnx.checker.check_model(source_model, full_check=True)
    required_names = {
        item["name"]
        for row in topology["topology"]["sessions"]
        for item in row["inputs"] + row["outputs"]
    }
    absent = sorted(required_names - graph_tensor_names(source_model))
    if absent:
        raise RuntimeError(f"mapped cloud boundaries absent from source: {absent}")
    base_variant = (
        "RCS13-FP32"
        if args.variant == "Cloud-RCS13-FP32"
        else "RCS13-FP16-A"
    )
    injected = base.inject_required_value_info(source_model, topology, base_variant)
    extractor = onnx.utils.Extractor(source_model)
    model_rows = []
    all_islands = []
    for row in topology["topology"]["sessions"]:
        ordinal = int(row["ordinal"])
        session_id = f"{args.variant.replace('-', '_')}_{ordinal:02d}"
        input_names = [item["name"] for item in row["inputs"]]
        output_names = [item["name"] for item in row["outputs"]]
        print(f"extracting {session_id}: {input_names} -> {output_names}", flush=True)
        extracted = extractor.extract_model(input_names, output_names)
        repairs = base.enforce_fp32_boundaries(extracted, session_id)
        onnx.checker.check_model(extracted, full_check=True)
        model_path = models_root / f"session_{ordinal:02d}.onnx"
        temporary_model = model_path.with_suffix(".onnx.tmp")
        onnx.save_model(extracted, str(temporary_model), save_as_external_data=False)
        reloaded = onnx.load_model(str(temporary_model), load_external_data=False)
        onnx.checker.check_model(reloaded, full_check=True)
        os.replace(temporary_model, model_path)
        feed_path = feeds_root / f"session_{ordinal:02d}.npz"
        feeds = {
            item["name"]: np.zeros(item["shape"], dtype=np.float32)
            for item in row["inputs"]
        }
        np.savez(feed_path, **feeds)
        islands = (
            base.fp32_islands(reloaded, session_id)
            if args.variant == "Cloud-RCS13-FP16-Transfer"
            else []
        )
        all_islands.extend(islands)
        model_rows.append(
            {
                **row,
                "session_id": session_id,
                "model": identity(model_path),
                "compile_feeds": identity(feed_path),
                "boundary_repairs": repairs,
                "graph_stats": base.graph_stats(reloaded),
                "onnx_checker": "PASS_FULL_TWICE",
            }
        )
        del reloaded
        del extracted
    del extractor
    del source_model

    manifest = {
        "schema": "phase7f_cloud_rcs13_variant_manifest_v1",
        "status": "BUILT_ONNX_CHECKED",
        "variant": args.variant,
        "hardware": "海光 K100 AI 加速卡",
        "precision_role": (
            "all_fp32"
            if args.variant == "Cloud-RCS13-FP32"
            else "transferred_fp16_with_new_cloud_head_audit"
        ),
        "source_model": identity(source),
        "source_topology": identity(topology_path),
        "frozen_flood_topology": identity(args.flood_topology),
        "base_builder": identity(args.base_builder),
        "injected_source_value_info": injected,
        "session_count": len(model_rows),
        "external_contract": topology["external_contract"],
        "topology": {"sessions": model_rows},
        "flood_assets_reused": {
            "logical_session_boundaries_only": True,
            "onnx": False,
            "mxr": False,
            "weights": False,
            "head_node_list": False,
            "feeds": False,
            "predictions": False,
        },
        "data_firewall": {
            "calibration_payload_access_count": 0,
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
        },
    }
    manifest_path = args.output_root / "cloud_rcs13_manifest.json"
    write_json(manifest_path, manifest)
    precision_path = args.output_root / "cloud_rcs13_precision_map.json"
    write_json(
        precision_path,
        {
            "schema": "phase7f_cloud_rcs13_precision_map_v1",
            "status": "FROZEN_BUILD_OUTPUT",
            "variant": args.variant,
            "blocks": [
                {
                    "block": block,
                    "precision": (
                        "fp32"
                        if args.variant == "Cloud-RCS13-FP32"
                        else "fp16_compatible"
                    ),
                }
                for block in range(24)
            ],
            "head": (
                "fp32"
                if args.variant == "Cloud-RCS13-FP32"
                else "newly_audited_cloud_fp16_with_fp32_layernorm_statistics"
            ),
            "external_and_inter_session_io": "fp32",
            "formal_test_payload_access_count": 0,
        },
    )
    if all_islands:
        island_path = args.output_root / "cloud_rcs13_fp32_islands.csv"
        fields = tuple(all_islands[0])
        with island_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_islands)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "variant": args.variant,
                "sessions": len(model_rows),
                "boundary_repairs": sum(
                    len(row["boundary_repairs"]) for row in model_rows
                ),
                "fp32_island_rows": len(all_islands),
                "formal_test_payload_access_count": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
