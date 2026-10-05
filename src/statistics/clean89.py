#!/usr/bin/env python3
'Post-hoc flood clean-89 sensitivity from scene counts.'

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analysis_utils import load_json, provenance, read_csv, require_file, safe_div, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = require_file(args.config.resolve())
    cfg = load_json(config_path)
    section = cfg["clean89"]
    source = require_file((config_path.parent / section["per_scene_csv"]).resolve())
    identity = require_file((config_path.parent / section["identity_manifest"]).resolve())
    identity_doc = load_json(identity)
    replay = identity_doc["frozen_test_replay"]["first_sample_numeric_replay"]
    if replay["historically_assigned_sample_id"] != section["historical_assigned_id"] or replay["actual_source_sample_id"] != section["actual_source_id"]:
        raise ValueError("registered historical-to-actual identity mapping does not match the manifest")
    exclude_index = int(section["excluded_sample_index"])
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    kept_counts: dict[str, int] = defaultdict(int)
    removed: dict[str, dict[str, str]] = {}
    for row in read_csv(source):
        role = row["variant"]
        if int(row["sample_index"]) == exclude_index:
            removed[role] = {"stored_sample_id": row["sample_id"], "actual_source_sample_id": section["actual_source_id"]}
            continue
        kept_counts[role] += 1
        for field in ["tn", "fp", "fn", "tp", "valid_pixels", "boundary_intersection", "boundary_union", "changed_valid_pixels"]:
            totals[role][field] += int(row[field])
    output: list[dict[str, object]] = []
    for role in sorted(totals):
        t = totals[role]
        if kept_counts[role] != 89:
            raise ValueError(f"{role}: expected 89 retained scenes, found {kept_counts[role]}")
        bg_iou = safe_div(t["tn"], t["tn"] + t["fp"] + t["fn"])
        water_iou = safe_div(t["tp"], t["tp"] + t["fp"] + t["fn"])
        output.append({
            "role": role, "scene_count": 89, "excluded_sample_index": exclude_index,
            "excluded_historical_label": section["historical_assigned_id"],
            "excluded_actual_source": section["actual_source_id"],
            "background_iou": bg_iou, "water_iou": water_iou,
            "miou": None if bg_iou is None or water_iou is None else (bg_iou + water_iou) / 2,
            "boundary_water_iou": safe_div(t["boundary_intersection"], t["boundary_union"]),
            "pixel_accuracy": safe_div(t["tn"] + t["tp"], t["valid_pixels"]),
            "agreement_vs_reference": 1.0 - (t["changed_valid_pixels"] / t["valid_pixels"]),
            "claim_scope": "post-hoc sensitivity from retained counts; not an independent confirmatory evaluation",
        })
    csv_path = args.output_dir / "clean89.csv"
    json_path = args.output_dir / "clean89.json"
    fields = ["role", "scene_count", "excluded_sample_index", "excluded_historical_label", "excluded_actual_source", "background_iou", "water_iou", "miou", "boundary_water_iou", "pixel_accuracy", "agreement_vs_reference", "claim_scope"]
    write_csv(csv_path, fields, output)
    write_json(json_path, {
        "schema": "flood_clean89_posthoc_sensitivity_v1", "status": "DERIVED_FROM_RETAINED_COUNTS",
        "inference_performed": False, "test_input_pack_opened": False,
        "removed_rows": removed, "provenance": provenance(config_path.parents[2], [source, identity], cfg["seed"]),
    })
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
