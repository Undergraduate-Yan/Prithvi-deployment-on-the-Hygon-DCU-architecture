#!/usr/bin/env python3
'Cloud class and binary metrics from scene confusion matrices.'

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis_utils import load_json, provenance, read_csv, require_file, safe_div, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON analysis configuration")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def metrics_from_matrix(matrix: list[list[int]], labels: list[str], role: str) -> list[dict[str, object]]:
    total = sum(sum(row) for row in matrix)
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(row[index] for row in matrix) - tp
        tn = total - tp - fn - fp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        iou = safe_div(tp, tp + fp + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        rows.append({
            "role": role, "aggregation": "four_class", "class": label,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "support": tp + fn,
            "prevalence": safe_div(tp + fn, total), "precision": precision,
            "recall": recall, "f1": f1, "iou": iou, "pixel_accuracy": safe_div(tp + tn, total),
        })
    clear_tp = matrix[0][0]
    clear_to_cloud = sum(matrix[0][1:])
    cloud_to_clear = sum(row[0] for row in matrix[1:])
    cloud_to_cloud = sum(sum(row[1:]) for row in matrix[1:])
    # Positive class is any cloud (thick, thin, or shadow).  The canonical
    # source-matrix order is Clear, Thick Cloud, Thin Cloud, Cloud Shadow.
    tp, fn, fp, tn = cloud_to_cloud, cloud_to_clear, clear_to_cloud, clear_tp
    rows.append({
        "role": role, "aggregation": "clear_vs_any_cloud", "class": "Any Cloud",
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "support": tp + fn,
        "prevalence": safe_div(tp + fn, total), "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn), "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn), "pixel_accuracy": safe_div(tp + tn, total),
    })
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = require_file(args.config.resolve())
    cfg = load_json(config_path)
    section = cfg["cloud_confusion"]
    labels = section["classes"]
    if len(labels) != 4:
        raise ValueError("this registered analysis requires exactly four cloud classes")
    inputs: list[Path] = []
    results: list[dict[str, object]] = []
    matrices: dict[str, list[list[int]]] = {}
    for role, relative in section["inputs"].items():
        path = require_file((config_path.parent / relative).resolve())
        inputs.append(path)
        matrix = [[0] * 4 for _ in range(4)]
        rows = read_csv(path)
        if len(rows) != 300:
            raise ValueError(f"{role}: expected 300 frozen scene summaries, found {len(rows)}")
        for row in rows:
            current = json.loads(row["confusion_matrix_json"])
            if len(current) != 4 or any(len(item) != 4 for item in current):
                raise ValueError(f"{role}: malformed 4x4 confusion matrix")
            for i in range(4):
                for j in range(4):
                    matrix[i][j] += int(current[i][j])
        matrices[role] = matrix
        results.extend(metrics_from_matrix(matrix, labels, role))
    output_csv = args.output_dir / "cloud_confusion.csv"
    output_json = args.output_dir / "cloud_confusion.json"
    fields = ["role", "aggregation", "class", "tn", "fp", "fn", "tp", "support", "prevalence", "precision", "recall", "f1", "iou", "pixel_accuracy"]
    write_csv(output_csv, fields, results)
    write_json(output_json, {
        "schema": "cloud_confusion_analysis_v1", "status": "DERIVED_FROM_FROZEN_SUMMARIES",
        "inference_performed": False, "formal_payload_opened": False,
        "matrix_orientation": "rows=true class, columns=predicted class", "classes": labels,
        "matrices": matrices, "provenance": provenance(config_path.parents[2], inputs, cfg["seed"]),
    })
    print(output_csv)
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
