#!/usr/bin/env python3
'Exploratory margin sensitivity from retained bootstrap bounds.'

from __future__ import annotations

import argparse
from pathlib import Path

from analysis_utils import load_json, provenance, read_csv, require_file, write_csv, write_json


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
    section = cfg["margin_sensitivity"]
    inputs: list[Path] = []
    output: list[dict[str, object]] = []
    for task, relative in section["inputs"].items():
        path = require_file((config_path.parent / relative).resolve())
        inputs.append(path)
        for row in read_csv(path):
            if task == "flood":
                role = row["variant"]
                reference = row["reference"]
                metric = row["metric"]
                point = float(row["observed_difference"])
                lower = float(row["one_sided_95_ci_lower_bound"])
                registered = abs(float(row["noninferiority_margin"]))
            else:
                role = row["role"]
                reference = "Cloud-14S-FP32"
                metric = row["metric"]
                point = float(row["point_difference"])
                lower = float(row["one_sided_lower95"])
                registered = abs(float(row["margin"])) if row.get("margin") else (0.01 if metric.endswith("_IoU") else 0.005)
            for margin in section["margins"]:
                output.append({
                    "task": task, "role": role, "reference": reference, "metric": metric,
                    "point_difference": point, "one_sided_lower95": lower,
                    "exploratory_margin": margin, "pass": lower >= -float(margin),
                    "is_registered_margin": abs(float(margin) - registered) < 1e-12,
                    "interpretation": "post-hoc robustness display; does not replace the registered decision",
                })
    csv_path = args.output_dir / "margin_sensitivity.csv"
    json_path = args.output_dir / "margin_sensitivity.json"
    fields = ["task", "role", "reference", "metric", "point_difference", "one_sided_lower95", "exploratory_margin", "pass", "is_registered_margin", "interpretation"]
    write_csv(csv_path, fields, output)
    write_json(json_path, {
        "schema": "noninferiority_margin_sensitivity_v1", "status": "EXPLORATORY_DERIVED_ANALYSIS",
        "inference_performed": False, "rows": len(output),
        "provenance": provenance(config_path.parents[2], inputs, cfg["seed"]),
    })
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
