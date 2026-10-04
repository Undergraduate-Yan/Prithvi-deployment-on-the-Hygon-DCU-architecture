#!/usr/bin/env python3
'Research implementation: aggregate blocks.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np


MEASURES = (
    "L_mae",
    "L_max_abs",
    "L_relative_l2",
    "C_mae",
    "C_max_abs",
    "C_relative_l2",
    "P_mae",
    "P_max_abs",
    "P_relative_l2",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "journal_stage1_input_pack_v1" or manifest.get("status") != "passed":
        raise ValueError("input manifest is not a passed Stage-1 input pack")
    gates = manifest.get("leakage_gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("not every Stage-1 leakage gate is true")
    expected_ids = manifest.get("configuration_validation", {}).get("sample_ids", [])
    if len(expected_ids) != 64:
        raise ValueError("expected exactly 64 locked configuration-validation IDs")

    grouped: DefaultDict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    samples: DefaultDict[int, set] = defaultdict(set)
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "block_index", *MEASURES}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("input is missing columns: %s" % sorted(missing))
        observed_pairs = set()
        row_count = 0
        for row in reader:
            block = int(row["block_index"])
            if block not in range(24):
                raise ValueError("block_index must be in 0..23")
            pair = (row["sample_id"], block)
            if pair in observed_pairs:
                raise ValueError("duplicate sample_id/block_index row: %s" % (pair,))
            observed_pairs.add(pair)
            row_count += 1
            samples[block].add(row["sample_id"])
            for measure in MEASURES:
                value = float(row[measure])
                if not np.isfinite(value):
                    raise ValueError("non-finite %s for %s" % (measure, pair))
                grouped[block][measure].append(value)

    expected_pairs = {(sample_id, block) for sample_id in expected_ids for block in range(24)}
    if row_count != 1536 or observed_pairs != expected_pairs:
        raise ValueError("diagnostic CSV is not the exact locked 64x24 grid")

    fields = ["block_index", "scene_count"]
    for measure in MEASURES:
        fields.extend(
            [
                measure + "_median",
                measure + "_mean",
                measure + "_p75",
                measure + "_p90",
                measure + "_max",
            ]
        )
    rows = []
    for block in sorted(grouped):
        output = {"block_index": block, "scene_count": len(samples[block])}
        for measure in MEASURES:
            values = np.asarray(grouped[block][measure], dtype=np.float64)
            if values.size == 0:
                raise ValueError("block %d has no finite %s values" % (block, measure))
            output.update(
                {
                    measure + "_median": float(np.median(values)),
                    measure + "_mean": float(values.mean()),
                    measure + "_p75": float(np.percentile(values, 75)),
                    measure + "_p90": float(np.percentile(values, 90)),
                    measure + "_max": float(values.max()),
                }
            )
        rows.append(output)
    if len(rows) != 24 or any(row["scene_count"] != 64 for row in rows):
        raise ValueError("aggregate must contain 24 blocks with 64 scenes each")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if args.plot is not None:
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        for axis, measure in zip(axes.ravel(), MEASURES):
            for block in range(24):
                values = np.asarray(grouped[block][measure], dtype=np.float64)
                axis.scatter(np.full(values.shape, block), values, s=4, alpha=0.25)
            axis.set_title(measure)
            axis.set_xlabel("block index")
            axis.set_ylabel("per-scene error")
            axis.grid(alpha=0.2)
        fig.savefig(args.plot, dpi=180)
        plt.close(fig)

    if args.report is not None:
        def file_identity(path: Path) -> dict:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}

        report = {
            "schema": "journal_stage1_aggregated_block_diagnostics_v1",
            "status": "passed",
            "selection_split": "configuration-validation only",
            "sample_count": 64,
            "block_count": 24,
            "raw_row_count": 1536,
            "measures": list(MEASURES),
            "statistics": ["median", "mean", "p75", "p90", "max"],
            "gates": {
                "exact_locked_64x24_grid": True,
                "all_values_finite": True,
                "formal_test_not_used": True,
            },
            "input": file_identity(args.input),
            "input_manifest": file_identity(args.input_manifest),
            "output": file_identity(args.output),
        }
        if args.plot is not None:
            report["distribution_plot"] = file_identity(args.plot)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
