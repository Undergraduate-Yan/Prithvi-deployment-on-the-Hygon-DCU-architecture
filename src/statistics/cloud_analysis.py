#!/usr/bin/env python3
'Research implementation: cloud analysis.'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLASSES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
MARGINS = {"mIoU": 0.005, "MacroF1": 0.005, "agreement": 0.005, **{f"{name}_IoU": 0.010 for name in CLASSES}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, help="ROLE=RESULT_DIRECTORY")
    parser.add_argument("--reference-role", required=True)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_candidates(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or role in result:
            raise ValueError(f"invalid or duplicate --candidate {value!r}")
        result[role] = Path(raw_path).resolve(strict=True)
    return result


def metric_vector(matrix: np.ndarray) -> dict[str, float]:
    value = matrix.astype(np.float64)
    tp = np.diag(value)
    reference, predicted = value.sum(axis=1), value.sum(axis=0)
    union = reference + predicted - tp
    iou = np.divide(tp, union, out=np.full(4, np.nan), where=union > 0)
    f1 = np.divide(2 * tp, reference + predicted, out=np.full(4, np.nan), where=(reference + predicted) > 0)
    return {
        "mIoU": float(np.nanmean(iou)),
        "MacroF1": float(np.nanmean(f1)),
        "PixelAccuracy": float(tp.sum() / value.sum()),
        **{f"{CLASSES[index]}_IoU": float(iou[index]) for index in range(4)},
    }


def metric_batch(matrix: np.ndarray) -> dict[str, np.ndarray]:
    value = matrix.astype(np.float64)
    tp = np.diagonal(value, axis1=1, axis2=2)
    reference, predicted = value.sum(axis=2), value.sum(axis=1)
    union = reference + predicted - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    f1 = np.divide(2 * tp, reference + predicted, out=np.full_like(tp, np.nan), where=(reference + predicted) > 0)
    return {
        "mIoU": np.nanmean(iou, axis=1),
        "MacroF1": np.nanmean(f1, axis=1),
        **{f"{CLASSES[index]}_IoU": iou[:, index] for index in range(4)},
    }


def read_scene_matrices(directory: Path) -> tuple[dict, np.ndarray, list[dict]]:
    result = json.loads((directory / "formal_candidate_result.json").read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("sample_count") != 300:
        raise RuntimeError(f"candidate formal result is not PASS: {directory}")
    with (directory / "scene_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 300 or [int(row["scene_index"]) for row in rows] != list(range(300)):
        raise RuntimeError(f"scene identity drift: {directory}")
    matrices = np.asarray([json.loads(row["confusion_matrix_json"]) for row in rows], dtype=np.int64)
    return result, matrices, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.candidate)
    if args.reference_role not in candidates:
        raise RuntimeError("reference role is absent")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    with args.records.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 300:
        raise RuntimeError("formal record count drift")

    results, matrices, scene_source = {}, {}, {}
    predictions = {}
    for role, directory in candidates.items():
        results[role], matrices[role], scene_source[role] = read_scene_matrices(directory)
        predictions[role] = directory / "predictions"

    reference = args.reference_role
    reference_metrics = metric_vector(matrices[reference].sum(axis=0))
    metric_rows, per_scene_rows, paired_rows = [], [], []
    agreement_counts = {role: np.zeros(300, dtype=np.int64) for role in candidates}
    valid_counts = np.zeros(300, dtype=np.int64)
    for index in range(300):
        with np.load(args.payload_root / "scene_payloads" / f"{index:03d}.npz", allow_pickle=False) as packed:
            label = np.asarray(packed["label"], dtype=np.uint8)
        valid = label != 255
        valid_counts[index] = int(np.count_nonzero(valid))
        reference_prediction = np.load(predictions[reference] / f"{index:03d}.u8.npy")
        for role in candidates:
            prediction = np.load(predictions[role] / f"{index:03d}.u8.npy")
            agreement_counts[role][index] = int(np.count_nonzero((prediction == reference_prediction) & valid))
            local = metric_vector(matrices[role][index])
            per_scene_rows.append({
                "scene_index": index,
                "datapoint_id": records[index]["datapoint_id"],
                "roi_id": records[index]["roi_id"],
                "region": records[index]["equi_zone"],
                "coverage_bin": records[index]["coverage_bin"],
                "thin_percentage": records[index]["thin_percentage"],
                "cloud_shadow_percentage": records[index]["cloud_shadow_percentage"],
                "role": role,
                **local,
                "agreement_vs_fp32": agreement_counts[role][index] / max(valid_counts[index], 1),
            })

    for role in candidates:
        current = metric_vector(matrices[role].sum(axis=0))
        agreement = int(agreement_counts[role].sum()) / int(valid_counts.sum())
        metric_rows.append({"role": role, **current, "agreement_vs_fp32": agreement, "non_finite_logits": results[role]["non_finite_logits"], "prediction_set_sha256": results[role]["prediction_set_sha256"]})
        if role == reference:
            continue
        for index in range(300):
            ref_local = metric_vector(matrices[reference][index])
            candidate_local = metric_vector(matrices[role][index])
            paired_rows.append({
                "role": role,
                "scene_index": index,
                "datapoint_id": records[index]["datapoint_id"],
                "mIoU_difference": candidate_local["mIoU"] - ref_local["mIoU"],
                "MacroF1_difference": candidate_local["MacroF1"] - ref_local["MacroF1"],
                "thin_cloud_IoU_difference": candidate_local["thin_cloud_IoU"] - ref_local["thin_cloud_IoU"],
                "cloud_shadow_IoU_difference": candidate_local["cloud_shadow_IoU"] - ref_local["cloud_shadow_IoU"],
                "agreement_vs_fp32": agreement_counts[role][index] / max(valid_counts[index], 1),
            })

    write_csv(args.output_root / "cloud_phase7r_formal_metrics.csv", metric_rows)
    write_csv(args.output_root / "cloud_phase7r_per_scene_metrics.csv", per_scene_rows)
    write_csv(args.output_root / "cloud_phase7r_paired_differences.csv", paired_rows)

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, 300, size=(args.bootstrap, 300), endpoint=False)
    reference_boot = metric_batch(matrices[reference][sample_indices].sum(axis=1))
    bootstrap_rows, noninferiority_rows = [], []
    for role in candidates:
        if role == reference:
            continue
        candidate_boot = metric_batch(matrices[role][sample_indices].sum(axis=1))
        point = metric_vector(matrices[role].sum(axis=0))
        for metric in ("mIoU", "MacroF1", *(f"{name}_IoU" for name in CLASSES)):
            delta = candidate_boot[metric] - reference_boot[metric]
            lower, upper, one_sided = (float(np.nanpercentile(delta, value)) for value in (2.5, 97.5, 5.0))
            row = {"role": role, "metric": metric, "point_difference": point[metric] - reference_metrics[metric], "ci95_lower": lower, "ci95_upper": upper, "one_sided_lower95": one_sided, "bootstrap_replicates": args.bootstrap, "seed": args.seed}
            bootstrap_rows.append(row)
            margin = MARGINS[metric]
            noninferiority_rows.append({**row, "margin": -margin, "status": "PASS" if one_sided >= -margin else "FAIL"})
        candidate_agreement = agreement_counts[role][sample_indices].sum(axis=1) / valid_counts[sample_indices].sum(axis=1)
        delta = candidate_agreement - 1.0
        lower, upper, one_sided = (float(np.percentile(delta, value)) for value in (2.5, 97.5, 5.0))
        point_difference = int(agreement_counts[role].sum()) / int(valid_counts.sum()) - 1.0
        row = {"role": role, "metric": "agreement", "point_difference": point_difference, "ci95_lower": lower, "ci95_upper": upper, "one_sided_lower95": one_sided, "bootstrap_replicates": args.bootstrap, "seed": args.seed}
        bootstrap_rows.append(row)
        noninferiority_rows.append({**row, "margin": -MARGINS["agreement"], "status": "PASS" if one_sided >= -MARGINS["agreement"] else "FAIL"})
    write_csv(args.output_root / "cloud_phase7r_bootstrap_summary.csv", bootstrap_rows)
    write_csv(args.output_root / "cloud_phase7r_noninferiority.csv", noninferiority_rows)

    stratified_rows = []
    stratum_sets = {
        "cloud_coverage": [record["coverage_bin"] for record in records],
        "thin_cloud": ["high" if int(record["thin_percentage"]) >= 10 else "low" for record in records],
        "cloud_shadow": ["high" if int(record["cloud_shadow_percentage"]) >= 10 else "low" for record in records],
        "region": [record["equi_zone"] for record in records],
    }
    for kind, labels in stratum_sets.items():
        for label in sorted(set(labels)):
            selected = np.asarray([value == label for value in labels])
            for role in candidates:
                local = metric_vector(matrices[role][selected].sum(axis=0))
                stratified_rows.append({"stratum_type": kind, "stratum": label, "role": role, "scenes": int(selected.sum()), **local})
    write_csv(args.output_root / "cloud_phase7r_stratified_metrics.csv", stratified_rows)

    hard_rows = []
    for role in candidates:
        if role == reference:
            continue
        role_rows = [row for row in paired_rows if row["role"] == role]
        criteria = {
            "lowest_agreement": sorted(role_rows, key=lambda row: row["agreement_vs_fp32"]),
            "largest_thin_cloud_iou_difference": sorted(role_rows, key=lambda row: (np.nan_to_num(abs(float(row["thin_cloud_IoU_difference"])), nan=-1)), reverse=True),
            "largest_cloud_shadow_iou_difference": sorted(role_rows, key=lambda row: (np.nan_to_num(abs(float(row["cloud_shadow_IoU_difference"])), nan=-1)), reverse=True),
            "high_cloud_coverage": sorted(role_rows, key=lambda row: int(records[int(row["scene_index"])]["clear_percentage"])),
        }
        for criterion, ordered in criteria.items():
            for rank, row in enumerate(ordered[:5], 1):
                hard_rows.append({"role": role, "criterion": criterion, "rank": rank, **row})
    write_csv(args.output_root / "cloud_phase7r_hard_scene_selection.csv", hard_rows)

    all_pass = all(row["status"] == "PASS" for row in noninferiority_rows)
    gate = {
        "schema": "cloud_phase7r_formal_gate_v1",
        "status": "PASS" if all_pass else "FAIL_NONINFERIORITY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_role": reference,
        "evaluated_roles": list(candidates),
        "scope": "Analysis of supplied fixed-protocol predictions; access chronology is not inferred by this program",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "all_candidate_noninferiority_checks_pass": all_pass,
        "failed_checks": [row for row in noninferiority_rows if row["status"] != "PASS"],
    }
    (args.output_root / "cloud_phase7r_formal_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "cloud_phase7r_formal_report.md").write_text(
        f"# Cloud segmentation evaluation\n\nStatus: `{gate['status']}`. Configurations: {', '.join(candidates)}. The analysis contains 300 paired scenes and uses {args.bootstrap:,} bootstrap replicates with seed {args.seed}. Interpretation of scene independence follows the study data-selection and access records.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": gate["status"], "roles": list(candidates), "failed_checks": len(gate["failed_checks"])}, ensure_ascii=False))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
