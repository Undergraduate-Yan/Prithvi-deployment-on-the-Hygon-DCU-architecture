"""Recompute selected revision statistics from the published derived records."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


GIB = 2**30
PRECISIONS = ("fp32", "fp16")
ROUNDS = set(range(1, 6))


def read_csv(directory, name):
    with (directory / name).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def check(label, actual, expected):
    if isinstance(actual, bool):
        matches = str(actual).lower() == str(expected).lower()
    elif isinstance(actual, str):
        matches = actual == expected
    else:
        matches = math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12)
    if not matches:
        raise ValueError(f"{label}: computed {actual!r}, retained {expected!r}")


def check_row(label, measured, recorded):
    for column, value in measured.items():
        check(f"{label}.{column}", value, recorded[column])
    return len(measured)


def values(rows, column, scale=1):
    return np.array([float(row[column]) / scale for row in rows])


def index_rounds(rows, label):
    indexed = {int(float(row["round"])): row for row in rows}
    if len(rows) != len(ROUNDS) or set(indexed) != ROUNDS:
        raise ValueError(f"{label}: expected one record for each round 1–5")
    return indexed


def check_flood_system(records, table):
    medians = {}
    count = 0
    for segments in ("25", "13"):
        selected = [row for row in records if row["segments"] == segments]
        rounds = index_rounds(selected, f"flood {segments} segments")
        for row in selected:
            check("flood calls per process", int(row["calls"]), 200)
        medians[segments] = [float(rounds[index]["median_ms"]) for index in sorted(rounds)]
        metrics = {
            "processes": len(selected),
            "calls": sum(int(row["calls"]) for row in selected),
            "median_process_peak_host_RSS_GiB": np.median(values(selected, "process_peak_host_RSS_GiB")),
            "maximum_sampled_device_GiB": np.max(values(selected, "device_sampled_peak_GiB")),
        }
        count += check_row(f"flood {segments} segments", metrics, table[segments])
    return count, np.array(medians["25"]) - np.array(medians["13"])


def check_cloud_system(processes, scenes, resource_table, scene_table, repeat_table):
    comparisons = defaultdict(dict)
    count = 0
    for precision in PRECISIONS:
        process_rows = [row for row in processes if row["precision"] == precision]
        rounds = index_rounds(process_rows, f"cloud {precision}")
        scene_rows = [row for row in scenes if row["precision"] == precision]
        scenes_by_id = defaultdict(list)
        scenes_by_round = defaultdict(list)
        for row in scene_rows:
            scenes_by_id[int(float(row["scene_index"]))].append(row)
            scenes_by_round[int(float(row["round"]))].append(row)
            check("cloud tile calls per scene", int(float(row["tile_calls"])), 81)
        if len(scenes_by_id) != 10 or set(scenes_by_round) != ROUNDS:
            raise ValueError(f"cloud {precision}: expected ten scenes in each of five rounds")
        for scene_id, repeats in scenes_by_id.items():
            if len(repeats) != len(ROUNDS) or {int(float(row["round"])) for row in repeats} != ROUNDS:
                raise ValueError(f"cloud {precision} scene {scene_id}: expected one record per round")
            unique = len({row["product_sha256"] for row in repeats})
            count += check_row(
                f"cloud {precision} scene {scene_id}",
                {"runs": len(repeats), "unique_product_hashes": unique, "stable": unique == 1},
                repeat_table[(precision, scene_id)],
            )

        idle_energy = values(process_rows, "tile_above_idle_device_energy_j_per_call")
        increment = values(process_rows, "tile_device_peak_increment_bytes", GIB)
        host_peak = np.max(values(process_rows, "host_VmHWM_bytes", GIB))
        resource_metrics = {
            "processes": len(process_rows),
            "tile_calls": int(np.sum(values(process_rows, "tile_calls"))),
            "median_tile_gross_device_J": np.median(values(process_rows, "tile_gross_device_energy_j_per_call")),
            "median_tile_above_idle_device_J": np.median(idle_energy),
            "median_incremental_device_GiB": np.median(increment),
            "median_sampled_absolute_device_GiB": np.median(values(process_rows, "tile_device_peak_bytes", GIB)),
            "maximum_host_VmHWM_GiB": host_peak,
            "minimum_tile_above_idle_device_J": np.min(idle_energy),
            "maximum_tile_above_idle_device_J": np.max(idle_energy),
            "minimum_incremental_device_GiB": np.min(increment),
            "maximum_incremental_device_GiB": np.max(increment),
        }
        count += check_row(f"cloud {precision} resources", resource_metrics, resource_table[precision])

        total = values(scene_rows, "total_s")
        scene_metrics = {
            "scenes": len(scenes_by_id),
            "processes": len(scenes_by_round),
            "measured_scene_calls": len(scene_rows),
            "median_scene_s": np.median(total),
            "p95_scene_s_descriptive": np.quantile(total, 0.95),
            "mean_scene_s": np.mean(total),
            "sampled_peak_device_GiB": np.max(values(scene_rows, "device_peak_bytes", GIB)),
            "maximum_process_VmHWM_GiB": host_peak,
            "mean_scenes_per_second": len(scene_rows) / np.sum(total),
        }
        for phase in ("read_s", "normalize_s", "tile_processing_s", "stitch_s", "product_s",
                      "gross_device_energy_j", "above_idle_device_energy_j"):
            scene_metrics[f"mean_{phase}"] = np.mean(values(scene_rows, phase))
        count += check_row(f"cloud {precision} full scenes", scene_metrics, scene_table[precision])

        comparisons["cloud_fp32_minus_fp16_process_mean_scene_s"][precision] = [
            np.mean(values(scenes_by_round[index], "total_s")) for index in sorted(rounds)
        ]
        comparisons["cloud_fp32_minus_fp16_process_mean_above_idle_J_per_tile"][precision] = [
            float(rounds[index]["tile_above_idle_device_energy_j_per_call"]) for index in sorted(rounds)
        ]
        comparisons["cloud_fp32_minus_fp16_process_mean_device_increment_GiB"][precision] = [
            float(rounds[index]["tile_device_peak_increment_bytes"]) / GIB for index in sorted(rounds)
        ]

    differences = {
        f"{metric}_95CI": np.array(data["fp32"]) - np.array(data["fp16"])
        for metric, data in comparisons.items()
    }
    return count, differences


def verify_system(directory):
    flood_count, flood_diff = check_flood_system(
        read_csv(directory, "flood_matched_processes.csv"),
        {row["segments"]: row for row in read_csv(directory, "flood_matched_topology_table.csv")},
    )
    cloud_count, comparisons = check_cloud_system(
        read_csv(directory, "cloud_resource_processes.csv"),
        read_csv(directory, "cloud_full_scene_records.csv"),
        {row["precision"]: row for row in read_csv(directory, "cloud_same_node_resource_table.csv")},
        {row["precision"]: row for row in read_csv(directory, "cloud_full_scene_table.csv")},
        {
            (row["precision"], int(row["scene_index"])): row
            for row in read_csv(directory, "full_scene_repeat_integrity.csv")
        },
    )
    comparisons["flood_25_minus13_process_median_ms_95CI"] = flood_diff
    with (directory / "process_uncertainty.json").open(encoding="utf-8-sig") as file:
        uncertainty = json.load(file)
    check("bootstrap seed", uncertainty["seed"], 42)
    check("bootstrap replicates", uncertainty["replicates"], 10_000)

    # Identical paired-round draws for all endpoints, as in the retained analysis.
    draws = np.random.default_rng(42).integers(0, len(ROUNDS), (10_000, len(ROUNDS)))
    for label, differences in comparisons.items():
        lower, upper = np.quantile(np.mean(differences[draws], axis=1), [0.025, 0.975])
        check(f"{label} lower bound", lower, uncertainty[label][0])
        check(f"{label} upper bound", upper, uncertainty[label][1])
    print(f"Verified {flood_count + cloud_count} system table cells and {len(comparisons)} paired-process intervals.")
    print("Pooled flood latency percentiles require the per-call traces, which this check does not read.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir", type=Path, default=Path(__file__).resolve().parent.parent / "evidence",
        help="directory containing the published flood/ and system/ records",
    )
    directory = parser.parse_args().evidence_dir.resolve()
    subprocess.run([sys.executable, str(directory / "flood" / "recompute_event_intervals.py")], check=True)
    verify_system(directory / "system")


if __name__ == "__main__":
    main()
