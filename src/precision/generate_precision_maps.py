#!/usr/bin/env python3
'Research implementation: generate precision maps.'

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List

from common import identity, write_json


BLOCKS = list(range(24))


def payload(
    name: str,
    fp16_blocks: List[int],
    rule: str,
    seed: int = 42,
    *,
    status: str = "defined_not_executed",
    evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    fp16 = sorted(fp16_blocks)
    result = {
        "schema": "journal_precision_map_v1",
        "name": name,
        "status": status,
        "seed": seed,
        "selection_source": rule,
        "external_io": {"input": "FP32[1,6,224,224]", "output": "FP32[1,2,224,224]"},
        "backbone": {
            "fp16_blocks": fp16,
            "int8_blocks": [block for block in BLOCKS if block not in fp16],
        },
        "head_precision": "FP32",
        "acceptance": {
            "configuration_selection_split": "configuration-validation only",
            "formal_test_split_may_not_select_configuration": True,
        },
    }
    if evidence is not None:
        result["selection_evidence"] = evidence
    return result


def revised_maps(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"block_index", "scene_count", "L_relative_l2_p90", "L_mae_p90", "L_max_abs_p90"}
    if len(rows) != 24 or any(required.difference(row) for row in rows):
        raise ValueError("aggregated diagnostics must contain the exact 24-block selection table")
    parsed = []
    for row in rows:
        block = int(row["block_index"])
        if block not in BLOCKS or int(row["scene_count"]) != 64:
            raise ValueError("aggregated diagnostics block/count drift")
        parsed.append(
            (
                block,
                float(row["L_relative_l2_p90"]),
                float(row["L_mae_p90"]),
                float(row["L_max_abs_p90"]),
            )
        )
    if {item[0] for item in parsed} != set(BLOCKS):
        raise ValueError("aggregated diagnostics block grid is incomplete or duplicated")
    ranked = sorted(parsed, key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    ordered_blocks = [item[0] for item in ranked]
    evidence = {
        "aggregated_diagnostics": identity(path),
        "configuration_validation_scene_count": 64,
        "ranking": ordered_blocks,
        "primary_score": "descending L_relative_l2_p90",
        "tie_breakers": ["descending L_mae_p90", "descending L_max_abs_p90", "ascending block_index"],
        "rationale": (
            "P90 local same-input relative-L2 isolates robust per-block CPU-versus-MIGraphX "
            "backend drift without ranking later blocks merely because they inherit upstream error. "
            "C and P remain reported as propagation diagnostics."
        ),
        "formal_test_used_for_selection": False,
    }
    cardinalities = {"M0-R": 0, "M1-R": 1, "M2-R": 2, "M3-R": 3, "M4-R": 4, "M5-R": 9}
    return [
        payload(
            name,
            ordered_blocks[:count],
            "locked 64-scene configuration-validation P90 local relative-L2 ranking",
            status="selected_not_task_evaluated",
            evidence={**evidence, "selected_prefix_length": count},
        )
        for name, count in cardinalities.items()
    ]


def blocked(name: str, reason: str) -> Dict[str, Any]:
    return {
        "schema": "journal_precision_map_v1",
        "name": name,
        "status": "BLOCKED",
        "reason": reason,
        "backbone": {"fp16_blocks": [], "int8_blocks": []},
        "head_precision": "FP32",
        "claim_boundary": "No precision map is selected until the named evidence exists.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--aggregated-diagnostics", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    maps = [
        payload("Legacy-M5", [0] + list(range(14, 22)), "frozen legacy single-test-image diagnosis"),
        payload("Early-9", list(range(9)), "equal-resource positional control"),
        payload("Late-9", list(range(15, 24)), "equal-resource positional control"),
    ]
    if args.aggregated_diagnostics is not None:
        maps.extend(revised_maps(args.aggregated_diagnostics.resolve(strict=True)))
    for seed in (42, 43, 44, 45, 46):
        rng = random.Random(seed)
        maps.append(payload("Random-9-seed-%d" % seed, rng.sample(BLOCKS, 9), "uniform random control", seed))
    maps.extend(
        [
            blocked(
                "Sensitivity-9",
                "requires task-loss/metric deltas on the independent configuration-validation split",
            ),
            blocked(
                "Runtime-9",
                "requires per-block net K100 latency benefits on the independent configuration-validation split",
            ),
        ]
    )
    if args.aggregated_diagnostics is None:
        maps.append(
            blocked(
                "M5-R",
                "requires aggregated multi-scene backend diagnostics from configuration-validation scenes",
            )
        )
    for item in maps:
        safe_name = str(item["name"]).replace("/", "_")
        write_json(args.output_dir / (safe_name + ".json"), item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
