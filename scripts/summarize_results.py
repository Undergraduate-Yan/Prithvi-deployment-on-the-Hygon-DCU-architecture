#!/usr/bin/env python3
"""Print the compact paper and deployment result tables without pandas."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    metrics = rows(ROOT / "results/paper_metrics.csv")
    deployment = rows(ROOT / "results/deployment_comparison.csv")
    print("Task/capacity/performance summary")
    print("variant       mIoU      WaterIoU  total_MiB  e2e_ms   strict")
    for row in metrics:
        print(
            f"{row['variant']:<13} {row['miou']:<9} {row['water_iou']:<9} "
            f"{row['total_mib'] or '-':<10} {row['end_to_end_logits_median_ms'] or '-':<8} "
            f"{row['strict_status']}"
        )
    print("\nDeployment CLI summary")
    print("bundle       payload_MiB  median_ms  P95_ms   status")
    for row in deployment:
        print(
            f"{row['bundle']:<12} {row['payload_mib']:<12} {row['median_ms']:<10} "
            f"{row['p95_ms']:<8} {row['deployment_status']}"
        )


if __name__ == "__main__":
    main()
