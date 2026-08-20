#!/usr/bin/env python3
"""In-memory static policy tests; these synthetic rows are not evidence artifacts."""
from __future__ import annotations

import combine_phase11_mixed_precision_m0_m5_pareto as target


def row(task: float, size: int, latency: float, vram: int) -> dict:
    return {
        "miou": task,
        "water_iou": task,
        "boundary_water_iou_corrected": task,
        "prediction_agreement_vs_fp32": task,
        "combined_logical_bytes": size,
        "end_to_end_median_ms": latency,
        "incremental_peak_vram_bytes": vram,
    }


def main() -> None:
    better = row(0.9, 10, 2.0, 100)
    worse = row(0.8, 11, 3.0, 101)
    latency_tradeoff = row(0.8, 11, 1.0, 101)
    assert target.dominates(better, worse)
    assert not target.dominates(worse, better)
    assert not target.dominates(better, latency_tradeoff)
    assert not target.dominates(latency_tradeoff, better)

    regrets = target.normalized_regrets({"M1": better, "M2": worse})
    assert regrets["M1"] == 0.0
    assert regrets["M2"] == 1.0

    assignments = target.parse_assignments(
        ["%s=frozen/%s.json" % (candidate, candidate) for candidate in target.CANDIDATES],
        "synthetic static test",
    )
    assert tuple(assignments) == target.CANDIDATES

    try:
        target.parse_assignments(["M0=a", "M0=b"], "synthetic static test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate assignment must fail closed")

    # This is the policy that the full merger additionally enforces against
    # the frozen source: a confirmation is not a replacement eligibility flag.
    original_m0_status = "unstable"
    confirmation_status = "passed"
    original_eligible = original_m0_status == "passed"
    assert confirmation_status == "passed" and not original_eligible
    print("static policy tests: passed")


if __name__ == "__main__":
    main()
