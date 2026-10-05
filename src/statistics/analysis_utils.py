'Deterministic utilities for analyses of retained research records.'

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance(repo: Path, inputs: list[Path], seed: int | None = None) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(repo),
        "python": sys.version.split()[0],
        "seed": seed,
        "allow_heldout_access": os.environ.get("ALLOW_HELDOUT_ACCESS", "NO"),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in inputs
        ],
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def bootstrap_median_ci(values: list[float], seed: int, resamples: int = 10000) -> tuple[float, float]:
    if not values:
        raise ValueError("no values supplied")
    rng = random.Random(seed)
    boot = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator

