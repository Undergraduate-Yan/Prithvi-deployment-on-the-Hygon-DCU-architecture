#!/usr/bin/env python3
'Parse measured MIGraphX compilation pass timings.'

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PATTERNS = (
    re.compile(r"^\s*(?P<pass>[A-Za-z_][A-Za-z0-9_:.+/-]*)\s*:\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|us|s)\b", re.I),
    re.compile(r"(?P<pass>[A-Za-z0-9_:.+/-]*pass[A-Za-z0-9_:.+/-]*)[^\n]*?(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|us|s)\b", re.I),
    re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|us|s)\s*[:,-]\s*(?P<pass>[A-Za-z0-9_:.+/-]+)", re.I),
)


def milliseconds(value: float, unit: str) -> float:
    return value * {"us": 0.001, "ms": 1.0, "s": 1000.0}[unit.lower()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.logs_root.rglob("*.log")):
        config_id = next((part for part in path.parts if re.fullmatch(r"C[0-5](?:R)?", part)), "unknown")
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for pattern in PATTERNS:
                match = pattern.search(line)
                if match:
                    value = float(match.group("value"))
                    unit = match.group("unit").lower()
                    rows.append(
                        {
                            "config_id": config_id,
                            "log_path": str(path),
                            "line_number": line_number,
                            "pass": match.group("pass"),
                            "duration_ms": milliseconds(value, unit),
                            "raw_line": line[:2000],
                        }
                    )
                    break
    fields = ["config_id", "log_path", "line_number", "pass", "duration_ms", "raw_line"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"parsed_pass_time_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
