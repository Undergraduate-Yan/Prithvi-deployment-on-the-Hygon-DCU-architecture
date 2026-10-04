#!/usr/bin/env python3
'Research implementation: collect compile memory.'

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def process_table(proc_root: Path = Path("/proc")) -> dict[int, tuple[int, int]]:
    """Return pid -> (ppid, rss_bytes) for processes readable by this user."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    table: dict[int, tuple[int, int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # The comm field may contain spaces and parentheses; split after its last ')'.
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            ppid = int(fields[1])
            rss_pages = int(fields[21])
            table[int(entry.name)] = (ppid, rss_pages * page_size)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, IndexError, ValueError):
            continue
    return table


def aggregate_tree(root_pid: int, table: dict[int, tuple[int, int]]) -> tuple[int, int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, ()))
    live = [pid for pid in seen if pid in table]
    return len(live), sum(table[pid][1] for pid in live)


def sample(root_pid: int) -> dict[str, object]:
    table = process_table()
    count, rss = aggregate_tree(root_pid, table)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root_pid": root_pid,
        "process_count": count,
        "process_tree_rss_bytes": rss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    fields = ["timestamp_utc", "root_pid", "process_count", "process_tree_rss_bytes"]
    if args.once:
        row = sample(args.root_pid)
        print("\t".join(str(row[name]) for name in fields))
        return 0
    if args.output is None:
        parser.error("--output is required unless --once is used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while True:
            row = sample(args.root_pid)
            writer.writerow(row)
            stream.flush()
            if row["process_count"] == 0:
                break
            time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
