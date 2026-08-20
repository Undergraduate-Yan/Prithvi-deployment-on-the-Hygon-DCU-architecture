#!/usr/bin/env python3
"""Fail-closed checks for files intended for a public Git repository."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_FILE = 50 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".ckpt", ".mxr", ".onnx", ".npy", ".npz", ".pt", ".pth", ".tar", ".tgz", ".whl"
}
REQUIRED = {
    "README.md", "README_zh.md", ".gitignore", "pyproject.toml",
    "results/paper_metrics.csv", "results/deployment_comparison.csv",
    "artifacts/required_artifacts.example.json",
    "docs/reproduction.md", "docs/evidence_boundaries.md",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "SSH password assignment": re.compile(
        r"(?i)\b(?:ssh_password|ssh_pass|login_password)\b\s*[:=]\s*['\"][^'\"]+"
    ),
}
TEXT_SUFFIXES = {
    ".c", ".csv", ".env", ".json", ".md", ".py", ".sh", ".tex", ".toml", ".txt", ".yml", ".yaml"
}


def tracked_candidates() -> list[Path]:
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    files = tracked_candidates()

    for relative in sorted(REQUIRED):
        if not (ROOT / relative).is_file():
            failures.append(f"required file missing: {relative}")

    for path in files:
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_GIT_FILE:
            failures.append(f"file exceeds 50 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"binary artifact must stay external: {relative}")

        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"invalid JSON: {relative}: {exc}")
        elif path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"invalid Python syntax: {relative}: {exc}")

        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith(".env"):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                warnings.append(f"non-UTF-8 text candidate: {relative}")
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"possible {label}: {relative}")

    print(f"root={ROOT}")
    print(f"files={len(files)} bytes={sum(p.stat().st_size for p in files)}")
    for warning in warnings:
        print(f"WARN {warning}")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"repository check failed: {len(failures)} issue(s)")
        return 1
    print("repository check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
