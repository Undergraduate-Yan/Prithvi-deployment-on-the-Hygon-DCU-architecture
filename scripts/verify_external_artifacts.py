#!/usr/bin/env python3
"""Verify external model artifacts against the public identity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes root: {relative}") from exc
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="External artifact root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/required_artifacts.example.json"),
    )
    parser.add_argument("--require-optional", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    verified = 0
    skipped = 0

    for row in manifest["artifacts"]:
        path = contained(root, row["path"])
        required = bool(row.get("required", True)) or args.require_optional
        if not path.is_file():
            if required:
                failures.append(f"missing: {row['name']} -> {path}")
            else:
                skipped += 1
            continue
        expected_size = row.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            failures.append(
                f"size mismatch: {row['name']} expected={expected_size} "
                f"actual={path.stat().st_size}"
            )
            continue
        actual_sha = sha256_file(path)
        if actual_sha.lower() != row["sha256"].lower():
            failures.append(
                f"sha256 mismatch: {row['name']} expected={row['sha256']} actual={actual_sha}"
            )
            continue
        verified += 1
        print(f"OK  {row['name']}: {path}")

    print(f"verified={verified} skipped_optional={skipped} failures={len(failures)}")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
