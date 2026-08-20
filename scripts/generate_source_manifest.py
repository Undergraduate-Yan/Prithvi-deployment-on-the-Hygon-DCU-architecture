#!/usr/bin/env python3
"""Generate deterministic SHA-256 identities for the Git-sized repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.parts)
        and path.suffix.lower() != ".pyc"
        and path != OUTPUT
    ]
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in sorted(files)]
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} with {len(lines)} entries")


if __name__ == "__main__":
    main()
