#!/usr/bin/env python3
"""Emit a content fingerprint for the Phase 11 ORT/MIGraphX runtime image."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_fingerprint(distribution) -> dict:
    rows = []
    for item in sorted(distribution.files or [], key=str):
        path = Path(distribution.locate_file(item))
        if path.is_file():
            rows.append((str(item), path.stat().st_size, sha256(path)))
    combined = hashlib.sha256()
    for name, size, digest in rows:
        combined.update(f"{name}\0{size}\0{digest}\n".encode())
    return {
        "name": distribution.metadata.get("Name"),
        "version": distribution.version,
        "file_count": len(rows),
        "total_bytes": sum(row[1] for row in rows),
        "content_manifest_sha256": combined.hexdigest(),
    }


def main() -> None:
    import onnxruntime as ort
    import torch

    distributions = sorted(
        (
            distribution_fingerprint(dist)
            for dist in metadata.distributions()
            if any(token in (dist.metadata.get("Name") or "").lower() for token in ("onnxruntime", "migraphx"))
        ),
        key=lambda row: (row["name"] or "").lower(),
    )
    inventory = sorted(
        (dist.metadata.get("Name") or "", dist.version)
        for dist in metadata.distributions()
    )
    inventory_digest = hashlib.sha256(
        "\n".join(f"{name}=={version}" for name, version in inventory).encode()
    ).hexdigest()
    search_root = Path("/usr/local/lib/python3.10/dist-packages")
    libraries = []
    for pattern in ("**/*onnxruntime*.so*", "**/*migraphx*.so*"):
        for path in sorted(search_root.glob(pattern)):
            if path.is_file():
                libraries.append(
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    unique_libraries = {row["path"]: row for row in libraries}
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "selected_distributions": distributions,
        "installed_distribution_count": len(inventory),
        "installed_distribution_inventory_sha256": inventory_digest,
        "critical_libraries": list(unique_libraries.values()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
