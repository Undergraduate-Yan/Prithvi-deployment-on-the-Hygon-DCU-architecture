#!/usr/bin/env python3
"""Emit a deterministic DTK/MIGraphX runtime fingerprint for formal trials.

The JSON intentionally excludes hostname, PID, wall-clock time, and container ID so
one file identity can be reused to prove that every fresh container used the same
software payload.  Container provenance is locked separately by the launcher and
recorded by each benchmark process.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


SCHEMA = "phase11_mixed_precision_performance_runtime_fingerprint_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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


def command_version(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "command": command,
        "available": True,
        "executable": identity(Path(executable)),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def matching_dpkg_packages() -> list[dict]:
    executable = shutil.which("dpkg-query")
    if executable is None:
        return []
    completed = subprocess.run(
        [executable, "-W", "-f=${Package}\t${Version}\n"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return []
    tokens = ("dtk", "migraphx", "rocm", "hip", "onnxruntime", "hyhal")
    rows = []
    for line in completed.stdout.splitlines():
        if "\t" not in line:
            continue
        name, version = line.split("\t", 1)
        if any(token in name.lower() for token in tokens):
            rows.append({"name": name, "version": version})
    return sorted(rows, key=lambda row: row["name"])


def main() -> None:
    import onnxruntime as ort
    import torch

    distributions = sorted(
        (
            distribution_fingerprint(distribution)
            for distribution in metadata.distributions()
            if any(
                token in (distribution.metadata.get("Name") or "").lower()
                for token in ("onnxruntime", "migraphx")
            )
        ),
        key=lambda row: (row["name"] or "").lower(),
    )
    inventory = sorted(
        (distribution.metadata.get("Name") or "", distribution.version)
        for distribution in metadata.distributions()
    )
    inventory_digest = hashlib.sha256(
        "\n".join(f"{name}=={version}" for name, version in inventory).encode()
    ).hexdigest()

    library_roots = (
        Path("/usr/local/lib/python3.10/dist-packages"),
        Path("/usr/lib"),
        Path("/opt/rocm/lib"),
    )
    libraries: dict[str, dict] = {}
    for root in library_roots:
        if not root.is_dir():
            continue
        for pattern in ("**/*onnxruntime*.so*", "**/*migraphx*.so*"):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    libraries[str(path)] = identity(path)

    result = {
        "schema": SCHEMA,
        "generator": identity(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "selected_distributions": distributions,
        "installed_distribution_count": len(inventory),
        "installed_distribution_inventory_sha256": inventory_digest,
        "matching_dpkg_packages": matching_dpkg_packages(),
        "version_commands": {
            "migraphx_driver": command_version(["migraphx-driver", "--version"]),
            "hipconfig": command_version(["hipconfig", "--version"]),
        },
        "critical_libraries": [libraries[key] for key in sorted(libraries)],
    }
    if ort.__version__ != "1.19.2":
        raise RuntimeError(f"onnxruntime identity drift: {ort.__version__}")
    if "MIGraphXExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("MIGraphXExecutionProvider is unavailable")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
