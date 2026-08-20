#!/usr/bin/env python3
"""Capture the origin/target DTK-MIGraphX runtime without making deployment claims."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "phase11_k100_deployment_runtime_fingerprint_v1"
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MGX = "MIGraphXExecutionProvider"
LSMOD_SIZE = 819_664
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def command(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False}
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    return {
        "command": command,
        "available": True,
        "executable": identity(Path(executable)),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_inventory() -> tuple[list[dict], str]:
    rows = []
    all_packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or ""
        all_packages.append((name, distribution.version))
        if any(token in name.lower() for token in ("onnxruntime", "migraphx")):
            files = []
            for item in sorted(distribution.files or [], key=str):
                path = Path(distribution.locate_file(item))
                if path.is_file():
                    files.append((str(item), path.stat().st_size, sha256(path)))
            combined = hashlib.sha256()
            for relative, size, digest in files:
                combined.update(f"{relative}\0{size}\0{digest}\n".encode())
            rows.append(
                {
                    "name": name,
                    "version": distribution.version,
                    "file_count": len(files),
                    "total_bytes": sum(item[1] for item in files),
                    "content_manifest_sha256": combined.hexdigest(),
                }
            )
    inventory = hashlib.sha256(
        "\n".join(f"{name}=={version}" for name, version in sorted(all_packages)).encode()
    ).hexdigest()
    return sorted(rows, key=lambda row: row["name"].lower()), inventory


def critical_libraries() -> list[dict]:
    roots = (Path("/usr/local/lib/python3.10/dist-packages"), Path("/usr/lib"), Path("/opt/rocm/lib"))
    found: dict[str, dict] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("**/*onnxruntime*.so*", "**/*migraphx*.so*"):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    found[str(path)] = identity(path)
    return [found[key] for key in sorted(found)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-image-id", required=True)
    parser.add_argument("--lsmod-shim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IMAGE_PATTERN.fullmatch(args.official_image_id):
        raise RuntimeError("official image ID must be a full sha256 digest")
    output = args.output.resolve(strict=False)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite fingerprint: {output}")

    shim = args.lsmod_shim.resolve(strict=True)
    shim_identity = identity(shim)
    if (shim_identity["size_bytes"], shim_identity["sha256"]) != (LSMOD_SIZE, LSMOD_SHA256):
        raise RuntimeError(f"static lsmod identity drift: {shim_identity}")
    shim_dir = str(shim.parent)
    current_path = os.environ.get("PATH", "")
    entries = current_path.split(os.pathsep) if current_path else []
    os.environ["PATH"] = os.pathsep.join([shim_dir, *[item for item in entries if item != shim_dir]])
    resolved_lsmod = shutil.which("lsmod")
    if resolved_lsmod is None or Path(resolved_lsmod).resolve(strict=True) != shim:
        raise RuntimeError("locked static lsmod is not first on PATH")

    # These imports are deliberately after the locked shim PATH is active.
    import onnxruntime as ort
    import torch

    if ort.__version__ != "1.19.2" or MGX not in ort.get_available_providers():
        raise RuntimeError(
            f"official runtime contract failed: ort={ort.__version__}, providers={ort.get_available_providers()}"
        )
    packages, inventory_digest = package_inventory()
    libraries = critical_libraries()
    portable_fields = {
        "official_image_id": args.official_image_id,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "selected_distributions": packages,
        "installed_distribution_inventory_sha256": inventory_digest,
        "critical_libraries": [
            {key: row[key] for key in ("path", "size_bytes", "sha256")} for row in libraries
        ],
        "static_lsmod": {"size_bytes": LSMOD_SIZE, "sha256": LSMOD_SHA256},
    }
    signature = hashlib.sha256(
        json.dumps(portable_fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attested = os.environ.get("PHASE11_K100_IMAGE_ID")
    if attested is not None and attested != args.official_image_id:
        raise RuntimeError(f"PHASE11_K100_IMAGE_ID drift: {attested}")
    result = {
        "schema": SCHEMA,
        "status": "captured",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": identity(Path(__file__)),
        "portable_runtime_signature_sha256": signature,
        "portable_fields": portable_fields,
        "static_lsmod": {**shim_identity, "path_prepend_active_before_runtime_imports": True},
        "observation": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "available_providers": ort.get_available_providers(),
            "attested_image_id": attested,
            "image_identity_attested": attested == args.official_image_id,
        },
        "commands": {
            "migraphx_driver": command(["migraphx-driver", "--version"]),
            "hipconfig": command(["hipconfig", "--version"]),
            "hy_smi": command(["/usr/local/hyhal/bin/hy-smi", "--showproductname"]),
        },
        "claim_boundary": {
            "same_signature_is_software_compatibility_evidence_not_cache_portability_proof": True,
            "image_digest_is_attested_only_when_host_launcher_sets_PHASE11_K100_IMAGE_ID": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
