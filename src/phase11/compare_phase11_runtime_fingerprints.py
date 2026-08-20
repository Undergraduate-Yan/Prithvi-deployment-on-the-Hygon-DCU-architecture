#!/usr/bin/env python3
"""Compare critical runtime content across two independently built K100 images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORIGIN = (
    2_879,
    "92b8e6f53ef6ed869d5b3d212a5f35af36c4f7bc6ecfd54700df8a956b5f8db7",
)
TARGET = (
    2_879,
    "81e78ae6ab3fb97e2850ac87a3682645bb7747a020efcbeb3d453922104a9e81",
)
ORIGIN_IMAGE = "sha256:97f1889c21c32f1798bf701c4fda80408ab2b387904b01e76463d521eaf18291"
TARGET_IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
ORIGIN_LAYER_MANIFEST_SHA256 = "a48dfce1e93c26fea8b46b84c588de891d31369a5de2b4ce8b8593773e2b1bf6"
TARGET_LAYER_MANIFEST_SHA256 = "8a2552348736202de6e00eb4dbbfe13cee9a44e32f298bb4301891cefaa3974b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock(path: Path, expected: tuple[int, str]) -> dict:
    path = path.resolve(strict=True)
    row = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if (row["size_bytes"], row["sha256"]) != expected:
        raise RuntimeError(f"fingerprint identity drift: {row}")
    return row


def selected(value: dict) -> dict:
    keys = (
        "onnxruntime",
        "available_providers",
        "torch",
        "torch_hip",
        "selected_distributions",
        "installed_distribution_inventory_sha256",
        "critical_libraries",
    )
    return {key: value[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identities = {"origin": lock(args.origin, ORIGIN), "target": lock(args.target, TARGET)}
    origin = json.loads(args.origin.read_text(encoding="utf-8"))
    target = json.loads(args.target.read_text(encoding="utf-8"))
    origin_selected = selected(origin)
    target_selected = selected(target)
    content_match = origin_selected == target_selected
    selected_digest = hashlib.sha256(
        json.dumps(origin_selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "status": "critical_runtime_content_match" if content_match else "failed",
        "identities": identities,
        "images": {
            "origin": ORIGIN_IMAGE,
            "target": TARGET_IMAGE,
            "exact_image_identity_match": ORIGIN_IMAGE == TARGET_IMAGE,
            "origin_layer_manifest_sha256": ORIGIN_LAYER_MANIFEST_SHA256,
            "target_layer_manifest_sha256": TARGET_LAYER_MANIFEST_SHA256,
            "exact_layer_manifest_match": ORIGIN_LAYER_MANIFEST_SHA256 == TARGET_LAYER_MANIFEST_SHA256,
        },
        "selected_runtime_content_sha256": selected_digest,
        "selected_runtime_content_match": content_match,
        "selected_runtime": origin_selected if content_match else None,
        "claims": {
            "critical_ort_migraphx_runtime_content_match": content_match,
            "exact_image_identity_match": False,
            "cache_portability_verified": False,
            "cross_node_performance_verified": False,
        },
        "evidence_boundary": (
            "The image IDs and final layer manifests differ. The installed package inventory, "
            "ORT/MIGraphX distribution content, critical shared libraries, providers, Torch and HIP "
            "content match exactly; this supports a matched-runtime rebuild claim, not an exact-image claim."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if content_match else 2)


if __name__ == "__main__":
    main()
