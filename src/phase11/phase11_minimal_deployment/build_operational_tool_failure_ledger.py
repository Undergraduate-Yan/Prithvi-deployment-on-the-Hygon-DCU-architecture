#!/usr/bin/env python3
"""Lock the known launcher write-permission failure without promoting it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "phase11_k100_operational_tool_failure_exclusion_v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular non-symlink file: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def tree_identity(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("failed launcher artifact must be a non-symlink directory")
    rows = []
    total = 0
    for parent, names, files in os.walk(directory, followlinks=False):
        parent_path = Path(parent)
        for name in names:
            if (parent_path / name).is_symlink():
                raise RuntimeError("failed launcher tree contains a directory symlink")
        for name in files:
            path = parent_path / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("failed launcher tree contains a non-regular file")
            row = {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            rows.append(row)
            total += row["size_bytes"]
    rows.sort(key=lambda row: row["path"])
    return {
        "file_count": len(rows),
        "total_bytes": total,
        "tree_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operational-root", type=Path, required=True)
    parser.add_argument("--failed-launcher-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", choices=("M5", "FP16"), required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--official-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_raw = args.operational_root.absolute()
    if root_raw.is_symlink():
        raise RuntimeError("operational root must not be a symlink")
    root = root_raw.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("operational root must be a directory")
    failed_raw = args.failed_launcher_dir.absolute()
    if failed_raw.is_symlink():
        raise RuntimeError("failed launcher directory must not be a symlink")
    failed = failed_raw.resolve(strict=True)
    if root not in failed.parents or not failed.is_dir():
        raise RuntimeError("failed launcher directory must resolve below operational root")
    if not SHA_PATTERN.fullmatch(args.manifest_sha256):
        raise RuntimeError("manifest SHA256 must be 64 lowercase hex digits")
    if not IMAGE_PATTERN.fullmatch(args.official_image_id):
        raise RuntimeError("official image must be sha256:<64 lowercase hex>")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite operational failure ledger: {output}")
    output_parent = output.parent.resolve(strict=True)
    if output_parent != root and root not in output_parent.parents:
        raise RuntimeError("operational ledger must be written below operational root")
    if failed in output.parents:
        raise RuntimeError("operational ledger must be outside the failed launcher tree")

    tree = tree_identity(failed)
    payload = {
        "schema": SCHEMA,
        "status": "locked_operational_tool_failure_excluded",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": file_identity(Path(__file__)),
        "claims": {
            "listed_run_is_not_formal_acceptance_evidence": True,
            "successful_inner_model_smoke_does_not_replace_missing_outer_attestation": True,
            "formal_receipts_are_not_overwritten": True,
            "fresh_formal_launcher_rerun_required": True,
            "listed_run_satisfies_any_deployment_gate": False,
        },
        "entry": {
            "entry_id": "k100_3_launcher_host_output_permission_failure_v1",
            "category": "operational_tool_failure",
            "failure_mode": "host_output_permission_denied_after_successful_container_smoke",
            "failure_stage": "post_container_smoke_host_attestation_write",
            "candidate_label": args.candidate_label,
            "node_label": "K100-3",
            "manifest_sha256": args.manifest_sha256,
            "official_image_id": args.official_image_id,
            "inner_model_smoke_status": "passed_observation_only",
            "outer_launch_attestation_written": False,
            "counts_toward_acceptance": False,
            "artifact": {
                "kind": "directory_tree",
                "path": failed.relative_to(root).as_posix(),
                "file_count": tree["file_count"],
                "total_bytes": tree["total_bytes"],
                "tree_sha256": tree["tree_sha256"],
            },
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ledger": file_identity(output),
                "next": "pass this exact SHA to aggregate_final_deployment_acceptance.py",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
