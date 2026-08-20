#!/usr/bin/env python3
"""Create, once, the locked ledger for the two known invalid K100-3 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "phase11_k100_invalid_environmental_run_exclusions_v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.absolute()
    if path.is_symlink():
        raise RuntimeError(f"symlinks are forbidden: {path}")
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"artifact must be a non-empty regular file: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def below(root: Path, path: Path, role: str) -> Path:
    unresolved = path.absolute()
    if unresolved.is_symlink():
        raise RuntimeError(f"{role} must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if root not in resolved.parents:
        raise RuntimeError(f"{role} must resolve below exclusion root")
    return resolved


def directory_tree_identity(directory: Path) -> dict:
    if not directory.is_dir():
        raise RuntimeError(f"not a directory tree: {directory}")
    rows = []
    total = 0
    for parent, names, files in os.walk(directory, followlinks=False):
        parent_path = Path(parent)
        for name in names:
            if (parent_path / name).is_symlink():
                raise RuntimeError("directory-tree artifacts must not contain symlinks")
        for name in files:
            node = parent_path / name
            if node.is_symlink() or not node.is_file():
                raise RuntimeError("directory-tree artifacts require regular files")
            row = {
                "path": node.relative_to(directory).as_posix(),
                "size_bytes": node.stat().st_size,
                "sha256": sha256(node),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclusion-root", type=Path, required=True)
    parser.add_argument("--late-path-artifact", type=Path, required=True)
    parser.add_argument("--nohyhal-hip100-artifact", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--official-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_raw = args.exclusion_root.absolute()
    if root_raw.is_symlink():
        raise RuntimeError("exclusion root must not be a symlink")
    root = root_raw.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("exclusion root must be a directory")
    if not SHA_PATTERN.fullmatch(args.manifest_sha256):
        raise RuntimeError("manifest SHA256 must be 64 lowercase hex digits")
    if not IMAGE_PATTERN.fullmatch(args.official_image_id):
        raise RuntimeError("official image ID must be sha256:<64 lowercase hex>")
    artifacts = {
        "late_path": below(root, args.late_path_artifact, "late-PATH artifact"),
        "nohyhal": below(root, args.nohyhal_hip100_artifact, "no-hyhal artifact"),
    }
    if len(set(artifacts.values())) != 2:
        raise RuntimeError("the two invalid attempts require distinct artifacts")
    late, nohyhal = artifacts["late_path"], artifacts["nohyhal"]
    if (late.is_dir() and late in nohyhal.parents) or (
        nohyhal.is_dir() and nohyhal in late.parents
    ):
        raise RuntimeError("the two invalid-attempt artifact trees must not overlap")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite exclusion ledger: {output}")
    output_parent = output.parent.resolve(strict=True)
    if output_parent != root and root not in output_parent.parents:
        raise RuntimeError("ledger output must be below exclusion root")
    if any(path.is_dir() and path in output.parents for path in artifacts.values()):
        raise RuntimeError("ledger output must be outside both locked artifact trees")

    def artifact_row(path: Path) -> dict:
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            row = identity(path)
            return {
                "kind": "file",
                "path": relative,
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
        if path.is_dir():
            return {
                "kind": "directory_tree",
                "path": relative,
                **directory_tree_identity(path),
            }
        raise RuntimeError(f"unsupported invalid-run artifact type: {path}")

    common = {
        "node_label": "K100-3",
        "manifest_sha256": args.manifest_sha256,
        "official_image_id": args.official_image_id,
        "accepted_environment": False,
        "counts_toward_acceptance": False,
        "model_execution_started": False,
        "successful_model_inferences": 0,
    }
    payload = {
        "schema": SCHEMA,
        "status": "locked_invalid_environmental_runs_excluded",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": identity(Path(__file__)),
        "claims": {
            "listed_runs_are_not_acceptance_evidence": True,
            "formal_receipts_are_not_overwritten": True,
            "fresh_formal_rerun_required": True,
            "listed_runs_satisfy_any_deployment_gate": False,
        },
        "entries": [
            {
                "entry_id": "k100_3_bdc7_hyhal_path_injected_too_late",
                "failure_mode": "bdc7_with_node3_host_hyhal_lsmod_recursion_before_model",
                "failure_stage": "runtime_import_before_model_inference",
                **common,
                "artifacts": [artifact_row(artifacts["late_path"])],
            },
            {
                "entry_id": "k100_3_bdc7_nohyhal_hip100",
                "failure_mode": "bdc7_without_host_hyhal_hip100_no_rocm_device",
                "failure_stage": "model_session_creation_before_successful_inference",
                **common,
                "artifacts": [artifact_row(artifacts["nohyhal"])],
            },
        ],
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    row = identity(output)
    print(
        json.dumps(
            {
                "status": "locked_invalid_environmental_runs_excluded",
                "ledger": row,
                "next": "pass this exact SHA to aggregate_final_deployment_acceptance.py",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
