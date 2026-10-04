#!/usr/bin/env python3
'Research implementation: assemble cloud mixed.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model-root", required=True, type=Path)
    parser.add_argument("--fp16-cache-root", required=True, type=Path)
    parser.add_argument("--fp16-tail-model", action="append", required=True, type=Path)
    parser.add_argument("--fp16-tail-cache", action="append", required=True, type=Path)
    parser.add_argument("--mp-model", required=True, type=Path)
    parser.add_argument("--mp-cache", required=True, type=Path)
    parser.add_argument("--replaced-ordinal", required=True, type=int)
    parser.add_argument("--int8-block", action="append", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    if not 0 <= args.replaced_ordinal < 12:
        raise ValueError("--replaced-ordinal must be in [0, 11]")
    if len(args.fp16_tail_model) != 2 or len(args.fp16_tail_cache) != 2:
        raise ValueError("exactly two FP16-v2 tail models and caches are required")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    models: list[Path] = []
    caches: list[Path] = []
    for ordinal in range(12):
        models.append(
            args.mp_model
            if ordinal == args.replaced_ordinal
            else args.fp16_model_root / f"session_{ordinal:02d}.onnx"
        )
        caches.append(
            args.mp_cache
            if ordinal == args.replaced_ordinal
            else args.fp16_cache_root / f"session_{ordinal:02d}.mxr"
        )
    models.extend(args.fp16_tail_model)
    caches.extend(args.fp16_tail_cache)
    for path in [*models, *caches]:
        path.resolve(strict=True)

    model_dir = args.output_root / "models"
    cache_dir = args.output_root / "caches"
    model_dir.mkdir(parents=True)
    cache_dir.mkdir()
    assets: list[dict[str, object]] = []
    for ordinal, (model, cache) in enumerate(zip(models, caches)):
        model_link = model_dir / f"session_{ordinal:02d}.onnx"
        cache_link = cache_dir / f"session_{ordinal:02d}.mxr"
        os.symlink(model.resolve(), model_link)
        os.symlink(cache.resolve(), cache_link)
        assets.append({
            "ordinal": ordinal,
            "precision": "INT8-QDQ-in-FP16" if ordinal == args.replaced_ordinal else "FP16",
            "model": identity(model_link),
            "cache": identity(cache_link),
        })

    result = {
        "schema": "phase7r_cloud_mp_task_hybrid_assembly_v1",
        "status": "PASS_ASSEMBLED_PENDING_K100_FULL_DV",
        "role": "Cloud-RCS-MP-Task-v2",
        "session_count": 14,
        "int8_blocks": sorted(set(args.int8_block)),
        "replaced_session_ordinal": args.replaced_ordinal,
        "single_variable": (
            "replace only the FP16-v2 session containing the admitted INT8 block; "
            "reuse the other 13 validated FP16-v2 model/cache pairs byte-for-byte"
        ),
        "full_chain_compile_attempt": (
            "not used for final identity; the failed 58 GiB full-chain attempt is retained "
            "as compile-resource evidence"
        ),
        "formal_payload_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }
    (args.output_root / "hybrid_assembly_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "session_count": result["session_count"],
        "int8_blocks": result["int8_blocks"],
        "replaced_session_ordinal": result["replaced_session_ordinal"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
