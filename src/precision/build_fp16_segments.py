#!/usr/bin/env python3
'Research implementation: build fp16 segments.'

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path


BUILDER_IDENTITY = (43_770, "d2555c1bdd5c6c975218e5c371e1ae5de92075fe3e734e21c7ff936ed9ae81b1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-builder", required=True, type=Path)
    parser.add_argument("--fp16-source", required=True, type=Path)
    parser.add_argument("--fp16-source-report", required=True, type=Path)
    parser.add_argument("--int8-quantization-report", required=True, type=Path)
    parser.add_argument("--frozen-int8-manifest", required=True, type=Path)
    parser.add_argument("--repaired-head-model", required=True, type=Path)
    parser.add_argument("--repaired-head-cache", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    builder = args.legacy_builder.resolve(strict=True)
    if (builder.stat().st_size, sha256(builder)) != BUILDER_IDENTITY:
        raise RuntimeError("frozen Legacy FP16 segment builder identity drift")
    spec = importlib.util.spec_from_file_location("legacy_stage1_fp16_builder", builder)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen Legacy FP16 segment builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The frozen implementation already contains the audited extraction, FP16
    # conversion, LayerNorm FP32-statistics island, boundary-cast, lineage, and
    # static-contract logic. Only its requested block set is expanded.
    module.CANDIDATES = {"M4": frozenset(range(24))}
    module.FP16_BLOCKS = tuple(range(24))
    sys.argv = [
        str(builder),
        "--fp16-source", str(args.fp16_source),
        "--fp16-source-report", str(args.fp16_source_report),
        "--int8-quantization-report", str(args.int8_quantization_report),
        "--frozen-int8-manifest", str(args.frozen_int8_manifest),
        "--repaired-head-model", str(args.repaired_head_model),
        "--repaired-head-cache", str(args.repaired_head_cache),
        "--output-root", str(args.output_root),
        "--candidates", "M4",
        "--payload-mode", "hardlink",
    ]
    module.main()


if __name__ == "__main__":
    main()
