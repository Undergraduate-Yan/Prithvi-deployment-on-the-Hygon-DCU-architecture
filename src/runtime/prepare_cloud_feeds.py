#!/usr/bin/env python3
'Research implementation: prepare cloud feeds.'
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--dv-inputs", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    inputs = np.load(args.dv_inputs, mmap_mode="r")
    if inputs.shape != (1280, 6, 224, 224) or not 0 <= args.sample_index < 1280:
        raise RuntimeError("deployment-validation input contract drift")
    state: dict[str, np.ndarray] = {"input": np.ascontiguousarray(inputs[args.sample_index:args.sample_index + 1], dtype=np.float32)}
    rows = []
    for ordinal, model in enumerate(args.model):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model.resolve(strict=True)), sess_options=options, providers=["CPUExecutionProvider"])
        names = [item.name for item in session.get_inputs()]
        feeds = {name: np.ascontiguousarray(state[name]) for name in names}
        if not all(np.isfinite(value).all() for value in feeds.values()):
            raise RuntimeError(f"non-finite feed at session {ordinal}")
        feed_path = args.output_root / f"session_{ordinal:02d}.npz"
        np.savez(feed_path, **feeds)
        output_names = [item.name for item in session.get_outputs()]
        outputs = session.run(output_names, feeds)
        if not all(np.isfinite(value).all() for value in outputs):
            raise RuntimeError(f"non-finite CPU output at session {ordinal}")
        state.update(zip(output_names, outputs))
        rows.append({
            "ordinal": ordinal, "model": identity(model), "feeds": identity(feed_path),
            "input_names": names, "input_shapes": {name: list(feeds[name].shape) for name in names},
            "input_dtypes": {name: str(feeds[name].dtype) for name in names},
            "output_names": output_names,
        })
    manifest = {
        "schema": "phase7r_chain_compile_feeds_v1", "status": "PASS", "formal_payload_accessed": False,
        "sample_scope": f"deployment-validation ordinal {args.sample_index}", "session_count": len(rows),
        "dv_inputs": identity(args.dv_inputs), "sessions": rows,
    }
    output = args.output_root / "compile_feeds_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "sessions": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
