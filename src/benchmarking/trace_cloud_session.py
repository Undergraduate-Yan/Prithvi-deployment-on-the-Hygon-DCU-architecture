#!/usr/bin/env python3
'Research implementation: trace cloud session.'
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"


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
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--feeds", required=True, type=Path)
    parser.add_argument("--session-ordinal", required=True, type=int)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.repetitions != 20 or args.output_root.exists():
        raise RuntimeError("trace is frozen at 20 repetitions and a new output root")
    args.output_root.mkdir(parents=True)
    os.environ.update({
        "ORT_MIGRAPHX_SAVE_COMPILED_MODEL": "0", "ORT_MIGRAPHX_LOAD_COMPILED_MODEL": "1",
        "ORT_MIGRAPHX_SAVE_COMPILE_PATH": str(args.cache.resolve(strict=True)),
        "ORT_MIGRAPHX_LOAD_COMPILE_PATH": str(args.cache.resolve(strict=True)),
        "MIGRAPHX_GPU_COMPILE_PARALLEL": "1", "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "0",
    })
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output_root / "provider_profile")
    session = ort.InferenceSession(str(args.model.resolve(strict=True)), sess_options=options, providers=[(MGX, {"device_id": 0})])
    session.disable_fallback()
    with np.load(args.feeds.resolve(strict=True), allow_pickle=False) as packed:
        feeds = {name: np.ascontiguousarray(packed[name]) for name in packed.files}
    if set(feeds) != {item.name for item in session.get_inputs()}:
        raise RuntimeError("trace feed contract drift")
    output_names = [item.name for item in session.get_outputs()]
    outputs = None
    for _ in range(args.repetitions):
        outputs = session.run(output_names, feeds)
    if not outputs or not all(np.isfinite(value).all() for value in outputs):
        raise RuntimeError("trace output is empty or non-finite")
    profile = Path(session.end_profiling()).resolve(strict=True)
    events = json.loads(profile.read_text(encoding="utf-8"))
    counts = Counter(str(row["args"]["provider"]) for row in events if row.get("args", {}).get("provider"))
    result = {
        "schema": "phase7r_direct_session_trace_v1", "status": "PASS_PENDING_HIPPROF_PARSE",
        "session_ordinal": args.session_ordinal, "repetitions": args.repetitions,
        "formal_payload_accessed": False, "model": identity(args.model), "cache": identity(args.cache),
        "feeds": identity(args.feeds), "provider_profile": identity(profile),
        "provider_event_counts": dict(counts), "cpu_fallback_events": counts[CPU],
        "output_names": output_names, "output_dtypes": [str(value.dtype) for value in outputs],
    }
    if counts[MGX] <= 0 or counts[CPU] != 0:
        result["status"] = "FAIL"
    output = args.output_root / "result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
