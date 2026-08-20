#!/usr/bin/env python3
"""CPU sequential parity for 24 one-block encoder segments plus head."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

SOURCE = (368_047_490, "edd2c3e69b64986dc6d0b78fe32230f9a9f7eb0396c2d77fdcb6bf4c59506a7a")
SAMPLE = (4_820_344, "4822aa763ccb1eba7ab3609326297255dc7d4cb41b19116f0b5b936818daec62")
RAW_SHA = "23bd7b08aa08352cfa52b0086c3a6aac18b8b0e6a63f3cdbcae7409c667aa62c"
BLOCK = {i: f"/task/model/encoder/blocks.{i}/Add_1_output_0" for i in range(24)}
LABELS = tuple(f"encoder_block_{i:02d}" for i in range(24)) + ("upernet_decoder_head",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def lock(path: Path, expected: tuple[int, str]) -> dict:
    path = path.resolve(strict=True); item = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if (item["size_bytes"], item["sha256"]) != expected: raise RuntimeError(f"identity drift: {item}")
    return item


def load_image(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False); found = []
    def visit(value):
        if torch.is_tensor(value) and value.ndim == 4 and tuple(value.shape[1:]) == (6, 224, 224): found.append(value)
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value: visit(child)
    visit(obj); value = np.ascontiguousarray(found[0][:1].numpy(), dtype=np.float32)
    if hashlib.sha256(value.tobytes()).hexdigest() != RAW_SHA: raise RuntimeError("raw input drift")
    return value


def record(value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value); return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(np.isfinite(value).all()), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", type=Path, required=True); ap.add_argument("--sample", type=Path, required=True); ap.add_argument("--segment-report", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    identities = {"source": lock(args.source, SOURCE), "sample": lock(args.sample, SAMPLE)}; report = json.loads(args.segment_report.read_text(encoding="utf-8")); rows = report.get("segments", [])
    if report.get("status") != "created_static_pass" or [r.get("label") for r in rows] != list(LABELS): raise RuntimeError("segment report drift")
    paths = []
    for row in rows:
        path = Path(row["path"]).resolve(strict=True); identities[row["label"]] = lock(path, (int(row["size_bytes"]), str(row["sha256"]))); paths.append(path)
    import onnxruntime as ort
    if ort.__version__ != "1.19.2": raise RuntimeError("ORT drift")
    opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; opts.intra_op_num_threads = 4; opts.inter_op_num_threads = 1
    raw = load_image(args.sample); started = perf_counter(); source = ort.InferenceSession(str(args.source), sess_options=opts, providers=["CPUExecutionProvider"]); expected = np.asarray(source.run(["logits"], {"image": raw})[0]); source_seconds = perf_counter()-started; del source
    sessions = [ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"]) for path in paths]; value = raw; retained = {}; intermediate = {}; seconds = {}
    for index in range(24):
        name = BLOCK[index]; started = perf_counter(); value = np.asarray(sessions[index].run([name], {sessions[index].get_inputs()[0].name: value})[0]); seconds[LABELS[index]] = perf_counter()-started; retained[name] = value; intermediate[name] = record(value)
    feeds = {item.name: retained[item.name] for item in sessions[24].get_inputs()}; started = perf_counter(); actual = np.asarray(sessions[24].run(["logits"], feeds)[0]); seconds[LABELS[24]] = perf_counter()-started
    diff = np.abs(expected.astype(np.float64)-actual.astype(np.float64)); p0 = np.argmax(expected,1); p1=np.argmax(actual,1); comparison={"mae":float(diff.mean()),"max_abs":float(diff.max()),"rmse":float(np.sqrt(np.mean(diff*diff))),"p99_abs":float(np.percentile(diff,99)),"pixel_class_agreement":float(np.mean(p0==p1)),"changed_pixels":int(np.count_nonzero(p0!=p1))}; gates={"all_finite":bool(np.isfinite(expected).all() and np.isfinite(actual).all() and all(x["finite"] for x in intermediate.values())),"mae_le_1e_6":comparison["mae"]<=1e-6,"max_abs_le_1e_5":comparison["max_abs"]<=1e-5,"agreement_eq_100pct":comparison["pixel_class_agreement"]==1.0}
    result={"status":"passed" if all(gates.values()) else "failed","variant":"int8_backbone_compat_25_static_batch1_onnx_segments_cpu_sequential","identities":identities,"runtime":{"onnxruntime":ort.__version__,"provider":"CPUExecutionProvider"},"raw_input_sha256":RAW_SHA,"intermediates":intermediate,"source_logits":record(expected),"segmented_logits":record(actual),"comparison":comparison,"gates":gates,"diagnostic_seconds":{"source":source_seconds,"segments":seconds},"claims":{"cpu_sequential_parity":all(gates.values()),"strict_migraphx_admission":False,"device_resident_intersegment_io":False,"task_accuracy_90":False,"performance":False,"native_int8_kernel_verified":False}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(result,indent=2,ensure_ascii=False),flush=True);raise SystemExit(0 if result["status"]=="passed" else 2)


if __name__ == "__main__": main()
