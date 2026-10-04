#!/usr/bin/env python3
'Research implementation: flood runner.'

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "journal_stage2_k100_variant_v1"
MGX = "MIGraphXExecutionProvider"
EXPECTED_ORT = "1.19.2"
INPUT_SHAPE = (1, 6, 224, 224)
LOGITS_SHAPE = (1, 2, 224, 224)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_identity(row: dict[str, Any], label: str) -> Path:
    path = Path(row["path"]).resolve(strict=True)
    observed = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    expected = {"size_bytes": int(row["size_bytes"]), "sha256": str(row["sha256"])}
    if observed != expected:
        raise RuntimeError(f"{label} identity drift: observed={observed}; expected={expected}")
    return path


def load_input(path: Path, expected: dict[str, Any] | None = None) -> np.ndarray:
    path = path.resolve(strict=True)
    if expected is not None:
        observed = identity(path)
        wanted = {key: expected[key] for key in ("path", "size_bytes", "sha256")}
        if observed != wanted:
            raise RuntimeError(f"input identity drift: observed={observed}; expected={wanted}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.dtype("float32") or tuple(value.shape) != INPUT_SHAPE:
        raise RuntimeError(f"input contract drift: {value.dtype}{value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError("input contains NaN or Inf")
    return np.ascontiguousarray(value)


def _session_options(ort: Any, profile_prefix: Path | None) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        profile_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def _set_cache(cache: Path) -> None:
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache)


def profile_counts(path: Path) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    return {
        "profile": identity(path),
        "provider_event_counts": counts,
        "migraphx_events_positive": counts.get(MGX, 0) > 0,
        "cpu_events_zero": counts.get("CPUExecutionProvider", 0) == 0,
    }


class K100BundleRunner:
    """One runner path for mono and segmented Stage-2 variants."""

    def __init__(self, manifest_path: Path, device: int = 0, profile_dir: Path | None = None) -> None:
        started = time.perf_counter()
        self.manifest_path = manifest_path.resolve(strict=True)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("status") != "frozen_ready":
            raise RuntimeError("variant manifest is unsupported or not ready")
        contract = self.manifest.get("external_contract", {})
        if contract != {
            "input": "FP32[1,6,224,224]",
            "output": "FP32[1,2,224,224] logits",
            "batch": 1,
        }:
            raise RuntimeError(f"external contract drift: {contract}")
        self.device = int(device)
        if self.device < 0:
            raise RuntimeError("device must be non-negative")
        shim = verify_identity(self.manifest["runtime_contract"]["static_lsmod"], "static lsmod")
        if not os.access(shim, os.X_OK):
            raise RuntimeError("static lsmod is not executable")
        entries = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join([str(shim.parent), *[item for item in entries if item != str(shim.parent)]])
        if Path(shutil.which("lsmod") or "").resolve(strict=False) != shim:
            raise RuntimeError("static lsmod is not first on PATH")

        import onnxruntime as ort

        if ort.__version__ != EXPECTED_ORT or MGX not in ort.get_available_providers():
            raise RuntimeError(f"runtime identity drift: {ort.__version__}, {ort.get_available_providers()}")
        self.ort = ort
        self.profile_dir = profile_dir.resolve(strict=False) if profile_dir else None
        self.sessions: list[Any] = []
        execution = self.manifest["execution"]
        self.kind = str(execution["kind"])
        rows = [execution] if self.kind == "mono" else execution.get("segments", [])
        if self.kind not in {"mono", "segment25", "segment13", "chain"}:
            raise RuntimeError(f"unsupported execution kind: {self.kind}")
        if self.kind == "segment25" and (
            len(rows) != 25 or [int(row["index"]) for row in rows] != list(range(25))
        ):
            raise RuntimeError("segment25 execution order drift")
        if self.kind == "segment13" and (
            len(rows) != 13 or [int(row["index"]) for row in rows] != list(range(13))
        ):
            raise RuntimeError("segment13 execution order drift")
        if self.kind == "chain" and (
            len(rows) < 2 or [int(row["index"]) for row in rows] != list(range(len(rows)))
        ):
            raise RuntimeError("chain execution order drift")
        self.rows = rows
        for index, row in enumerate(rows):
            model = verify_identity(row["model"], f"model {index}")
            cache = verify_identity(row["cache"], f"cache {index}")
            _set_cache(cache)
            prefix = self.profile_dir / f"session_{index:02d}" if self.profile_dir else None
            session = ort.InferenceSession(
                str(model),
                sess_options=_session_options(ort, prefix),
                providers=[(MGX, {"device_id": self.device})],
            )
            session.disable_fallback()
            if not session.get_providers() or session.get_providers()[0] != MGX:
                raise RuntimeError(f"provider priority drift: {session.get_providers()}")
            self.sessions.append(session)
        self.retain_after = tuple(int(value) for value in execution.get("retain_after", []))
        if self.kind == "segment25" and self.retain_after != (5, 11, 17, 23):
            raise RuntimeError(f"retained feature contract drift: {self.retain_after}")
        self.runtime = {
            "hostname": platform.node(),
            "onnxruntime": ort.__version__,
            "provider": MGX,
            "cpu_fallback_disabled": True,
            "device": self.device,
            "variant": self.manifest["variant"],
            "session_count": len(self.sessions),
        }
        self.load_seconds = time.perf_counter() - started

    @staticmethod
    def _bind(session: Any, inputs: dict[str, Any], output_names: list[str], device: int) -> list[Any]:
        if {item.name for item in session.get_inputs()} != set(inputs):
            raise RuntimeError("session input-name contract drift")
        binding = session.io_binding()
        for name, value in inputs.items():
            binding.bind_ortvalue_input(name, value)
        for name in output_names:
            binding.bind_output(name, "cuda", device)
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        outputs = binding.get_outputs()
        if len(outputs) != len(output_names) or not all(value.device_name() == "cuda" for value in outputs):
            raise RuntimeError("device output contract drift")
        return outputs

    def _run_mono(self, raw: np.ndarray) -> np.ndarray:
        session = self.sessions[0]
        row = self.rows[0]
        device_input = self.ort.OrtValue.ortvalue_from_numpy(raw, "cuda", self.device)
        values = self._bind(session, {str(row["input_name"]): device_input}, [str(row["output_name"])], self.device)
        return values[0].numpy()

    def _run_segment25(self, raw: np.ndarray) -> np.ndarray:
        current = self.ort.OrtValue.ortvalue_from_numpy(raw, "cuda", self.device)
        retained: dict[str, Any] = {}
        for index, (row, session) in enumerate(zip(self.rows, self.sessions, strict=True)):
            if index < 24:
                inputs = {session.get_inputs()[0].name: current}
            else:
                inputs = {item.name: retained[item.name] for item in session.get_inputs()}
            output_names = [item.name for item in session.get_outputs()]
            values = self._bind(session, inputs, output_names, self.device)
            output_map = dict(zip(output_names, values, strict=True))
            current = output_map["logits"] if index == 24 else values[0]
            if index in self.retain_after:
                retained[output_names[0]] = current
        return current.numpy()

    def _run_chain(self, raw: np.ndarray) -> np.ndarray:
        available: dict[str, Any] = {
            "image": self.ort.OrtValue.ortvalue_from_numpy(raw, "cuda", self.device)
        }
        current = available["image"]
        for index, (row, session) in enumerate(zip(self.rows, self.sessions, strict=True)):
            session_inputs = [item.name for item in session.get_inputs()]
            missing = [name for name in session_inputs if name not in available]
            if missing:
                raise RuntimeError(f"chain inputs have no earlier producer at session {index}: {missing}")
            inputs = {name: available[name] for name in session_inputs}
            output_names = [item.name for item in session.get_outputs()]
            values = self._bind(session, inputs, output_names, self.device)
            output_map = dict(zip(output_names, values, strict=True))
            primary = str(row["primary_output"])
            if primary not in output_map:
                raise RuntimeError(f"chain primary output missing at session {index}: {primary}")
            current = output_map[primary]
            available.update(output_map)
        return current.numpy()

    def run(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if raw.dtype != np.dtype("float32") or tuple(raw.shape) != INPUT_SHAPE or not np.isfinite(raw).all():
            raise RuntimeError("run input violates FP32[1,6,224,224] finite contract")
        started = time.perf_counter_ns()
        if self.kind == "mono":
            logits = self._run_mono(raw)
        elif self.kind == "segment25":
            logits = self._run_segment25(raw)
        else:
            logits = self._run_chain(raw)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if logits.dtype != np.dtype("float32") or tuple(logits.shape) != LOGITS_SHAPE or not np.isfinite(logits).all():
            raise RuntimeError("run output violates FP32[1,2,224,224] finite logits contract")
        logits = np.ascontiguousarray(logits)
        prediction = np.ascontiguousarray(np.argmax(logits, axis=1).astype(np.uint8, copy=False))
        return logits, prediction, elapsed_ms

    def finish_profiles(self) -> list[dict[str, Any]]:
        if self.profile_dir is None:
            raise RuntimeError("profiling was not enabled")
        rows = []
        for index, session in enumerate(self.sessions):
            path = Path(session.end_profiling()).resolve(strict=True)
            row = {"session_index": index, **profile_counts(path)}
            if not row["migraphx_events_positive"] or not row["cpu_events_zero"]:
                raise RuntimeError(f"provider placement gate failed: {row}")
            rows.append(row)
        return rows
