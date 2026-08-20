#!/usr/bin/env python3
"""Fail-closed K100 inference for a manifest-defined Phase 11 bundle.

The public interface is intentionally small::

    infer_k100.py --bundle BUNDLE --input INPUT.npy --output OUTPUT.npz --device 0

No preprocessing is performed.  In particular, the six input bands must already
have been multiplied by 1e-4 and mean/std normalization must not be applied by the
caller because it is embedded in the ONNX graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "phase11_k100_minimal_deployment_bundle_v1"
MGX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"
EXPECTED_ORT = "1.19.2"
LSMOD_SIZE = 819_664
LSMOD_SHA256 = "9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12"
INPUT_SHAPE = (1, 6, 224, 224)
LOGITS_SHAPE = (1, 2, 224, 224)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _contained(root: Path, child: Path) -> bool:
    return child == root or root in child.parents


def resolve_payload(bundle: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"bundle path must be non-empty and relative: {relative!r}")
    path = (bundle / relative).resolve(strict=True)
    if not _contained(bundle, path):
        raise RuntimeError(f"bundle path escapes root: {relative!r}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"bundle payload must be a regular non-symlink file: {relative}")
    return path


def verify_identity(path: Path, expected: dict[str, Any]) -> None:
    observed = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    wanted = {"size_bytes": int(expected["size_bytes"]), "sha256": str(expected["sha256"])}
    if observed != wanted:
        raise RuntimeError(f"payload identity drift for {path}: {observed}; expected={wanted}")


def ensure_locked_lsmod(bundle: Path, manifest: dict) -> dict:
    """Install the frozen static lsmod at PATH front before ORT/torch imports."""
    shim = resolve_payload(bundle, "tools/bin/lsmod")
    observed = {"size_bytes": shim.stat().st_size, "sha256": sha256(shim)}
    expected = {"size_bytes": LSMOD_SIZE, "sha256": LSMOD_SHA256}
    if observed != expected:
        raise RuntimeError(f"static lsmod identity drift: {observed}; expected={expected}")
    if not os.access(shim, os.X_OK):
        raise RuntimeError(f"static lsmod is not executable: {shim}")
    recorded = manifest.get("runtime_contract", {}).get("static_lsmod", {})
    if recorded.get("path") != "tools/bin/lsmod" or {
        "size_bytes": int(recorded.get("size_bytes", -1)),
        "sha256": str(recorded.get("sha256", "")),
    } != expected:
        raise RuntimeError("manifest static lsmod contract drift")
    shim_dir = str(shim.parent)
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    os.environ["PATH"] = os.pathsep.join([shim_dir, *[item for item in entries if item != shim_dir]])
    selected = shutil.which("lsmod")
    if selected is None or Path(selected).resolve(strict=True) != shim:
        raise RuntimeError("locked static lsmod is not first on PATH")
    return {"path": str(shim), **observed, "path_prepend_active": True}


def load_manifest(bundle: Path, verify_all_payloads: bool = True) -> tuple[dict, dict]:
    bundle = bundle.resolve(strict=True)
    manifest_path = resolve_payload(bundle, "manifest.json")
    manifest_identity = {
        "path": str(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256(manifest_path),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "static_bundle_identity_locked":
        raise RuntimeError("unsupported or unfinished K100 bundle manifest")
    if not manifest.get("claims", {}).get("static_bundle_ready"):
        raise RuntimeError("manifest does not admit static bundle use")
    if verify_all_payloads:
        expected = {str(row["path"]): row for row in manifest.get("files", [])}
        if not expected or len(expected) != len(manifest.get("files", [])):
            raise RuntimeError("manifest payload inventory is empty or contains duplicate paths")
        actual = []
        for path in sorted(bundle.rglob("*")):
            relative = str(path.relative_to(bundle)).replace("\\", "/")
            if path.is_symlink():
                raise RuntimeError(f"symlink is forbidden in bundle: {relative}")
            if path.is_file() and relative != "manifest.json":
                actual.append(relative)
        if set(actual) != set(expected):
            raise RuntimeError(
                f"bundle path-set drift: missing={sorted(set(expected)-set(actual))}, "
                f"unexpected={sorted(set(actual)-set(expected))}"
            )
        for relative in actual:
            verify_identity(resolve_payload(bundle, relative), expected[relative])
    return manifest, manifest_identity


def load_input(path: Path) -> np.ndarray:
    path = path.resolve(strict=True)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as pack:
            if "input" not in pack.files:
                raise RuntimeError("NPZ input must contain an array named 'input'")
            if pack.files != ["input"]:
                raise RuntimeError(f"NPZ input must contain only 'input'; found={pack.files}")
            array = pack["input"]
    else:
        raise RuntimeError("input must be .npy or .npz")
    if array.dtype != np.dtype("float32"):
        raise RuntimeError(f"input dtype must be exactly float32; got {array.dtype}")
    if tuple(array.shape) != INPUT_SHAPE:
        raise RuntimeError(f"input shape must be exactly {INPUT_SHAPE}; got {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("input contains NaN or Inf")
    # Making an equivalent contiguous view/copy is not a dtype, shape, or value conversion.
    return np.ascontiguousarray(array)


def _session_options(ort):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


def _set_cache(cache: Path) -> None:
    os.environ["ORT_MIGRAPHX_SAVE_COMPILED_MODEL"] = "0"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILED_MODEL"] = "1"
    os.environ["ORT_MIGRAPHX_LOAD_COMPILE_PATH"] = str(cache)
    os.environ["ORT_MIGRAPHX_SAVE_COMPILE_PATH"] = str(cache)


def _create_session(ort, model: Path, cache: Path, device: int):
    _set_cache(cache)
    session = ort.InferenceSession(
        str(model),
        sess_options=_session_options(ort),
        providers=[(MGX, {"device_id": int(device)})],
    )
    session.disable_fallback()
    providers = session.get_providers()
    # ORT may report CPUExecutionProvider as a registered provider even when
    # `session.disable_cpu_ep_fallback=1` forbids graph assignment to it.
    if not providers or providers[0] != MGX:
        raise RuntimeError(f"provider contract drift: {providers}")
    return session


class K100BundleRunner:
    """Loaded bundle runner used by the CLI and long-running acceptance harness."""

    def __init__(self, bundle: Path, device: int = 0, verify_payloads: bool = True) -> None:
        started = time.perf_counter()
        self.bundle = bundle.resolve(strict=True)
        self.device = int(device)
        if self.device < 0:
            raise RuntimeError("device must be a non-negative integer")
        integrity_started = time.perf_counter()
        self.manifest, self.manifest_identity = load_manifest(self.bundle, verify_payloads)
        self.payloads_fully_verified = bool(verify_payloads)
        self.integrity_validation_seconds = time.perf_counter() - integrity_started
        contract = self.manifest.get("runtime_contract", {})
        locked_lsmod = ensure_locked_lsmod(self.bundle, self.manifest)
        import onnxruntime as ort

        if ort.__version__ != str(contract.get("onnxruntime", EXPECTED_ORT)):
            raise RuntimeError(
                f"onnxruntime version drift: {ort.__version__}; expected={contract.get('onnxruntime')}"
            )
        providers = ort.get_available_providers()
        if MGX not in providers:
            raise RuntimeError(f"{MGX} unavailable: {providers}")
        attested_image = os.environ.get("PHASE11_K100_IMAGE_ID")
        expected_image = str(contract.get("official_image_id", ""))
        if attested_image is not None and attested_image != expected_image:
            raise RuntimeError(
                f"container image attestation drift: {attested_image}; expected={expected_image}"
            )
        self.runtime = {
            "onnxruntime": ort.__version__,
            "available_providers": providers,
            "selected_provider": MGX,
            "cpu_fallback_disabled": True,
            "expected_official_image_id": expected_image,
            "attested_image_id": attested_image,
            "image_identity_attested": attested_image == expected_image,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "device": self.device,
            "static_lsmod": locked_lsmod,
        }
        self.ort = ort
        execution = self.manifest.get("execution", {})
        self.kind = str(execution.get("kind"))
        load_started = time.perf_counter()
        if self.kind == "segment25":
            rows = execution.get("segments", [])
            if len(rows) != 25 or [row.get("index") for row in rows] != list(range(25)):
                raise RuntimeError("segment25 execution contract must contain ordered indices 0..24")
            self.rows = rows
            self.sessions = []
            for row in rows:
                model = resolve_payload(self.bundle, row["model"]["path"])
                cache = resolve_payload(self.bundle, row["cache"]["path"])
                if not self.payloads_fully_verified:
                    verify_identity(model, row["model"])
                    verify_identity(cache, row["cache"])
                self.sessions.append(_create_session(ort, model, cache, self.device))
            self.retain_after = tuple(int(value) for value in execution["retain_encoder_outputs_after_segments"])
            if self.retain_after != (5, 11, 17, 23):
                raise RuntimeError(f"retained-output contract drift: {self.retain_after}")
        elif self.kind == "fp16_full":
            model_row = execution["model"]
            cache_row = execution["cache"]
            model = resolve_payload(self.bundle, model_row["path"])
            cache = resolve_payload(self.bundle, cache_row["path"])
            if not self.payloads_fully_verified:
                verify_identity(model, model_row)
                verify_identity(cache, cache_row)
            self.session = _create_session(ort, model, cache, self.device)
            self.input_name = str(execution["input_name"])
            self.output_name = str(execution["output_name"])
        else:
            raise RuntimeError(f"unsupported execution kind: {self.kind!r}")
        self.session_load_seconds = time.perf_counter() - load_started
        self.total_load_seconds = time.perf_counter() - started

    def _run_segment25(self, raw: np.ndarray) -> np.ndarray:
        current = self.ort.OrtValue.ortvalue_from_numpy(raw, "cuda", self.device)
        if current.device_name() != "cuda":
            raise RuntimeError("input OrtValue was not allocated on K100")
        retained: dict[str, Any] = {}
        for index, session in enumerate(self.sessions):
            binding = session.io_binding()
            inputs = session.get_inputs()
            if index < 24:
                if len(inputs) != 1:
                    raise RuntimeError(f"encoder segment {index} input-count drift")
                binding.bind_ortvalue_input(inputs[0].name, current)
            else:
                if {item.name for item in inputs} != set(retained):
                    raise RuntimeError("head inputs do not match retained device features")
                for item in inputs:
                    binding.bind_ortvalue_input(item.name, retained[item.name])
            output_names = [item.name for item in session.get_outputs()]
            for name in output_names:
                binding.bind_output(name, "cuda", self.device)
            binding.synchronize_inputs()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            values = binding.get_outputs()
            if len(values) != len(output_names) or not all(value.device_name() == "cuda" for value in values):
                raise RuntimeError(f"segment {index} output device/count contract failure")
            output_map = dict(zip(output_names, values, strict=True))
            if index == 24:
                if "logits" not in output_map:
                    raise RuntimeError("head output is missing logits")
                current = output_map["logits"]
            else:
                current = output_map[output_names[0]]
            if index in self.retain_after:
                retained[output_names[0]] = current
        logits = current.numpy()
        return logits

    def _run_fp16_full(self, raw: np.ndarray) -> np.ndarray:
        device_input = self.ort.OrtValue.ortvalue_from_numpy(raw, "cuda", self.device)
        binding = self.session.io_binding()
        binding.bind_ortvalue_input(self.input_name, device_input)
        binding.bind_output(self.output_name, "cuda", self.device)
        binding.synchronize_inputs()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        values = binding.get_outputs()
        if len(values) != 1 or values[0].device_name() != "cuda":
            raise RuntimeError("FP16 full output device/count contract failure")
        return values[0].numpy()

    def run(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if raw.dtype != np.dtype("float32") or tuple(raw.shape) != INPUT_SHAPE:
            raise RuntimeError("in-memory input violates strict float32[1,6,224,224] contract")
        if not np.isfinite(raw).all():
            raise RuntimeError("in-memory input contains NaN or Inf")
        started = time.perf_counter()
        logits = self._run_segment25(raw) if self.kind == "segment25" else self._run_fp16_full(raw)
        elapsed = time.perf_counter() - started
        if logits.dtype != np.dtype("float32"):
            raise RuntimeError(f"logits dtype must be exactly float32; got {logits.dtype}")
        if tuple(logits.shape) != LOGITS_SHAPE:
            raise RuntimeError(f"logits shape must be exactly {LOGITS_SHAPE}; got {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError("logits contain NaN or Inf")
        logits = np.ascontiguousarray(logits)
        prediction = np.ascontiguousarray(np.argmax(logits, axis=1).astype(np.uint8, copy=False))
        return logits, prediction, elapsed

    def fixed_prediction_gate(self, raw: np.ndarray, prediction: np.ndarray) -> dict:
        fixed = self.manifest.get("fixed_sample", {})
        input_digest = array_sha256(raw)
        is_fixed = input_digest == fixed.get("input_array_sha256")
        expected = fixed.get("expected_prediction_array_sha256")
        observed = array_sha256(prediction)
        return {
            "input_matches_bundled_fixed_sample": is_fixed,
            "expected_prediction_array_sha256": expected,
            "observed_prediction_array_sha256": observed,
            "prediction_matches_expected": bool(is_fixed and expected and observed == expected),
        }


def atomic_save_npz(path: Path, logits: np.ndarray, prediction: np.ndarray) -> None:
    path = path.resolve(strict=False)
    if path.suffix.lower() != ".npz":
        raise RuntimeError("output must use the .npz suffix")
    if path.exists():
        raise RuntimeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, logits=logits, prediction=prediction)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--skip-full-payload-hash",
        action="store_true",
        help="Only for a bundle already verified in this process/container; referenced model/cache hashes remain checked.",
    )
    args = parser.parse_args()
    bundle_root = args.bundle.resolve(strict=True)
    output_candidate = args.output.resolve(strict=False)
    receipt_candidate = (
        args.receipt.resolve(strict=False)
        if args.receipt
        else args.output.with_suffix(args.output.suffix + ".receipt.json").resolve(strict=False)
    )
    if output_candidate == receipt_candidate:
        raise RuntimeError("output NPZ and JSON receipt paths must differ")
    for role, candidate in (("output", output_candidate), ("receipt", receipt_candidate)):
        if _contained(bundle_root, candidate):
            raise RuntimeError(f"{role} must be outside the immutable bundle: {candidate}")
    if args.expected_manifest_sha256:
        manifest_path = bundle_root / "manifest.json"
        observed_manifest_sha256 = sha256(manifest_path)
        if observed_manifest_sha256 != args.expected_manifest_sha256:
            raise RuntimeError(
                f"detached manifest SHA256 drift: {observed_manifest_sha256}; "
                f"expected={args.expected_manifest_sha256}"
            )
    raw = load_input(args.input)
    runner = K100BundleRunner(args.bundle, args.device, not args.skip_full_payload_hash)
    logits, prediction, inference_seconds = runner.run(raw)
    atomic_save_npz(args.output, logits, prediction)
    fixed_gate = runner.fixed_prediction_gate(raw, prediction)
    receipt_path = args.receipt or args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt_path = receipt_path.resolve(strict=False)
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite receipt: {receipt_path}")
    receipt = {
        "schema": "phase11_k100_inference_receipt_v1",
        "status": "passed",
        "bundle": str(runner.bundle),
        "bundle_id": runner.manifest["bundle_id"],
        "manifest": runner.manifest_identity,
        "execution_kind": runner.kind,
        "runtime": runner.runtime,
        "timing": {
            "integrity_validation_seconds": runner.integrity_validation_seconds,
            "session_load_seconds": runner.session_load_seconds,
            "total_runner_load_seconds": runner.total_load_seconds,
            "single_inference_seconds": inference_seconds,
        },
        "input": {
            "path": str(args.input.resolve(strict=True)),
            "file_sha256": sha256(args.input.resolve(strict=True)),
            "array_sha256": array_sha256(raw),
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "finite": True,
            "preprocessing_applied_by_cli": False,
        },
        "output": {
            "path": str(args.output.resolve(strict=True)),
            "file_sha256": sha256(args.output.resolve(strict=True)),
            "logits_array_sha256": array_sha256(logits),
            "prediction_array_sha256": array_sha256(prediction),
            "logits_shape": list(logits.shape),
            "prediction_shape": list(prediction.shape),
            "logits_dtype": str(logits.dtype),
            "prediction_dtype": str(prediction.dtype),
            "finite_logits": True,
        },
        "fixed_sample_gate": fixed_gate,
        "evidence_boundary": {
            "strict_logit_equivalence_inferred_from_this_run": False,
            "native_kernel_precision_inferred_from_provider_placement": False,
            "image_digest_attestation_requires_PHASE11_K100_IMAGE_ID": True,
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
