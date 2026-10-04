#!/usr/bin/env python3
'Research implementation: export cloud onnx.'
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--trainer-module", type=Path, required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verify-ort", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_trainer_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase7_cloud_trainer", path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module, extract_logits: Any) -> None:
        super().__init__()
        self.model = model
        self.extract_logits = extract_logits

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.extract_logits(self.model(image))


class StaticAdaptiveAvgPool2d(nn.Module):
    """ONNX-friendly exact expansion of AdaptiveAvgPool2d for a frozen shape.

    The legacy PyTorch exporter cannot lower adaptive pooling when the requested
    output is not an integer factor of the input.  Phase 7 exports one static
    224x224 contract, so the adaptive windows can be expanded deterministically
    using PyTorch's documented floor/ceil boundary rule.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        output_size: int | tuple[int | None, int | None],
    ) -> None:
        super().__init__()
        if isinstance(output_size, int):
            normalized_output: tuple[int | None, int | None] = (output_size, output_size)
        else:
            normalized_output = tuple(output_size)
        if len(normalized_output) != 2:
            raise ValueError(f"Invalid AdaptiveAvgPool2d output size: {output_size!r}")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.output_size = (
            self.input_size[0] if normalized_output[0] is None else int(normalized_output[0]),
            self.input_size[1] if normalized_output[1] is None else int(normalized_output[1]),
        )

    @staticmethod
    def _window(index: int, input_size: int, output_size: int) -> tuple[int, int]:
        start = (index * input_size) // output_size
        end = ((index + 1) * input_size + output_size - 1) // output_size
        return start, end

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for output_y in range(self.output_size[0]):
            start_y, end_y = self._window(output_y, self.input_size[0], self.output_size[0])
            columns: list[torch.Tensor] = []
            for output_x in range(self.output_size[1]):
                start_x, end_x = self._window(
                    output_x, self.input_size[1], self.output_size[1]
                )
                columns.append(
                    image[..., start_y:end_y, start_x:end_x].mean(
                        dim=(-2, -1), keepdim=True
                    )
                )
            rows.append(torch.cat(columns, dim=-1))
        return torch.cat(rows, dim=-2)


def rewrite_adaptive_pooling_for_static_onnx(
    wrapper: nn.Module,
    example: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Replace adaptive pools only after measuring their frozen input shapes."""

    observed_shapes: dict[int, set[tuple[int, int]]] = {}
    hooks: list[Any] = []
    adaptive_modules: list[tuple[str, nn.AdaptiveAvgPool2d]] = []
    for name, module in wrapper.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            adaptive_modules.append((name, module))

            def capture_shape(
                current_module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                module_id: int = id(module),
            ) -> None:
                shape = tuple(int(value) for value in inputs[0].shape[-2:])
                observed_shapes.setdefault(module_id, set()).add(shape)

            hooks.append(module.register_forward_pre_hook(capture_shape))

    with torch.inference_mode():
        original_reference = wrapper(example)
    for hook in hooks:
        hook.remove()

    rewrites: list[dict[str, Any]] = []

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.AdaptiveAvgPool2d):
                shapes = observed_shapes.get(id(child), set())
                if len(shapes) != 1:
                    raise RuntimeError(
                        f"Adaptive pool {qualified_name} had non-static input shapes: {shapes}"
                    )
                input_size = next(iter(shapes))
                replacement = StaticAdaptiveAvgPool2d(input_size, child.output_size)
                setattr(parent, child_name, replacement)
                rewrites.append(
                    {
                        "module": qualified_name,
                        "input_size": list(input_size),
                        "output_size": list(replacement.output_size),
                        "lowering": "static Slice + ReduceMean + Concat",
                        "boundary_rule": "start=floor(i*in/out), end=ceil((i+1)*in/out)",
                    }
                )
            else:
                replace_children(child, qualified_name)

    replace_children(wrapper)
    remaining = [
        name for name, module in wrapper.named_modules() if isinstance(module, nn.AdaptiveAvgPool2d)
    ]
    if remaining:
        raise RuntimeError(f"Adaptive pooling rewrite incomplete: {remaining}")
    with torch.inference_mode():
        rewritten_reference = wrapper(example)
    max_abs_error = float(
        torch.max(torch.abs(original_reference - rewritten_reference)).detach().cpu().item()
    )
    equivalent = bool(
        torch.allclose(original_reference, rewritten_reference, rtol=1e-6, atol=1e-6)
    )
    if not equivalent:
        raise RuntimeError(
            f"Static adaptive-pooling rewrite changed PyTorch output: max_abs_error={max_abs_error}"
        )
    return rewritten_reference, {
        "applied": bool(rewrites),
        "rewrite_count": len(rewrites),
        "rewrites": rewrites,
        "pytorch_equivalence": {
            "status": "PASS",
            "rtol": 1e-6,
            "atol": 1e-6,
            "max_abs_error": max_abs_error,
        },
    }


def main() -> int:
    args = parse_args()
    if args.opset < 17:
        raise ValueError("Phase-7 cloud export requires opset 17 or newer")
    for path in (
        args.checkpoint,
        args.backbone_checkpoint,
        args.data_manifest,
        args.trainer_module,
    ):
        path.resolve(strict=True)
    manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("Cloud training data manifest is not PASS")
    firewall = manifest.get("firewall", {})
    if firewall.get("formal_test_payload_access_count") != 0:
        raise RuntimeError("Formal cloud test payload access is not zero")
    class_weights = [float(value) for value in manifest["training_class_weights"]]

    trainer_module = load_trainer_module(args.trainer_module)
    build_args = Namespace(
        backbone_checkpoint=args.backbone_checkpoint,
        learning_rate=5e-5,
        weight_decay=0.05,
        max_epochs=50,
    )
    task = trainer_module.build_task(build_args, class_weights)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    task.load_state_dict(state_dict, strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but no K100 is visible")
    wrapper = LogitsWrapper(task.model, trainer_module.extract_logits).eval().to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    example = torch.rand((1, 6, 224, 224), generator=generator, device=device)
    reference, static_rewrite = rewrite_adaptive_pooling_for_static_onnx(wrapper, example)
    if reference.shape != (1, 4, 224, 224) or not bool(torch.isfinite(reference).all().item()):
        raise RuntimeError(f"Invalid PyTorch reference output: {tuple(reference.shape)}")

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        example,
        args.output_onnx,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - started
    if not args.output_onnx.is_file() or args.output_onnx.stat().st_size == 0:
        raise RuntimeError("ONNX exporter did not create a complete file")

    import onnx

    onnx.checker.check_model(str(args.output_onnx), full_check=True)
    ort_result: dict[str, Any] = {"executed": False}
    if args.verify_ort:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(args.output_onnx), providers=providers)
        ort_output = session.run(["logits"], {"input": example.detach().cpu().numpy()})[0]
        reference_np = reference.detach().cpu().numpy()
        max_abs_error = float(np.max(np.abs(reference_np - ort_output)))
        if not math.isfinite(max_abs_error):
            raise RuntimeError("Non-finite export-side ONNX Runtime error")
        ort_result = {
            "executed": True,
            "providers": session.get_providers(),
            "max_abs_error": max_abs_error,
            "output_shape": list(ort_output.shape),
            "finite": bool(np.isfinite(ort_output).all()),
        }

    report = {
        "schema": "phase7_cloud_onnx_export_report_v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(args.checkpoint.resolve(strict=True)),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        },
        "backbone_checkpoint_sha256": sha256_file(args.backbone_checkpoint),
        "training_data_manifest_sha256": sha256_file(args.data_manifest),
        "trainer_module_sha256": sha256_file(args.trainer_module),
        "onnx": {
            "path": str(args.output_onnx.resolve(strict=True)),
            "size_bytes": args.output_onnx.stat().st_size,
            "sha256": sha256_file(args.output_onnx),
            "opset": args.opset,
            "checker": "PASS",
            "static_shape": True,
            "input": {"name": "input", "dtype": "FP32", "shape": [1, 6, 224, 224]},
            "output": {"name": "logits", "dtype": "FP32", "shape": [1, 4, 224, 224]},
            "normalization_inside_graph": False,
            "output_class_count": 4,
        },
        "pytorch_reference": {
            "device": str(device),
            "finite": True,
            "output_shape": list(reference.shape),
        },
        "onnx_static_equivalence_rewrite": static_rewrite,
        "export_seconds": export_seconds,
        "onnxruntime_verification": ort_result,
        "firewall": {
            "calibration_payload_access_count": 0,
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
        },
    }
    write_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
