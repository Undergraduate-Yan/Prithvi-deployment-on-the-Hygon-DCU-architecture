'Research implementation: export onnx fp32.'

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import onnx
import torch
import onnxruntime as ort
from torch import nn
from terratorch.datamodules.sen1floods11 import MEANS, STDS

from evaluate_fp32_baseline import METRICS_FILE, sha256
from train_fp32_baseline import BANDS, PROJECT_ROOT, build_datamodule, build_task


ONNX_DIR = PROJECT_ROOT / "models/onnx"
ONNX_FILE = ONNX_DIR / "prithvi300_upernet_fp32.onnx"
ONNX_TEMP_FILE = ONNX_DIR / "prithvi300_upernet_fp32.onnx.partial"
REPORT_FILE = ONNX_DIR / "prithvi300_upernet_fp32_report.json"
OPSET_VERSION = 17
MAX_ALLOWED_ABS_ERROR = 1.0e-4
MAX_ALLOWED_POOL_REWRITE_ERROR = 1.0e-5


class DeploymentWrapper(nn.Module):
    """Embed normalization and expose the segmentation logits tensor."""

    def __init__(self, task: nn.Module) -> None:
        super().__init__()
        self.task = task
        means = torch.tensor([MEANS[band] for band in BANDS], dtype=torch.float32)
        stds = torch.tensor([STDS[band] for band in BANDS], dtype=torch.float32)
        self.register_buffer("means", means.view(1, 6, 1, 1), persistent=True)
        self.register_buffer("stds", stds.view(1, 6, 1, 1), persistent=True)

    def forward(self, raw_scaled_image: torch.Tensor) -> torch.Tensor:
        normalized = (raw_scaled_image - self.means) / self.stds
        return self.task(normalized).output


class FixedAdaptiveAvgPool2d(nn.Module):
    """Exact adaptive pooling expressed with ONNX-supported operations.

    UPerNet's PPM receives a fixed 7x7 map for the fixed 224x224 input. PyTorch
    adaptive pooling uses floor/ceil window boundaries; explicitly spelling out
    those windows exports as Slice, ReduceMean, and Concat.
    """

    def __init__(
        self,
        output_size: int | tuple[int, int],
        input_size: tuple[int, int] = (7, 7),
    ) -> None:
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(int(value) for value in output_size)
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Expected pooling input {self.input_size}, found {tuple(x.shape[-2:])}"
            )

        input_h, input_w = self.input_size
        output_h, output_w = self.output_size
        rows: list[torch.Tensor] = []
        for row in range(output_h):
            h_start = (row * input_h) // output_h
            h_end = ((row + 1) * input_h + output_h - 1) // output_h
            cells: list[torch.Tensor] = []
            for column in range(output_w):
                w_start = (column * input_w) // output_w
                w_end = ((column + 1) * input_w + output_w - 1) // output_w
                cells.append(
                    x[:, :, h_start:h_end, w_start:w_end].mean(
                        dim=(-2, -1),
                        keepdim=True,
                    )
                )
            rows.append(torch.cat(cells, dim=3))
        return torch.cat(rows, dim=2)


def replace_unsupported_adaptive_pools(module: nn.Module) -> int:
    """Replace UPerNet PPM adaptive pools in-place and return their count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool2d):
            module._modules[name] = FixedAdaptiveAvgPool2d(child.output_size)
            replaced += 1
        else:
            replaced += replace_unsupported_adaptive_pools(child)
    return replaced


def load_frozen_reference() -> tuple[Path, str, DeploymentWrapper]:
    if not METRICS_FILE.is_file():
        raise FileNotFoundError(
            f"Strict FP32 metrics are missing: {METRICS_FILE}. "
            "Run evaluate_fp32_baseline.py first."
        )

    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    checkpoint_path = Path(metrics["checkpoint"])
    expected_sha256 = metrics["checkpoint_sha256"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint recorded by FP32 evaluation is missing: {checkpoint_path}")

    actual_sha256 = sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Checkpoint SHA256 no longer matches the strict FP32 baseline. "
            f"Expected {expected_sha256}, found {actual_sha256}."
        )

    task = build_task()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper = DeploymentWrapper(task.float().eval()).float().eval()
    return checkpoint_path, actual_sha256, wrapper


def representative_input() -> torch.Tensor:
    datamodule = build_datamodule()
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    sample = batch["image"][:1].contiguous().float()
    if sample.shape != (1, 6, 224, 224):
        raise ValueError(f"Unexpected representative input shape: {tuple(sample.shape)}")
    return sample


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    if ONNX_FILE.exists() and REPORT_FILE.exists():
        raise FileExistsError(
            "An ONNX FP32 artifact already exists. Move it to an archive directory "
            "before intentionally exporting a replacement:\n"
            f"  {ONNX_FILE}\n  {REPORT_FILE}"
        )
    if ONNX_FILE.exists() and not REPORT_FILE.exists():
        unverified_file = ONNX_FILE.with_name(ONNX_FILE.name + ".unverified")
        counter = 1
        while unverified_file.exists():
            unverified_file = ONNX_FILE.with_name(ONNX_FILE.name + f".unverified{counter}")
            counter += 1
        ONNX_FILE.rename(unverified_file)
        print(f"Moved an unverified earlier export to: {unverified_file}")
    if REPORT_FILE.exists() and not ONNX_FILE.exists():
        raise FileNotFoundError(
            f"Export report exists but its ONNX model is missing: {REPORT_FILE}"
        )
    if ONNX_TEMP_FILE.exists():
        stale_partial = ONNX_TEMP_FILE.with_name(ONNX_TEMP_FILE.name + ".stale")
        counter = 1
        while stale_partial.exists():
            stale_partial = ONNX_TEMP_FILE.with_name(ONNX_TEMP_FILE.name + f".stale{counter}")
            counter += 1
        ONNX_TEMP_FILE.rename(stale_partial)
        print(f"Moved an earlier partial export to: {stale_partial}")

    checkpoint_path, checkpoint_sha256, wrapper = load_frozen_reference()
    sample = representative_input()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for export equivalence validation")

    print("Validating the ONNX-compatible adaptive-pooling rewrite...")
    device = torch.device("cuda:0")
    wrapper = wrapper.to(device).eval()
    sample_cuda = sample.to(device)
    with torch.inference_mode():
        original_pytorch_output = wrapper(sample_cuda).cpu().numpy()

    replaced_pool_count = replace_unsupported_adaptive_pools(wrapper)
    if replaced_pool_count != 4:
        raise RuntimeError(
            f"Expected to replace 4 UPerNet adaptive pools, replaced {replaced_pool_count}"
        )

    with torch.inference_mode():
        rewritten_pytorch_output = wrapper(sample_cuda).cpu().numpy()
    pool_rewrite_max_abs_error = float(
        np.max(
            np.abs(
                original_pytorch_output.astype(np.float64)
                - rewritten_pytorch_output.astype(np.float64)
            )
        )
    )
    if pool_rewrite_max_abs_error >= MAX_ALLOWED_POOL_REWRITE_ERROR:
        raise AssertionError(
            "ONNX-compatible pooling rewrite changed the PyTorch output: "
            f"max abs error={pool_rewrite_max_abs_error}"
        )
    print(f"Pooling rewrite max absolute error: {pool_rewrite_max_abs_error:.8g}")

    wrapper = wrapper.cpu().eval()
    del sample_cuda
    torch.cuda.empty_cache()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA256: {checkpoint_sha256}")
    print(f"Export target: {ONNX_FILE}")
    print("Exporting ONNX FP32. This large model can take several minutes...")

    export_start = perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample,),
            str(ONNX_TEMP_FILE),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes=None,
            keep_initializers_as_inputs=False,
            dynamo=False,
        )
    export_seconds = perf_counter() - export_start

    if not ONNX_TEMP_FILE.is_file() or ONNX_TEMP_FILE.stat().st_size == 0:
        raise RuntimeError("ONNX exporter did not produce a non-empty file")

    print("Running ONNX structural validation...")
    onnx_model = onnx.load(str(ONNX_TEMP_FILE), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=False)
    node_count = len(onnx_model.graph.node)
    initializer_count = len(onnx_model.graph.initializer)
    del onnx_model

    # Export equivalence is deliberately checked on CPU. CUDA availability and
    # real provider placement are verified separately by
    # validate_onnx_fp32_cuda.py, so a failed CUDA load cannot silently turn
    # this check into a mislabeled CPU run.
    available_providers = ort.get_available_providers()
    print("Creating ONNX Runtime CPU consistency session...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(ONNX_TEMP_FILE),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    active_providers = session.get_providers()
    onnx_output = session.run(["logits"], {"image": sample.numpy()})[0]

    difference = np.abs(
        original_pytorch_output.astype(np.float64) - onnx_output.astype(np.float64)
    )
    denominator = np.maximum(np.abs(original_pytorch_output.astype(np.float64)), 1.0e-8)
    relative_difference = difference / denominator
    prediction_agreement = float(
        np.mean(
            np.argmax(original_pytorch_output, axis=1)
            == np.argmax(onnx_output, axis=1)
        )
    )

    max_abs_error = float(difference.max())
    mean_abs_error = float(difference.mean())
    max_rel_error = float(relative_difference.max())
    passed = max_abs_error < MAX_ALLOWED_ABS_ERROR

    if passed:
        ONNX_TEMP_FILE.replace(ONNX_FILE)
    artifact_file = ONNX_FILE if passed else ONNX_TEMP_FILE

    report = {
        "variant": "onnx_fp32",
        "precision": "fp32",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_file": str(artifact_file),
        "onnx_sha256": sha256(artifact_file),
        "model_size_mb": artifact_file.stat().st_size / 1024**2,
        "opset_version": OPSET_VERSION,
        "input_name": "image",
        "input_shape": [1, 6, 224, 224],
        "input_dtype": "float32",
        "input_semantics": "reflectance after constant_scale=0.0001; normalization embedded",
        "output_name": "logits",
        "output_shape": list(onnx_output.shape),
        "output_dtype": str(onnx_output.dtype),
        "node_count": node_count,
        "initializer_count": initializer_count,
        "adaptive_pool_replacements": replaced_pool_count,
        "pool_rewrite_max_abs_error": pool_rewrite_max_abs_error,
        "export_seconds": export_seconds,
        "onnxruntime_available_providers": available_providers,
        "onnxruntime_active_providers": active_providers,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "prediction_agreement": prediction_agreement,
        "required_max_abs_error": MAX_ALLOWED_ABS_ERROR,
        "consistency_passed": passed,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("ONNX FP32 EXPORT COMPLETE")
    print(f"Model size:            {report['model_size_mb']:.2f} MB")
    print(f"Max absolute error:    {max_abs_error:.8g}")
    print(f"Mean absolute error:   {mean_abs_error:.8g}")
    print(f"Prediction agreement:  {prediction_agreement:.6%}")
    print(f"Consistency threshold: < {MAX_ALLOWED_ABS_ERROR}")
    print(f"Consistency passed:    {passed}")
    print(f"Report:                {REPORT_FILE}")
    print("=" * 72)

    if not passed:
        raise AssertionError(
            f"ONNX FP32 max absolute error {max_abs_error} exceeds "
            f"the required threshold {MAX_ALLOWED_ABS_ERROR}."
        )


if __name__ == "__main__":
    main()
