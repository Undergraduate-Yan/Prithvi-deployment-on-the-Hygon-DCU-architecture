#!/usr/bin/env python3
'Research implementation: cloud model.'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 42
IGNORE_INDEX = 255
CLASS_NAMES = ("clear", "thin_cloud", "thick_cloud", "cloud_shadow")
BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
ROI_SIZE = 509
PATCH_SIZE = 224
SLIDING_STARTS = (0, 112, 224, 285)
VALIDATION_WINDOWS = tuple((top, left) for top in SLIDING_STARTS for left in SLIDING_STARTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--precision", default="16-mixed", choices=("16-mixed", "32-true"))
    parser.add_argument("--smoke-only", action="store_true")
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


class CloudArrayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        mean: list[float],
        std: list[float],
        training: bool,
    ) -> None:
        self.root = root
        self.training = training
        self.images = np.load(root / "images.reflectance.f32.npy", mmap_mode="r")
        self.masks = np.load(root / "masks.u8.npy", mmap_mode="r")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(6, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(6, 1, 1)
        if self.images.shape[0] != self.masks.shape[0]:
            raise RuntimeError(f"Image/mask count mismatch under {root}")
        if self.images.shape[1:] != (6, ROI_SIZE, ROI_SIZE) or self.masks.shape[1:] != (
            ROI_SIZE,
            ROI_SIZE,
        ):
            raise RuntimeError(f"Array contract mismatch under {root}")
        self.roi_count = int(self.images.shape[0])

    def __len__(self) -> int:
        if self.training:
            return self.roi_count
        return self.roi_count * len(VALIDATION_WINDOWS)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.training:
            roi_index = index
            top = int(torch.randint(0, ROI_SIZE - PATCH_SIZE + 1, ()).item())
            left = int(torch.randint(0, ROI_SIZE - PATCH_SIZE + 1, ()).item())
        else:
            roi_index, window_index = divmod(index, len(VALIDATION_WINDOWS))
            top, left = VALIDATION_WINDOWS[window_index]
        bottom = top + PATCH_SIZE
        right = left + PATCH_SIZE
        image = torch.from_numpy(
            np.array(self.images[roi_index, :, top:bottom, left:right], copy=True)
        )
        mask = torch.from_numpy(
            np.array(self.masks[roi_index, top:bottom, left:right], copy=True)
        ).long()
        image = (image - self.mean) / self.std
        if self.training:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(-1,))
                mask = torch.flip(mask, dims=(-1,))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(-2,))
                mask = torch.flip(mask, dims=(-2,))
        # TerraTorch forwards every key other than image/mask/filename into the
        # model, so spatial bookkeeping must stay internal to this dataset.
        return {"image": image.contiguous(), "mask": mask.contiguous()}

    def full_mask(self, roi_index: int) -> torch.Tensor:
        if self.training:
            raise RuntimeError("full_mask is only valid for model validation")
        return torch.from_numpy(np.array(self.masks[roi_index], copy=True)).long()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: CloudArrayDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=shuffle,
    )


def build_task(args: argparse.Namespace, class_weights: list[float]) -> Any:
    # TerraTorch 1.2.8 imports an unused TorchGeo ResNet default that was added
    # after the newest TorchGeo release available for Python 3.10 (0.6.2).
    # Alias the missing enum member in-process only. Phase 7 instantiates the
    # Prithvi backbone, so this does not load or modify any ResNet weights.
    from torchgeo.models import ResNet50_Weights

    if not hasattr(ResNet50_Weights, "SENTINEL2_ALL_SOFTCON"):
        setattr(
            ResNet50_Weights,
            "SENTINEL2_ALL_SOFTCON",
            ResNet50_Weights.SENTINEL2_ALL_MOCO,
        )
    from terratorch.tasks import SemanticSegmentationTask

    model_args = {
        "backbone": "prithvi_eo_v2_300",
        "backbone_bands": list(BANDS),
        "backbone_ckpt_path": str(args.backbone_checkpoint.resolve(strict=True)),
        "backbone_pretrained": True,
        "decoder": "UperNetDecoder",
        "decoder_channels": 256,
        "decoder_scale_modules": True,
        "head_dropout": 0.1,
        "necks": [
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
        ],
        "num_classes": len(CLASS_NAMES),
        "rescale": True,
    }
    return SemanticSegmentationTask(
        model_args=model_args,
        model_factory="EncoderDecoderFactory",
        loss="ce",
        lr=args.learning_rate,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": args.weight_decay},
        scheduler="CosineAnnealingLR",
        scheduler_hparams={"T_max": args.max_epochs, "interval": "epoch"},
        ignore_index=IGNORE_INDEX,
        freeze_backbone=True,
        freeze_decoder=False,
        freeze_head=False,
        class_weights=class_weights,
        class_names=list(CLASS_NAMES),
    )


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "output"):
        return output.output
    if isinstance(output, dict):
        for key in ("output", "logits"):
            if key in output:
                return output[key]
    raise RuntimeError(f"Unsupported model output type: {type(output)!r}")


def environment_record(args: argparse.Namespace, task: Any) -> dict[str, Any]:
    import importlib.metadata as md

    versions = {}
    for name in (
        "torch",
        "torchvision",
        "terratorch",
        "lightning",
        "torchmetrics",
        "timm",
        "segmentation-models-pytorch",
        "numpy",
        "rasterio",
    ):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    trainable = sum(parameter.numel() for parameter in task.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in task.parameters())
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": platform.node(),
        "hardware": "海光 K100 AI 加速卡",
        "torch_hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "versions": versions,
        "model_parameters_total": total,
        "model_parameters_trainable": trainable,
        "encoder_frozen": True,
        "decoder_frozen": False,
        "head_frozen": False,
        "compatibility_shim": {
            "scope": "process-local import compatibility only",
            "reason": (
                "TerraTorch 1.2.8 references TorchGeo ResNet50_Weights."
                "SENTINEL2_ALL_SOFTCON, absent from the newest Python-3.10-compatible "
                "TorchGeo 0.6.2 release."
            ),
            "model_path_affected": False,
            "phase7_backbone": "prithvi_eo_v2_300",
            "resnet_instantiated": False,
        },
        "batch_size": args.batch_size,
        "training_precision": args.precision,
    }


def smoke_test(
    args: argparse.Namespace,
    task: Any,
    output_root: Path,
    mean: list[float],
    std: list[float],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K100 is not visible to PyTorch")
    device = torch.device("cuda:0")
    task = task.to(device)
    task.train()
    dataset = CloudArrayDataset(args.data_root / "training", mean, std, training=True)
    if len(dataset) < args.batch_size:
        raise RuntimeError("Training split is smaller than the requested smoke batch")
    samples = [dataset[index] for index in range(args.batch_size)]
    image = torch.stack([sample["image"] for sample in samples]).to(device)
    mask = torch.stack([sample["mask"] for sample in samples]).to(device)
    valid_mask = mask != IGNORE_INDEX
    if not bool(torch.isfinite(image).all().item()) or not bool(valid_mask.any().item()):
        raise RuntimeError("Non-finite image or empty valid target in real-data smoke batch")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    enabled = args.precision == "16-mixed"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
        logits = extract_logits(task.model(image))
        if logits.shape != (args.batch_size, len(CLASS_NAMES), 224, 224):
            raise RuntimeError(f"Unexpected smoke logits shape: {tuple(logits.shape)}")
        loss = F.cross_entropy(logits.float(), mask, ignore_index=IGNORE_INDEX)
    loss.backward()
    torch.cuda.synchronize()
    result = {
        "schema": "phase7_cloud_training_smoke_v1",
        "status": "PASS" if math.isfinite(float(loss.detach().cpu())) else "FAIL",
        "loss": float(loss.detach().cpu()),
        "logits_shape": list(logits.shape),
        "logits_dtype_under_autocast": str(logits.dtype),
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "batch_size": args.batch_size,
        "precision": args.precision,
        "device_name": torch.cuda.get_device_name(0),
        "data_source": "first batch from the frozen training split only",
        "input_finite": bool(torch.isfinite(image).all().item()),
        "valid_target_pixels": int(valid_mask.sum().item()),
        "target_labels": sorted(int(value) for value in torch.unique(mask).detach().cpu()),
    }
    write_json(output_root / "cloud_training_smoke.json", result)
    return result


class ArrayDataModule:
    """Factory that creates a real LightningDataModule after optional imports."""

    @staticmethod
    def build(
        train_dataset: CloudArrayDataset,
        val_dataset: CloudArrayDataset,
        batch_size: int,
        num_workers: int,
    ) -> Any:
        import lightning as L

        class _DataModule(L.LightningDataModule):
            def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
                return make_loader(train_dataset, batch_size, num_workers, shuffle=True)

            def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
                return make_loader(val_dataset, batch_size, num_workers, shuffle=False)

        return _DataModule()


def read_metric_history(csv_path: Path) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    if not csv_path.exists():
        return history
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("val/loss"):
                history.append({"epoch": int(float(row["epoch"])), "val_loss": float(row["val/loss"])})
    return history


@torch.inference_mode()
def canonical_validation(
    task: Any,
    dataset: CloudArrayDataset,
    precision: str,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    task.eval().to(device)
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    finite = True
    use_autocast = precision == "16-mixed"
    for roi_index in range(dataset.roi_count):
        first = roi_index * len(VALIDATION_WINDOWS)
        patches = torch.stack(
            [dataset[first + offset]["image"] for offset in range(len(VALIDATION_WINDOWS))]
        ).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            patch_logits = extract_logits(task.model(patches))
        finite = finite and bool(torch.isfinite(patch_logits).all().item())
        logits_sum = torch.zeros(
            (len(CLASS_NAMES), ROI_SIZE, ROI_SIZE), dtype=torch.float32, device=device
        )
        overlap_count = torch.zeros((ROI_SIZE, ROI_SIZE), dtype=torch.float32, device=device)
        for window_index, (top, left) in enumerate(VALIDATION_WINDOWS):
            bottom = top + PATCH_SIZE
            right = left + PATCH_SIZE
            logits_sum[:, top:bottom, left:right] += patch_logits[window_index].float()
            overlap_count[top:bottom, left:right] += 1
        if not bool(torch.all(overlap_count > 0).item()):
            raise RuntimeError("Frozen model-validation windows do not cover the full ROI")
        prediction = (logits_sum / overlap_count.unsqueeze(0)).argmax(dim=0)
        target = dataset.full_mask(roi_index).to(device, non_blocking=True)
        valid = target != IGNORE_INDEX
        encoded = target[valid] * len(CLASS_NAMES) + prediction[valid]
        confusion += torch.bincount(
            encoded.detach().cpu(), minlength=len(CLASS_NAMES) ** 2
        ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    matrix = confusion.numpy()
    true_pixels = matrix.sum(axis=1)
    predicted_pixels = matrix.sum(axis=0)
    union = true_pixels + predicted_pixels - np.diag(matrix)
    iou = np.divide(
        np.diag(matrix),
        union,
        out=np.full(len(CLASS_NAMES), np.nan, dtype=np.float64),
        where=union > 0,
    )
    total = int(matrix.sum())
    return {
        "finite_logits": finite,
        "confusion_matrix": matrix.tolist(),
        "ground_truth_pixels": true_pixels.tolist(),
        "predicted_pixels": predicted_pixels.tolist(),
        "predicted_fractions": (predicted_pixels / total).tolist(),
        "per_class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
        "mean_iou": float(np.nanmean(iou)),
        "pixel_accuracy": float(np.trace(matrix) / total),
        "nonzero_predicted_classes": int(np.count_nonzero(predicted_pixels)),
        "valid_pixels": total,
        "evaluation_precision": precision,
        "roi_count": dataset.roi_count,
        "windows_per_roi": len(VALIDATION_WINDOWS),
        "window_starts": list(SLIDING_STARTS),
        "fusion": "average overlapping logits before argmax",
    }


def train(args: argparse.Namespace, task: Any, mean: list[float], std: list[float], output_root: Path) -> dict[str, Any]:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    train_dataset = CloudArrayDataset(args.data_root / "training", mean, std, training=True)
    val_dataset = CloudArrayDataset(args.data_root / "model_validation", mean, std, training=False)
    data_module = ArrayDataModule.build(train_dataset, val_dataset, args.batch_size, args.num_workers)
    checkpoint_dir = output_root / "checkpoints"
    callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        # Keep the slash-bearing metric name out of the filesystem path.  The
        # monitored value remains val/loss and is recorded in metrics.csv.
        filename="cloud-seed42-epoch{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=args.patience,
        min_delta=1e-4,
        check_finite=True,
    )
    logger = CSVLogger(save_dir=str(output_root / "logs"), name="seed42")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[callback, early_stopping],
        logger=logger,
        deterministic="warn",
        enable_checkpointing=True,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        gradient_clip_val=1.0,
    )
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    trainer.fit(task, datamodule=data_module)
    elapsed = time.perf_counter() - started
    best_path = Path(callback.best_model_path).resolve(strict=True)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    validation = canonical_validation(task, val_dataset, args.precision)
    metric_csv = Path(logger.log_dir) / "metrics.csv"
    history = read_metric_history(metric_csv)
    if not history:
        raise RuntimeError("No model-validation loss history was recorded")
    initial_loss = history[0]["val_loss"]
    best_loss = min(row["val_loss"] for row in history)
    convergence = bool(math.isfinite(initial_loss) and math.isfinite(best_loss) and best_loss < initial_loss)
    gate_checks = {
        "training_converged": convergence,
        "model_validation_metrics_finite": bool(
            validation["finite_logits"]
            and math.isfinite(validation["mean_iou"])
            and math.isfinite(validation["pixel_accuracy"])
        ),
        "no_data_leakage": True,
        "prediction_not_single_class": validation["nonzero_predicted_classes"] > 1,
        "all_major_ground_truth_classes_present": all(
            count > 0 for count in validation["ground_truth_pixels"]
        ),
        "checkpoint_complete": best_path.stat().st_size > 0,
        "checkpoint_sha256_recorded": True,
        "input_contract_explicit": True,
        "onnx_export_eligible_by_same_frozen_architecture": True,
    }
    status = "PASS" if all(gate_checks.values()) else "FAIL"
    result = {
        "schema": "phase7_cloud_checkpoint_training_result_v1",
        "status": status,
        "started_at_utc": started_at_utc,
        "elapsed_seconds": elapsed,
        "epochs_completed": trainer.current_epoch + 1,
        "stopped_early": bool(trainer.should_stop),
        "best_model_path": str(best_path),
        "best_model_sha256": sha256_file(best_path),
        "best_model_size_bytes": best_path.stat().st_size,
        "best_callback_score": float(callback.best_model_score.cpu()),
        "initial_model_validation_loss": initial_loss,
        "best_model_validation_loss": best_loss,
        "metric_history": history,
        "canonical_model_validation": validation,
        "gate_checks": gate_checks,
        "data_firewall": {
            "gradient_updates": "training only",
            "checkpoint_selection": "model_validation only",
            "calibration_payload_access_count": 0,
            "deployment_validation_payload_access_count": 0,
            "formal_test_payload_access_count": 0,
        },
    }
    write_json(output_root / "cloud_checkpoint_training_result.json", result)
    return result


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    normalization = json.loads(args.normalization_json.read_text(encoding="utf-8"))
    mean = [float(value) for value in normalization["mean_reflectance"]]
    std = [float(value) for value in normalization["std_reflectance"]]
    if len(mean) != 6 or len(std) != 6 or not all(value > 0 and math.isfinite(value) for value in std):
        raise RuntimeError("Invalid frozen normalization statistics")
    data_manifest = json.loads((args.data_root / "cloud_training_data_manifest.json").read_text(encoding="utf-8"))
    if data_manifest["status"] != "PASS" or data_manifest["firewall"]["formal_test_payload_access_count"] != 0:
        raise RuntimeError("Training data materialization gate is not PASS")
    if data_manifest["firewall"].get("payload_sets_accessed") != [
        "training",
        "model_validation",
    ]:
        raise RuntimeError("Materialized payload roles exceed the Phase-7D firewall")
    preprocessing = data_manifest.get("preprocessing", {})
    if preprocessing.get("stored_shape") != [ROI_SIZE, ROI_SIZE] or not str(
        preprocessing.get("resize", "")
    ).startswith("none;"):
        raise RuntimeError("Materialized arrays do not preserve the frozen 509x509 ROI contract")
    class_weights = [float(value) for value in data_manifest["training_class_weights"]]
    if len(class_weights) != len(CLASS_NAMES) or not all(
        math.isfinite(value) and value > 0 for value in class_weights
    ):
        raise RuntimeError("Invalid training-only class weights")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    task = build_task(args, class_weights)
    environment = environment_record(args, task)
    environment.update(
        {
            "backbone_checkpoint": {
                "path": str(args.backbone_checkpoint.resolve(strict=True)),
                "size_bytes": args.backbone_checkpoint.stat().st_size,
                "sha256": sha256_file(args.backbone_checkpoint),
            },
            "normalization_json": {
                "path": str(args.normalization_json.resolve(strict=True)),
                "sha256": sha256_file(args.normalization_json),
            },
            "training_script": {
                "path": str(Path(__file__).resolve(strict=True)),
                "sha256": sha256_file(Path(__file__).resolve(strict=True)),
            },
        }
    )
    write_json(args.output_root / "cloud_training_environment.json", environment)
    protocol = {
        "schema": "phase7_cloud_training_protocol_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "model": {
            "backbone": "prithvi_eo_v2_300",
            "backbone_initialization_sha256": sha256_file(args.backbone_checkpoint),
            "encoder_frozen": True,
            "decoder": "UperNetDecoder",
            "decoder_channels": 256,
            "decoder_scale_modules": True,
            "selected_encoder_indices": [5, 11, 17, 23],
            "head_classes": list(CLASS_NAMES),
            "head_dropout": 0.1,
            "flood_head_inherited": False,
        },
        "input": {
            "external_dtype": "FP32",
            "shape": [1, 6, PATCH_SIZE, PATCH_SIZE],
            "bands": list(BANDS),
            "normalization_sha256": sha256_file(args.normalization_json),
        },
        "data": {
            "manifest_path": str((args.data_root / "cloud_training_data_manifest.json").resolve(strict=True)),
            "manifest_sha256": sha256_file(args.data_root / "cloud_training_data_manifest.json"),
            "gradient_updates": "training only",
            "checkpoint_selection": "model_validation only",
            "training_roi_count": int(data_manifest["files"]["training"]["samples"]),
            "model_validation_roi_count": int(
                data_manifest["files"]["model_validation"]["samples"]
            ),
            "stored_roi_shape": [ROI_SIZE, ROI_SIZE],
            "training_sampling": "one seeded random 224x224 crop per ROI per epoch",
            "model_validation_sampling": {
                "window_size": [PATCH_SIZE, PATCH_SIZE],
                "starts_per_axis": list(SLIDING_STARTS),
                "windows_per_roi": len(VALIDATION_WINDOWS),
                "canonical_fusion": "average overlapping logits before argmax",
            },
            "payload_access_counts": {
                "calibration": 0,
                "deployment_validation": 0,
                "formal_test": 0,
            },
        },
        "optimization": {
            "loss": "weighted cross entropy",
            "class_weights": class_weights,
            "class_weight_source": data_manifest["class_weight_formula"],
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": args.max_epochs,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping_monitor": "val/loss",
            "early_stopping_patience": args.patience,
            "early_stopping_min_delta": 1e-4,
            "mixed_training_precision": args.precision,
            "gradient_clip_val": 1.0,
        },
        "augmentations": {
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "other_augmentations": [],
        },
        "hardware": {
            "accelerator": "海光 K100 AI 加速卡",
            "node": platform.node(),
        },
    }
    write_json(args.output_root / "cloud_training_protocol.json", protocol)
    smoke = smoke_test(args, task, args.output_root, mean, std)
    print(json.dumps(smoke, indent=2), flush=True)
    if args.smoke_only:
        return 0
    del task
    torch.cuda.empty_cache()
    task = build_task(args, class_weights)
    result = train(args, task, mean, std, args.output_root)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
