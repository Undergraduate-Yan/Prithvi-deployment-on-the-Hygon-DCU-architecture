'Research implementation: train fp32 baseline.'

from __future__ import annotations

import json
import os
import platform
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

# Set these before importing PyTorch/OpenMP-dependent packages.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import albumentations as A
import lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
from terratorch.tasks import SemanticSegmentationTask


# ----------------------------- Fixed experiment -----------------------------
PROJECT_ROOT = Path(os.environ.get("PRITHVI_FLOOD_ROOT", "external/flood")).resolve()
DATA_ROOT = PROJECT_ROOT / "data/sen1floods11"
WEIGHT_FILE = (
    PROJECT_ROOT
    / "models/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"
)
RUN_ROOT = PROJECT_ROOT / "outputs/fp32_baseline_full"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"

SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 8
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 0.05

BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file is missing or empty: {path}")


def validate_inputs() -> None:
    require_file(WEIGHT_FILE)

    split_dir = DATA_ROOT / "v1.1/splits/flood_handlabeled"
    for split in ("train", "valid", "test"):
        require_file(split_dir / f"flood_{split}_data.txt")

    image_dir = DATA_ROOT / "v1.1/data/flood_events/HandLabeled/S2Hand"
    label_dir = DATA_ROOT / "v1.1/data/flood_events/HandLabeled/LabelHand"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Sen1Floods11 image or label directory is missing")


def build_datamodule() -> Sen1Floods11NonGeoDataModule:
    train_transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            ToTensorV2(),
        ]
    )

    return Sen1Floods11NonGeoDataModule(
        data_root=str(DATA_ROOT),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        constant_scale=0.0001,
        no_data_replace=0,
        no_label_replace=-1,
        use_metadata=False,
        bands=BANDS,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
    )


def build_task() -> SemanticSegmentationTask:
    return SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "prithvi_eo_v2_300",
            # TerraTorch 1.2.8 only reads ckpt_path when pretrained=True.
            "backbone_pretrained": True,
            "backbone_ckpt_path": str(WEIGHT_FILE),
            "backbone_bands": BANDS,
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [5, 11, 17, 23],
                },
                {"name": "ReshapeTokensToImage"},
            ],
            "decoder": "UperNetDecoder",
            "decoder_channels": 256,
            # Kept to match the official Prithvi configuration in the pinned
            # TerraTorch 1.2.8 environment. Its deprecation warning is benign.
            "decoder_scale_modules": True,
            "num_classes": 2,
            "head_dropout": 0.1,
            "rescale": True,
        },
        loss="ce",
        ignore_index=-1,
        class_names=["background", "water"],
        freeze_backbone=True,
        freeze_decoder=False,
        freeze_head=False,
        plot_on_val=False,
        optimizer="AdamW",
        lr=LEARNING_RATE,
        optimizer_hparams={"weight_decay": WEIGHT_DECAY},
        scheduler="CosineAnnealingLR",
        scheduler_hparams={"T_max": MAX_EPOCHS, "interval": "epoch"},
    )


def newest_checkpoint(pattern: str) -> Path | None:
    candidates = list(CHECKPOINT_DIR.glob(pattern))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    validate_inputs()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select the AutoDL GPU mode before training.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    L.seed_everything(SEED, workers=True)

    print("=" * 72)
    print("Prithvi FP32 baseline training")
    print(f"Project: {PROJECT_ROOT}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TerraTorch: {version('terratorch')}")
    print(f"Epochs: {MAX_EPOCHS}; batch size: {BATCH_SIZE}; seed: {SEED}")
    print("=" * 72)

    datamodule = build_datamodule()
    task = build_task()

    total_params = sum(parameter.numel() for parameter in task.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in task.parameters() if parameter.requires_grad
    )
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print(f"Frozen ratio: {(1 - trainable_params / total_params) * 100:.2f}%")

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best-epoch{epoch:02d}-step{step}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=1),
    ]

    loggers = [
        TensorBoardLogger(save_dir=RUN_ROOT, name="tensorboard"),
        CSVLogger(save_dir=RUN_ROOT, name="csv"),
    ]

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="16-mixed",
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=RUN_ROOT,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=2,
        enable_checkpointing=True,
    )

    # Automatically resume the newest interrupted full-run checkpoint.
    resume_checkpoint = newest_checkpoint("last*.ckpt")
    if resume_checkpoint is not None:
        print(f"Resuming from: {resume_checkpoint}")
    else:
        print("Starting a new full training run")

    started_at = datetime.now().astimezone()
    start_time = perf_counter()

    trainer.fit(
        model=task,
        datamodule=datamodule,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )

    elapsed_seconds = perf_counter() - start_time
    best_checkpoint = Path(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else None
    if best_checkpoint is None or not best_checkpoint.is_file():
        best_checkpoint = newest_checkpoint("best*.ckpt") or newest_checkpoint("last*.ckpt")
    if best_checkpoint is None:
        raise RuntimeError("Training ended without producing a checkpoint")

    print(f"Best checkpoint: {best_checkpoint}")
    print("Testing the best checkpoint on the fixed test split...")
    test_results = trainer.test(
        model=task,
        datamodule=datamodule,
        ckpt_path=str(best_checkpoint),
    )

    summary = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "status": "complete",
        "project_root": PROJECT_ROOT,
        "data_root": DATA_ROOT,
        "pretrained_weight": WEIGHT_FILE,
        "run_root": RUN_ROOT,
        "best_checkpoint": best_checkpoint,
        "best_val_loss": checkpoint_callback.best_model_score,
        "test_results": test_results,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "terratorch": version("terratorch"),
        "gpu": torch.cuda.get_device_name(0),
    }
    summary_file = RUN_ROOT / "run_summary.json"
    summary_file.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("TRAINING AND TESTING COMPLETE")
    print(f"Summary: {summary_file}")
    print(f"Best checkpoint: {best_checkpoint}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        failure_file = RUN_ROOT / "FAILED.txt"
        failure_file.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        print(f"Failure details saved to: {failure_file}")
        raise
