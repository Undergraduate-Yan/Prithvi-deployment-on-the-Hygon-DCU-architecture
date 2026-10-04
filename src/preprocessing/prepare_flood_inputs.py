#!/usr/bin/env python3
'Research implementation: prepare flood inputs.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio


BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
BAND_INDICES = np.asarray((1, 2, 3, 8, 11, 12), dtype=np.int64)
OUTPUT_SHAPE = (6, 224, 224)
TARGET_SHAPE = (224, 224)
CONSTANT_SCALE = 0.0001
EXPECTED_OPENCV = "4.11.0"
REPLAY_MAX_ABS_TOLERANCE = 2e-6
REPLAY_MEAN_ABS_TOLERANCE = 1e-7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV manifest: {path}")
    required = {
        "sample_id",
        "source_split",
        "image_path",
        "target_path",
        "image_sha256",
        "target_sha256",
        "sample_sha256",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def validate_source_identity(row: dict[str, str]) -> tuple[Path, Path]:
    image_path = Path(row["image_path"])
    target_path = Path(row["target_path"])
    for kind, path, expected in (
        ("image", image_path, row["image_sha256"]),
        ("target", target_path, row["target_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{row['sample_id']} {kind} SHA-256 mismatch: {actual} != {expected}"
            )
    combined = hashlib.sha256(
        bytes.fromhex(row["image_sha256"]) + bytes.fromhex(row["target_sha256"])
    ).hexdigest()
    if combined != row["sample_sha256"]:
        raise ValueError(f"{row['sample_id']} combined source SHA-256 mismatch")
    return image_path, target_path


def read_masked(path: Path, fill_value: float) -> np.ndarray:
    with rasterio.open(path) as dataset:
        data = dataset.read(masked=True)
    if np.ma.isMaskedArray(data):
        data = data.filled(fill_value)
    return np.asarray(data)


def preprocess(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    image_path, target_path = validate_source_identity(row)
    image = read_masked(image_path, 0)
    target = read_masked(target_path, -1)
    if image.shape != (13, 512, 512):
        raise ValueError(f"{row['sample_id']} image shape is {image.shape}, expected (13,512,512)")
    if target.shape != (1, 512, 512):
        raise ValueError(f"{row['sample_id']} target shape is {target.shape}, expected (1,512,512)")

    # TerraTorch 0.99.7 Sen1Floods11NonGeo followed by Albumentations 1.4.10.
    image_hwc = np.moveaxis(image, 0, -1)[..., BAND_INDICES]
    image_hwc = image_hwc.astype(np.float32) * CONSTANT_SCALE
    # Albumentations 1.4.10's maybe_process_in_chunks splits >4-channel images.
    resized_chunks = [
        cv2.resize(image_hwc[..., index : index + 4], (224, 224), interpolation=cv2.INTER_LINEAR)
        for index in range(0, image_hwc.shape[-1], 4)
    ]
    image_hwc = np.dstack(resized_chunks)
    target_hw = cv2.resize(target[0], (224, 224), interpolation=cv2.INTER_NEAREST)
    image_chw = np.ascontiguousarray(np.moveaxis(image_hwc, -1, 0), dtype=np.float32)
    target_hw = np.ascontiguousarray(target_hw, dtype=np.int64)
    if image_chw.shape != OUTPUT_SHAPE or target_hw.shape != TARGET_SHAPE:
        raise AssertionError((image_chw.shape, target_hw.shape))
    if not np.isfinite(image_chw).all():
        raise ValueError(f"{row['sample_id']} preprocessed image contains non-finite values")
    if not set(np.unique(target_hw).tolist()).issubset({-1, 0, 1}):
        raise ValueError(f"{row['sample_id']} preprocessed target has unexpected labels")
    return image_chw, target_hw


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assert_locked_split_contract(
    split_manifest: dict[str, Any],
    selection_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> None:
    if split_manifest.get("schema") != "journal_independent_splits_v2":
        raise ValueError("split manifest is not journal_independent_splits_v2")
    gates = split_manifest.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("not every split-manifest leakage gate is true")
    sets = split_manifest["sets"]
    selection_ids = [row["sample_id"] for row in selection_rows]
    reference_ids = [row["sample_id"] for row in reference_rows]
    if selection_ids != sets["configuration_validation"]["sample_ids"]:
        raise ValueError("selection CSV differs from locked configuration-validation order")
    if reference_ids != sets["frozen_test"]["sample_ids"]:
        raise ValueError("reference CSV differs from locked frozen-test order")
    if len(selection_ids) != 64 or len(reference_ids) != 90:
        raise ValueError("expected exactly 64 configuration-validation and 90 frozen-test IDs")

    fields = ("sample_ids", "image_sha256", "sample_sha256")
    names = sorted(sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = sets[left_name]
            right = sets[right_name]
            for field in fields:
                left_values = set(left[field] if field == "sample_ids" else left[field].values())
                right_values = set(right[field] if field == "sample_ids" else right[field].values())
                overlap = left_values.intersection(right_values)
                if overlap:
                    raise ValueError(
                        f"{left_name}/{right_name} overlap by {field}: {sorted(overlap)[:3]}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--frozen-test-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--frozen-test-evidence", required=True, type=Path)
    parser.add_argument("--frozen-first-sample-tensor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    if cv2.__version__ != EXPECTED_OPENCV:
        raise RuntimeError(
            f"OpenCV {cv2.__version__} is active; exact Stage-1 runtime requires {EXPECTED_OPENCV}"
        )

    selection_rows = read_csv(args.selection_manifest.resolve())
    reference_rows = read_csv(args.frozen_test_manifest.resolve())
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    evidence = json.loads(args.frozen_test_evidence.read_text(encoding="utf-8"))
    assert_locked_split_contract(split_manifest, selection_rows, reference_rows)
    if any(row["source_split"] != "train" for row in selection_rows):
        raise ValueError("configuration-validation contains a non-training sample")
    if any(row["source_split"] != "test" for row in reference_rows):
        raise ValueError("frozen-test manifest contains a non-test sample")

    evidence_samples = evidence.get("samples", [])
    evidence_ids = [item["sample_id"] for item in evidence_samples]
    reference_ids = [row["sample_id"] for row in reference_rows]
    if evidence.get("sample_count") != 90 or evidence_ids != reference_ids:
        raise ValueError("prior frozen-test evidence identity/order differs from locked 90 IDs")

    # TerraTorch 0.99.7 filters a globally sorted file list, while the historical
    # evaluator assigned IDs from split-file line order. Reconstruct that legacy
    # runtime order only to audit the old evidence; new Stage-1 samples are always
    # paired directly by ID/path from their locked CSV manifest.
    runtime_rows = sorted(reference_rows, key=lambda row: row["image_path"])
    if [row["sample_id"] for row in runtime_rows] == reference_ids:
        raise ValueError("expected historical runtime-order/assigned-ID mismatch was not observed")
    replay_rows: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    target_hashes: list[str] = []
    valid_pixels = 0
    for row, expected in zip(runtime_rows, evidence_samples):
        image, target = preprocess(row)
        image_hash = sha256_array(image[np.newaxis, ...])
        target_hash = sha256_array(target[np.newaxis, ...])
        raw_hashes.append(image_hash)
        target_hashes.append(target_hash)
        valid = int(np.count_nonzero((target >= 0) & (target < 2)))
        valid_pixels += valid
        replay_rows.append(
            {
                "historically_assigned_sample_id": expected["sample_id"],
                "actual_source_sample_id": row["sample_id"],
                "raw_hash_exact_on_current_platform": image_hash
                == expected["raw_scaled_fp32_sha256"],
                "target_hash_exact": target_hash == expected["target_int64_sha256"],
                "valid_pixels_exact": valid == expected["valid_pixels"],
            }
        )
    if not all(item["target_hash_exact"] and item["valid_pixels_exact"] for item in replay_rows):
        failed = [item for item in replay_rows if not item["target_hash_exact"]]
        raise ValueError(f"frozen-test target/runtime-order replay mismatch: {failed[:3]}")
    if valid_pixels != evidence["valid_pixels"]:
        raise ValueError("frozen-test replay valid-pixel total mismatch")
    raw_list_hash = hashlib.sha256(("\n".join(raw_hashes) + "\n").encode("utf-8")).hexdigest()
    target_list_hash = hashlib.sha256(("\n".join(target_hashes) + "\n").encode("utf-8")).hexdigest()
    if target_list_hash != evidence["target_hash_list_sha256"]:
        raise ValueError("frozen-test target hash-list digest mismatch")

    import torch

    frozen_tensor = torch.load(args.frozen_first_sample_tensor.resolve(), map_location="cpu")
    historical_raw = frozen_tensor["raw_image"].detach().cpu().contiguous().numpy()
    historical_target = frozen_tensor["target"].detach().cpu().contiguous().numpy()
    replay_image, replay_target = preprocess(runtime_rows[0])
    replay_image = replay_image[np.newaxis, ...]
    replay_target = replay_target[np.newaxis, ...]
    absolute_difference = np.abs(historical_raw.astype(np.float64) - replay_image.astype(np.float64))
    replay_max_abs = float(absolute_difference.max())
    replay_mean_abs = float(absolute_difference.mean())
    denominator = float(np.linalg.norm(historical_raw.astype(np.float64).ravel()))
    replay_relative_l2 = float(
        np.linalg.norm((historical_raw.astype(np.float64) - replay_image).ravel())
        / max(denominator, np.finfo(np.float64).tiny)
    )
    if replay_max_abs > REPLAY_MAX_ABS_TOLERANCE or replay_mean_abs > REPLAY_MEAN_ABS_TOLERANCE:
        raise ValueError("first-sample cross-platform input replay exceeds locked tolerance")
    if not np.array_equal(historical_target, replay_target):
        raise ValueError("first-sample historical target tensor does not replay exactly")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    selection_samples: list[dict[str, Any]] = []
    for row in selection_rows:
        image, target = preprocess(row)
        inputs.append(image)
        targets.append(target)
        selection_samples.append(
            {
                "sample_id": row["sample_id"],
                "source_split": row["source_split"],
                "region": row.get("region", ""),
                "coverage_bin": row.get("coverage_bin", ""),
                "raw_scaled_fp32_sha256": sha256_array(image[np.newaxis, ...]),
                "target_int64_sha256": sha256_array(target[np.newaxis, ...]),
                "source_image_sha256": row["image_sha256"],
                "source_target_sha256": row["target_sha256"],
                "source_sample_sha256": row["sample_sha256"],
            }
        )

    input_array = np.ascontiguousarray(np.stack(inputs), dtype=np.float32)
    target_array = np.ascontiguousarray(np.stack(targets), dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in selection_rows], dtype="<U64")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = args.output_dir / "configuration_validation_inputs.npz"
    np.savez_compressed(pack_path, inputs=input_array, targets=target_array, sample_ids=sample_ids)
    with np.load(pack_path, allow_pickle=False) as pack:
        if not np.array_equal(pack["inputs"], input_array):
            raise ValueError("written input pack failed byte-exact reload")
        if not np.array_equal(pack["targets"], target_array):
            raise ValueError("written target pack failed byte-exact reload")
        if pack["sample_ids"].tolist() != sample_ids.tolist():
            raise ValueError("written sample IDs failed exact-order reload")

    report: dict[str, Any] = {
        "schema": "journal_stage1_input_pack_v1",
        "status": "passed",
        "selection_role": "configuration-validation only; precision-map selection permitted",
        "formal_test_role": "identity/preprocessing replay only; no tensor persisted; no selection permitted",
        "leakage_gates": {
            "all_locked_split_gates_true": True,
            "selection_exactly_64_official_train_samples": True,
            "frozen_test_exactly_90_identity_order_only": True,
            "all_five_sets_pairwise_disjoint_by_id_image_and_image_target_content": True,
            "official_validation_reserved": True,
            "no_frozen_test_tensor_written": True,
        },
        "preprocessing_contract": {
            "terratorch": "0.99.7",
            "albumentations": "1.4.10",
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "bands": list(BANDS),
            "zero_based_band_indices": BAND_INDICES.tolist(),
            "constant_scale": CONSTANT_SCALE,
            "image_resize": "OpenCV INTER_LINEAR to 224x224",
            "mask_resize": "OpenCV INTER_NEAREST to 224x224",
            "normalization": "not applied; embedded in locked ONNX graph",
            "input_dtype_shape": ["float32", 64, *OUTPUT_SHAPE],
            "target_dtype_shape": ["int64", 64, *TARGET_SHAPE],
        },
        "frozen_test_replay": {
            "historical_runtime_order_reconstructed": True,
            "historical_id_tensor_misalignment_detected": True,
            "historical_assigned_ids_sha256": evidence["sample_ids_sha256"],
            "actual_runtime_source_ids": [row["sample_id"] for row in runtime_rows],
            "expected_samples": 90,
            "valid_pixels": valid_pixels,
            "target_hashes_and_valid_pixels_matched_samples": 90,
            "raw_hashes_exact_on_current_platform": sum(
                int(item["raw_hash_exact_on_current_platform"]) for item in replay_rows
            ),
            "current_platform_raw_hash_list_sha256": raw_list_hash,
            "historical_raw_hash_list_sha256": evidence["raw_hash_list_sha256"],
            "target_hash_list_sha256": target_list_hash,
            "first_sample_numeric_replay": {
                "historically_assigned_sample_id": evidence_samples[0]["sample_id"],
                "actual_source_sample_id": runtime_rows[0]["sample_id"],
                "max_abs": replay_max_abs,
                "mean_abs": replay_mean_abs,
                "relative_l2": replay_relative_l2,
                "max_abs_tolerance": REPLAY_MAX_ABS_TOLERANCE,
                "mean_abs_tolerance": REPLAY_MEAN_ABS_TOLERANCE,
                "passed": True,
            },
            "interpretation": (
                "Target hashes reproduce 90/90 exactly after reconstructing TerraTorch file order. "
                "Floating image interpolation is numerically equivalent but not byte-identical across "
                "the historical Linux/K100 AI accelerator and current Windows runtime. New Stage-1 data uses direct "
                "ID-to-path pairing and does not inherit the historical ID/tensor misalignment."
            ),
        },
        "configuration_validation": {
            "sample_count": 64,
            "sample_ids": sample_ids.tolist(),
            "sample_ids_sha256": hashlib.sha256(
                ("\n".join(sample_ids.tolist()) + "\n").encode("utf-8")
            ).hexdigest(),
            "inputs_sha256": sha256_array(input_array),
            "targets_sha256": sha256_array(target_array),
            "samples": selection_samples,
        },
        "inputs": {
            "selection_manifest": identity(args.selection_manifest.resolve()),
            "frozen_test_manifest": identity(args.frozen_test_manifest.resolve()),
            "split_manifest": identity(args.split_manifest.resolve()),
            "prior_frozen_test_evidence": identity(args.frozen_test_evidence.resolve()),
            "prior_first_sample_tensor": identity(args.frozen_first_sample_tensor.resolve()),
        },
        "output_pack": identity(pack_path),
    }
    report_path = args.output_dir / "input_pack_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": "passed", "pack": str(pack_path), "pack_sha256": sha256_file(pack_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
