"""Decode CloudSEN12+ scenes using the study band, scale and validity rules."""

from __future__ import annotations

import hashlib

import numpy as np

import pyarrow as pa

import pyarrow.parquet as pq

from rasterio.io import MemoryFile

BANDS = (2, 3, 4, 9, 11, 12)
VALID_SIZE = 2000
SCALE = 0.0001
IGNORE = 255
CLASSES = {0, 1, 2, 3}

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def nested_assets(payload: bytes, datapoint_id: str) -> tuple[bytes, bytes]:
    if payload[:2] not in {b"#y", b"WX"}:
        raise RuntimeError(f"invalid nested TACO magic for {datapoint_id}")
    offset = int.from_bytes(payload[2:10], "little")
    length = int.from_bytes(payload[10:18], "little")
    if offset < 18 or offset + length > len(payload):
        raise RuntimeError(f"invalid nested footer for {datapoint_id}")
    rows = {row["tortilla:id"]: row for row in pq.read_table(pa.BufferReader(payload[offset:offset + length])).to_pylist()}
    if set(rows) != {"s2l2a", "target"}:
        raise RuntimeError(f"unexpected nested assets for {datapoint_id}: {sorted(rows)}")

    def extract(name: str) -> bytes:
        row = rows[name]
        begin, size = int(row["tortilla:offset"]), int(row["tortilla:length"])
        value = payload[begin:begin + size]
        if len(value) != size:
            raise RuntimeError(f"short nested asset {name} for {datapoint_id}")
        return value

    return extract("s2l2a"), extract("target")

def decode(index: int, record: dict[str, str], payload: bytes) -> tuple[int, dict, np.ndarray, np.ndarray]:
    image_bytes, label_bytes = nested_assets(payload, record["datapoint_id"])
    with MemoryFile(image_bytes) as memory, memory.open() as source:
        if source.count != 14 or source.width < VALID_SIZE or source.height < VALID_SIZE:
            raise RuntimeError(f"unexpected image raster contract: {source.count}/{source.width}/{source.height}")
        image = source.read(BANDS, window=((0, VALID_SIZE), (0, VALID_SIZE))).astype(np.float32)
        valid = source.read_masks(BANDS, window=((0, VALID_SIZE), (0, VALID_SIZE))) > 0
    with MemoryFile(label_bytes) as memory, memory.open() as source:
        if source.count != 1 or source.width < VALID_SIZE or source.height < VALID_SIZE:
            raise RuntimeError(f"unexpected label raster contract: {source.count}/{source.width}/{source.height}")
        label = source.read(1, window=((0, VALID_SIZE), (0, VALID_SIZE))).astype(np.uint8)
        label_valid = source.read_masks(1, window=((0, VALID_SIZE), (0, VALID_SIZE))) > 0
    combined = np.all(valid, axis=0) & label_valid
    label[~combined] = IGNORE
    image[:, ~combined] = 0
    image *= SCALE
    if not np.isfinite(image).all():
        raise RuntimeError(f"non-finite image for {record['datapoint_id']}")
    unexpected = set(int(value) for value in np.unique(label)) - CLASSES - {IGNORE}
    if unexpected:
        raise RuntimeError(f"unexpected labels for {record['datapoint_id']}: {sorted(unexpected)}")
    evidence = {
        "index": index,
        "datapoint_id": record["datapoint_id"],
        "roi_id": record["roi_id"],
        "source_part": record["source_part"],
        "source_begin": int(record["source_begin"]),
        "source_length": int(record["source_length"]),
        "source_payload_sha256": sha256_bytes(payload),
        "image_geotiff_sha256": sha256_bytes(image_bytes),
        "label_geotiff_sha256": sha256_bytes(label_bytes),
        "valid_pixels": int(np.count_nonzero(label != IGNORE)),
        "class_pixel_counts": [int(np.count_nonzero(label == value)) for value in range(4)],
    }
    return index, evidence, image, label
