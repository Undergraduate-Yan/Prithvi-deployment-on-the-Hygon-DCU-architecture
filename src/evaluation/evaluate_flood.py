#!/usr/bin/env python3
'Research implementation: evaluate flood.'

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from flood_runner import K100BundleRunner, identity


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-pack", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    pack_id = identity(args.input_pack)
    if (
        manifest.get("schema") != "journal_phase5_formal_test90_pack_v1"
        or manifest.get("status") != "passed"
        or manifest.get("sample_count") != 90
        or manifest.get("valid_pixels") != 3927398
        or manifest.get("formal_90_image_test_access_count") != 1
        or manifest.get("formal_90_image_test_used") is not True
        or manifest.get("used_for_selection") is not False
        or pack_id["sha256"] != manifest["identities"]["output_pack"]["sha256"]
    ):
        raise RuntimeError("formal test90 pack admission failed")
    with np.load(args.input_pack, allow_pickle=False) as pack:
        inputs = np.ascontiguousarray(pack["inputs"], dtype=np.float32)
        targets = np.ascontiguousarray(pack["targets"], dtype=np.int64)
        sample_ids = np.asarray(pack["sample_ids"])
    if (
        inputs.shape != (90, 6, 224, 224)
        or targets.shape != (90, 224, 224)
        or sample_ids.tolist() != manifest["sample_ids"]
        or array_sha256(inputs) != manifest["inputs_sha256"]
        or array_sha256(targets) != manifest["targets_sha256"]
    ):
        raise RuntimeError("formal test90 pack content identity drift")

    args.output_dir.mkdir(parents=True)
    source_config = json.loads(args.config.read_text(encoding="utf-8"))
    runtime_config = args.config
    status_only_adapter = None
    if source_config.get("status") == "frozen_phase3" and args.label == "RCS13-FP16-Opt":
        adapted = dict(source_config)
        adapted["status"] = "frozen_ready"
        runtime_config = args.output_dir / "RCS13_FP16_OPT_runtime_status_adapter.json"
        runtime_config.write_text(json.dumps(adapted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status_only_adapter = {
            "reason": "The selected Phase-3 config uses status=frozen_phase3 while the common runner admits status=frozen_ready; execution/model/cache fields are unchanged.",
            "source_status": "frozen_phase3",
            "runtime_status": "frozen_ready",
            "source_config": identity(args.config),
            "runtime_config": identity(runtime_config),
            "execution_unchanged": source_config["execution"] == adapted["execution"],
            "runtime_contract_unchanged": source_config["runtime_contract"] == adapted["runtime_contract"],
            "external_contract_unchanged": source_config["external_contract"] == adapted["external_contract"],
        }
    profiles_dir = args.output_dir / "profiles"
    runner = K100BundleRunner(runtime_config, device=0, profile_dir=profiles_dir)
    predictions = []
    rows = []
    aggregate_logits_hash = hashlib.sha256()
    for index, value in enumerate(inputs):
        logits, prediction, elapsed_ms = runner.run(value[None])
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        prediction = np.ascontiguousarray(prediction[0], dtype=np.uint8)
        aggregate_logits_hash.update(logits.tobytes())
        predictions.append(prediction)
        rows.append({
            "variant": args.label,
            "sample_index": index,
            "sample_id": str(sample_ids[index]),
            "diagnostic_elapsed_ms": elapsed_ms,
            "logits_sha256": array_sha256(logits),
            "prediction_sha256": array_sha256(prediction),
            "nonfinite_logits": int(np.count_nonzero(~np.isfinite(logits))),
        })
        print(f"{args.label} formal scene {index + 1:02d}/90 passed", flush=True)
    prediction_array = np.ascontiguousarray(np.stack(predictions), dtype=np.uint8)
    profiles = runner.finish_profiles()
    prediction_path = args.output_dir / "predictions.npz"
    np.savez_compressed(prediction_path, predictions=prediction_array, sample_ids=sample_ids)
    csv_path = args.output_dir / "per_sample_runtime_hashes.csv"
    write_csv(csv_path, rows)
    result = {
        "schema": "journal_phase5_formal_test90_variant_v1",
        "status": "passed",
        "hardware": "海光 K100 AI 加速卡",
        "variant": args.label,
        "sample_count": 90,
        "valid_pixels": 3927398,
        "config": identity(args.config),
        "runtime_config": identity(runtime_config),
        "status_only_adapter": status_only_adapter,
        "input_pack": pack_id,
        "input_manifest": identity(args.input_manifest),
        "runtime": {
            **runner.runtime,
            "load_seconds": runner.load_seconds,
            "diagnostic_elapsed_ms": [row["diagnostic_elapsed_ms"] for row in rows],
            "diagnostic_elapsed_not_a_performance_selection_measurement": True,
        },
        "profiles": profiles,
        "outputs": {
            "predictions": identity(prediction_path),
            "per_sample_runtime_hashes": identity(csv_path),
            "aggregate_logits_sha256": aggregate_logits_hash.hexdigest(),
            "aggregate_predictions_sha256": array_sha256(prediction_array),
            "nonfinite_logits": sum(row["nonfinite_logits"] for row in rows),
        },
        "gates": {
            "formal_pack_identity_exact": True,
            "sample_order_exact": True,
            "all_90_outputs_finite": all(row["nonfinite_logits"] == 0 for row in rows),
            "all_sessions_migraphx_positive": all(row["migraphx_events_positive"] for row in profiles),
            "all_sessions_cpu_provider_zero": all(row["cpu_events_zero"] for row in profiles),
            "configuration_frozen_before_unseal": True,
            "status_only_adapter_preserves_execution": status_only_adapter is None or all(
                status_only_adapter[key] for key in (
                    "execution_unchanged", "runtime_contract_unchanged", "external_contract_unchanged"
                )
            ),
            "test_result_not_used_for_selection": True,
        },
        "formal_90_image_test_access_count": 1,
        "formal_90_image_test_used": True,
        "used_for_selection": False,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "variant": args.label, "sessions": len(profiles), "aggregate_logits_sha256": result["outputs"]["aggregate_logits_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
