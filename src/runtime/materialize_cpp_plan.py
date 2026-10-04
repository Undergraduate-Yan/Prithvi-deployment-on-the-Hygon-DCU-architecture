#!/usr/bin/env python3
"""Construct a C++ execution plan from identity-checked flood graph interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMAGE = "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01"
HARDWARE = "海光 K100 AI 加速卡"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_identity(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    observed = (path.stat().st_size, sha256(path))
    expected = (int(record["size_bytes"]), str(record["sha256"]))
    if observed != expected:
        raise RuntimeError(f"{label} identity drift: {path}")
    return path


def clean_field(value: object, label: str) -> str:
    result = str(value)
    if not result or any(character in result for character in "\t\r\n"):
        raise RuntimeError(f"unsafe/empty plan field: {label}")
    return result


def _stage2_interfaces(manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        manifest.get("schema") != "journal_stage2_k100_variant_v1"
        or manifest.get("status") != "frozen_ready"
        or manifest.get("external_contract")
        != {"input": "FP32[1,6,224,224]", "output": "FP32[1,2,224,224] logits", "batch": 1}
        or manifest.get("benchmark_input", {}).get("formal_test_used") is not False
        or manifest.get("runtime_contract", {}).get("onnxruntime") != "1.19.2"
        or manifest.get("runtime_contract", {}).get("provider") != "MIGraphXExecutionProvider"
        or manifest.get("runtime_contract", {}).get("cpu_fallback_disabled") is not True
    ):
        raise RuntimeError("Stage-2 variant manifest frozen contract drift")
    execution = manifest.get("execution", {})
    kind = execution.get("kind")
    if kind == "mono":
        rows = [execution]
        rows[0]["session_id"] = f"{manifest['variant']}_00"
        interfaces = [
            {
                "ordinal": 0,
                "session_id": rows[0]["session_id"],
                "status": "passed",
                "gates": {"manifest_interface_frozen": True},
                "model": rows[0]["model"],
                "cache": rows[0]["cache"],
                "inputs": [execution["input_name"]],
                "outputs": [execution["output_name"]],
                "output_records": {
                    execution["output_name"]: {
                        "shape": [1, 2, 224, 224],
                        "dtype": "float32",
                        "finite": True,
                    }
                },
            }
        ]
    elif kind in {"segment25", "segment13"}:
        rows = execution.get("segments", [])
        expected_count = 25 if kind == "segment25" else 13
        if len(rows) != expected_count or [row.get("index") for row in rows] != list(range(expected_count)):
            raise RuntimeError(f"Stage-2 {kind} order/count drift")
        interfaces = []
        for ordinal, row in enumerate(rows):
            row["session_id"] = f"{manifest['variant']}_{ordinal:02d}"
            raw_inputs = row.get("inputs", [])
            raw_outputs = row.get("outputs", [])
            if not raw_inputs or not raw_outputs:
                raise RuntimeError(f"Stage-2 empty interface at session {ordinal}")
            if all(isinstance(item, dict) for item in raw_inputs + raw_outputs):
                if any(item.get("elem_type") != 1 for item in raw_inputs + raw_outputs):
                    raise RuntimeError(f"Stage-2 static float32 interface drift at session {ordinal}")
                inputs = [str(item["name"]) for item in raw_inputs]
                outputs = [str(item["name"]) for item in raw_outputs]
                output_records = {
                    str(item["name"]): {
                        "shape": item["shape"],
                        "dtype": "float32",
                        "finite": True,
                    }
                    for item in raw_outputs
                }
            elif all(isinstance(item, str) for item in raw_inputs + raw_outputs):
                # The frozen B2 manifest predates embedded elem_type/shape records.
                # Its 24 encoder interfaces are fixed FP32[1,197,1024], while the
                # public head contract exposes only FP32 logits. The auxiliary head
                # output is deliberately not bound or synchronized by the runner.
                if manifest.get("variant") != "B2":
                    raise RuntimeError(f"legacy static interface is allowed only for B2 at {ordinal}")
                inputs = list(raw_inputs)
                outputs = list(raw_outputs[:1] if ordinal == 24 else raw_outputs)
                output_records = {
                    name: {
                        "shape": [1, 2, 224, 224] if name == "logits" else [1, 197, 1024],
                        "dtype": "float32",
                        "finite": True,
                    }
                    for name in outputs
                }
            else:
                raise RuntimeError(f"mixed/unknown static interface at session {ordinal}")
            interfaces.append(
                {
                    "ordinal": ordinal,
                    "session_id": row["session_id"],
                    "status": "passed",
                    "gates": {"manifest_interface_frozen": True},
                    "model": row["model"],
                    "cache": row["cache"],
                    "inputs": inputs,
                    "outputs": outputs,
                    "output_records": output_records,
                }
            )
    else:
        raise RuntimeError(f"unsupported Stage-2 execution kind: {kind}")
    return str(manifest["variant"]), rows, interfaces


def materialize(
    manifest_path: Path, validation_path: Path | None, output_path: Path
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") == "journal_phase2d_rcs13_candidate_manifest_v1":
        if validation_path is None:
            raise RuntimeError("RCS-13 materialization requires validation evidence")
        validation_path = validation_path.resolve(strict=True)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "prepared_for_cache_load_validation"
            or manifest.get("expected_candidate_sessions") != 13
            or len(manifest.get("candidate_sessions", [])) != 13
            or not all(manifest.get("gates", {}).values())
            or manifest.get("selection_boundary", {}).get("formal_90_image_test_used") is not False
        ):
            raise RuntimeError("candidate manifest is not the admitted RCS-13 contract")
        if (
            validation.get("schema") != "journal_phase2d_rcs_cache_load_validation_v2"
            or validation.get("status") != "passed"
            or validation.get("hardware") != HARDWARE
            or validation.get("runtime")
            != {"device_id": 0, "onnxruntime": "1.19.2", "provider": "MIGraphXExecutionProvider"}
            or validation.get("formal_90_image_test_used") is not False
            or not all(validation.get("gates", {}).values())
            or validation.get("manifest", {}).get("sha256") != sha256(manifest_path)
        ):
            raise RuntimeError("interface validation evidence is not the admitted RCS-13 result")
        variant_name = str(manifest["candidate_name"])
        candidate_rows = manifest["candidate_sessions"]
        interface_rows = validation.get("candidate_sessions", [])
    elif manifest.get("schema") == "journal_stage2_k100_variant_v1":
        variant_name, candidate_rows, interface_rows = _stage2_interfaces(manifest)
    else:
        raise RuntimeError("unsupported manifest schema for C++ runner")
    if len(interface_rows) != len(candidate_rows):
        raise RuntimeError("candidate/interface session-count drift")
    lines = [
        "K100_PIPELINE_PLAN_V1",
        f"variant\t{clean_field(variant_name, 'variant')}",
        f"hardware\t{HARDWARE}",
        "runtime\t1.19.2",
        f"image\t{IMAGE}",
        f"session_count\t{len(candidate_rows)}",
        "input\timage\t1,6,224,224",
        "final_logits\tlogits\t1,2,224,224",
    ]
    verified_bytes = 0
    produced = {"image": (1, 6, 224, 224)}
    for ordinal, (row, interface) in enumerate(zip(candidate_rows, interface_rows, strict=True)):
        if (
            interface.get("ordinal") != ordinal
            or interface.get("session_id") != row.get("session_id")
            or interface.get("status") != "passed"
            or not all(interface.get("gates", {}).values())
            or interface.get("model", {}).get("sha256") != row.get("model", {}).get("sha256")
            or interface.get("cache", {}).get("sha256") != row.get("cache", {}).get("sha256")
        ):
            raise RuntimeError(f"candidate/interface identity drift at session {ordinal}")
        model = verify_identity(row["model"], f"session {ordinal} model")
        cache = verify_identity(row["cache"], f"session {ordinal} cache")
        verified_bytes += model.stat().st_size + cache.stat().st_size
        inputs = [clean_field(value, f"session {ordinal} input") for value in interface["inputs"]]
        if not inputs or any(value not in produced for value in inputs):
            raise RuntimeError(f"unresolved session input at ordinal {ordinal}")
        outputs: list[tuple[str, tuple[int, ...]]] = []
        records = interface.get("output_records", {})
        for name_value in interface["outputs"]:
            name = clean_field(name_value, f"session {ordinal} output")
            record = records.get(name, {})
            shape = tuple(int(value) for value in record.get("shape", []))
            if record.get("dtype") != "float32" or not record.get("finite") or not shape or min(shape) <= 0:
                raise RuntimeError(f"invalid output interface at session {ordinal}: {name}")
            if name in produced:
                raise RuntimeError(f"duplicate tensor producer: {name}")
            produced[name] = shape
            outputs.append((name, shape))
        fields = [
            "S",
            str(ordinal),
            clean_field(row["session_id"], f"session {ordinal} id"),
            clean_field(model, f"session {ordinal} model path"),
            str(model.stat().st_size),
            clean_field(cache, f"session {ordinal} cache path"),
            str(cache.stat().st_size),
            str(len(inputs)),
            *inputs,
            str(len(outputs)),
        ]
        for name, shape in outputs:
            fields.extend([name, ",".join(str(value) for value in shape)])
        lines.append("\t".join(fields))
    if produced.get("logits") != (1, 2, 224, 224):
        raise RuntimeError("final logits contract drift")
    lines.append("END")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "schema": "journal_phase2e_cpp_runner_plan_materialization_v1",
        "status": "passed",
        "variant": variant_name,
        "session_count": len(candidate_rows),
        "verified_payload_bytes": verified_bytes,
        "formal_90_image_test_used": False,
        "plan": {
            "path": str(output_path.resolve(strict=True)),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation-result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.manifest, args.validation_result, args.output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
