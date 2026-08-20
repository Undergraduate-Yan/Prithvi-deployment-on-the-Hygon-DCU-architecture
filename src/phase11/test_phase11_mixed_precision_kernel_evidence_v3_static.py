#!/usr/bin/env python3
"""Offline invariants for the Phase 11 v3 direct-hipprof toolchain."""
from __future__ import annotations

import csv
import contextlib
import argparse
import hashlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

import materialize_phase11_mixed_precision_block_input as prepass
import plan_phase11_mixed_precision_kernel_evidence as planner
import profile_phase11_mixed_precision_block as target
import summarize_phase11_mixed_precision_kernel_evidence as summarizer


ROOT = Path(__file__).resolve().parent


class CompatBooleanOptionalAction(argparse.Action):
    """Python 3.8 test-host shim; the admitted container uses Python 3.10."""

    def __init__(self, option_strings, dest, default=None, **kwargs):
        options = []
        for option in option_strings:
            options.append(option)
            if option.startswith("--"):
                options.append("--no-" + option[2:])
        super().__init__(option_strings=options, dest=dest, nargs=0, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not str(option_string).startswith("--no-"))


def identity(path: Path) -> tuple[int, str]:
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert prepass.SCHEMA == "phase11_mixed_precision_boundary_prepass_v3"
    assert planner.SCHEMA == "phase11_mixed_precision_kernel_profile_plan_v3"
    assert target.SCHEMA == "phase11_mixed_precision_block_external_hipprof_target_v3"
    assert summarizer.SCHEMA == "phase11_mixed_precision_kernel_evidence_summary_v3"
    assert summarizer.PLAN_SCHEMA == planner.SCHEMA
    assert summarizer.TARGET_SCHEMA == target.SCHEMA
    assert summarizer.PREPASS_SCHEMA == prepass.SCHEMA
    assert summarizer.EXPECTED_TARGET_IDENTITY == identity(
        ROOT / "profile_phase11_mixed_precision_block.py"
    )
    assert summarizer.EXPECTED_PREPASS_IDENTITY == identity(
        ROOT / "materialize_phase11_mixed_precision_block_input.py"
    )

    protocol = json.loads(
        (ROOT / "PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_PROTOCOL_V3.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["schema"] == "phase11_mixed_precision_kernel_evidence_protocol_v3"
    assert protocol["profile_scope"]["deduplicated_tasks_for_full_M0_M5"] == 33
    assert protocol["profile_scope"]["upstream_segments_in_trace"] == 0
    assert "no --trace-off" in protocol["profile_scope"]["dynamic_profiler_controls"]

    launcher = (ROOT / "run_phase11_mixed_precision_kernel_evidence_node2.sh").read_text(
        encoding="utf-8"
    )
    assert "materialize_phase11_mixed_precision_block_input.py" in launcher
    assert "profile_phase11_mixed_precision_block.py" in launcher
    assert launcher.count('-v "${SAMPLE}:${SAMPLE_CONTAINER}:ro"') == 1
    direct = '"${IMAGE}" --hip-trace --hsa-trace --hiptx-trace \\\n'
    assert direct in launcher
    assert "--trace-off" not in launcher
    assert "--hipprof-session" not in launcher
    assert ' --session "${SESSION}"' not in launcher
    assert "--boundary-input" in launcher and "--boundary-prepass-result" in launcher
    locked_paths = {
        "PREPASS": ROOT / "materialize_phase11_mixed_precision_block_input.py",
        "TARGET": ROOT / "profile_phase11_mixed_precision_block.py",
        "PLANNER": ROOT / "plan_phase11_mixed_precision_kernel_evidence.py",
        "SUMMARIZER": ROOT / "summarize_phase11_mixed_precision_kernel_evidence.py",
        "PROTOCOL": ROOT / "PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_PROTOCOL_V3.json",
    }
    for variable, path in locked_paths.items():
        matched = re.search(
            rf'lock_file "\$\{{{variable}\}}" (\d+) ([0-9a-f]{{64}})', launcher
        )
        assert matched, variable
        assert (int(matched.group(1)), matched.group(2)) == identity(path), variable

    target_source = (ROOT / "profile_phase11_mixed_precision_block.py").read_text(encoding="utf-8")
    assert "subprocess" not in target_source
    assert "range(args.segment_index)" not in target_source
    assert "run_control" not in target_source
    assert "upstream_segments_executed_in_target_process\": 0" in target_source
    assert "final_output_d2h_for_finite_sha_validation_is_inside_outer_trace" in target_source

    planner_source = (ROOT / "plan_phase11_mixed_precision_kernel_evidence.py").read_text(
        encoding="utf-8"
    )
    assert "boundary_prepass_lineage" in planner_source
    assert "boundary_input_contract" in planner_source
    summarizer_source = (
        ROOT / "summarize_phase11_mixed_precision_kernel_evidence.py"
    ).read_text(encoding="utf-8")
    assert "all_int8 = all_tasks_parsed and all(" in summarizer_source
    assert "all_fp16 = all_tasks_parsed and all(" in summarizer_source
    assert '"selected_profiled_blocks_provider_placement_passed"' in summarizer_source
    assert '"all_blocks_provider_placement_passed": (not smoke_mode)' in summarizer_source
    assert "final output D2H" in summarizer_source

    assert summarizer.classify_kernel("Cijk_Ailk_Bljk_I8II_BH_x", "int8_qdq") == "int8_gemm_i8ii"
    assert summarizer.classify_kernel("Cijk_Ailk_Bljk_HBH_x", "fp16") == "fp16_gemm_hbh"
    assert summarizer.classify_kernel("quant_nearbyint_convert", "int8_qdq") == "qdq_conversion"
    assert summarizer.classify_kernel("transpose_pack", "fp16") == "layout_conversion"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        kernel = tmp / "hipprof.json.kernel.csv"
        with kernel.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Name": "Cijk_Ailk_Bljk_I8II_BH_x",
                    "Calls": "20",
                    "TotalDurationNs": "2000",
                    "AverageNs": "100",
                    "Percentage": "80",
                }
            )
            writer.writerow(
                {
                    "Name": "dequant_nearbyint_convert",
                    "Calls": "20",
                    "TotalDurationNs": "500",
                    "AverageNs": "25",
                    "Percentage": "20",
                }
            )
        parsed = summarizer.parse_kernel_csv(kernel, "int8_qdq")
        assert parsed["gpu_kernel_total_duration_ns"] == 2500
        assert parsed["categories"]["int8_gemm_i8ii"]["calls"] == 20
        assert parsed["categories"]["qdq_conversion"]["total_duration_ns"] == 500

        # Exercise planner.main with six manifest paths and a deterministic mock
        # artifact graph. Shared per-block INT8 identities plus the configured nine
        # FP16 blocks must produce exactly 33 trace tasks and complete prepass lineage.
        bundle = tmp / "bundle"
        bundle.mkdir()
        manifests = {}
        for candidate in planner.EXPECTED:
            path = bundle / f"{candidate}.json"
            path.write_text("{}\n", encoding="utf-8")
            manifests[candidate] = path

        def digest(label: str) -> str:
            return hashlib.sha256(label.encode("utf-8")).hexdigest()

        def fake_load_candidate(_bundle: Path, manifest_path: Path):
            candidate = manifest_path.stem
            fp16 = tuple(planner.common.EXPECTED_VARIANTS[candidate])
            rows = []
            for index in range(25):
                precision = (
                    "fp32_compat_barrier"
                    if index == 24
                    else "fp16"
                    if index in fp16
                    else "int8_qdq"
                )
                rows.append(
                    {
                        "index": index,
                        "label": (
                            f"encoder_block_{index:02d}"
                            if index < 24
                            else "upernet_decoder_head_fpn4barrier"
                        ),
                        "precision": precision,
                        "verified_model_identity": {
                            "size_bytes": index + 100,
                            "sha256": digest(f"model:{index}:{precision}"),
                        },
                        "verified_cache_identity": {
                            "size_bytes": index + 200,
                            "sha256": digest(f"cache:{index}:{precision}"),
                        },
                        "input_contracts": [
                            {"name": f"input_{index}", "elem_type": 1, "shape": [1, 4, 8, 8]}
                        ],
                    }
                )
            manifest = {
                "candidate_id": candidate,
                "fp16_backbone_blocks": list(fp16),
                "int8_backbone_blocks": [index for index in range(24) if index not in fp16],
            }
            manifest_identity = {
                "path": str(manifest_path.resolve()),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            return manifest, manifest_identity, rows

        plan_dir = tmp / "plan"
        argv = ["planner", "--bundle", str(bundle), "--output-dir", str(plan_dir)]
        for candidate, path in manifests.items():
            argv.extend(["--manifest", f"{candidate}={path}"])
        argv.append("--require-all-m0-m5")
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            planner.common, "load_candidate", side_effect=fake_load_candidate
        ), mock.patch.object(
            planner.argparse,
            "BooleanOptionalAction",
            CompatBooleanOptionalAction,
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            planner.main()
        planned = json.loads((plan_dir / "profile_plan.json").read_text(encoding="utf-8"))
        assert planned["counts"] == {
            "candidate_count": 6,
            "matrix_rows": 144,
            "unique_profile_tasks": 33,
            "int8_qdq_tasks": 24,
            "fp16_tasks": 9,
        }
        assert all(
            len(task["boundary_prepass_lineage"]["upstream_segments"])
            == task["segment_index"]
            for task in planned["profile_tasks"]
        )

        smoke_dir = tmp / "smoke_plan"
        smoke_argv = [
            "planner",
            "--bundle",
            str(bundle),
            "--output-dir",
            str(smoke_dir),
            "--manifest",
            f"M0={manifests['M0']}",
            "--no-require-all-m0-m5",
            "--max-profile-tasks",
            "1",
        ]
        with mock.patch.object(sys, "argv", smoke_argv), mock.patch.object(
            planner.common, "load_candidate", side_effect=fake_load_candidate
        ), mock.patch.object(
            planner.argparse,
            "BooleanOptionalAction",
            CompatBooleanOptionalAction,
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            planner.main()
        smoke = json.loads((smoke_dir / "profile_plan.json").read_text(encoding="utf-8"))
        assert smoke["status"] == "planned_identity_locked_smoke_no_full_kernel_claim"
        assert smoke["plan_mode"] == "selected_task_smoke"
        assert smoke["counts"]["unique_profile_tasks"] == 1
        assert smoke["full_counts_before_smoke_filter"]["unique_profile_tasks"] == 24
        assert len(smoke["candidate_block_matrix"]) == 1
        assert smoke["profile_tasks"][0]["segment_index"] == 0
        assert smoke["profile_tasks"][0]["precision"] == "int8_qdq"

    stats = """HIP PROF:HIP API statistics
|Name|Calls|TotalDurationNs|AverageNs|Percentage|
|hipMemcpyHtoD|1|100|100|10|
|Total|1|1000|1000|100|
HIP PROF:HSA API statistics
|Name|Calls|TotalDurationNs|AverageNs|Percentage|
|hsa_amd_memory_async_copy|1|200|200|20|
|Total|1|1000|1000|100|
"""
    api = summarizer.parse_stats_tables(stats)
    assert all(api["statistics_headers_found"].values())
    assert api["hip_memcpy_api"]["total_duration_ns"] == 100
    assert api["hsa_async_copy_api"]["total_duration_ns"] == 200

    assert (ROOT / "PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_PROTOCOL_V2.json").is_file()
    assert (ROOT / "PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_V2_README.md").is_file()
    print("phase11 mixed-precision kernel v3 static tests: passed")


if __name__ == "__main__":
    main()
