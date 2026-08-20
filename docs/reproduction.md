# Reproduction guide

This repository supports three levels of reproduction. Use a new output directory for every run; never overwrite frozen evidence.

## 0. Offline repository audit

```bash
python scripts/repo_check.py
python scripts/summarize_results.py
python -m pytest -q
```

This verifies source syntax, JSON integrity, repository size policy, secret patterns, and the compact paper tables. It does not execute a model.

## 1. Prepare external artifacts

Copy `.env.example` to `.env` locally and set only filesystem paths. Do not add credentials.

Place the checkpoint, source ONNX models, compiled caches, dataset, and fixed inputs below a separate artifact root. Use:

```bash
python scripts/verify_external_artifacts.py \
  --manifest artifacts/required_artifacts.example.json \
  --root "$K100_ARTIFACT_ROOT"
```

The artifact root is outside Git. See `artifacts/README.md` for storage options and the difference between source artifacts and generated caches.

## 2. Baseline checkpoint and Full FP16

The initial migration utilities are retained under `src/baseline/`:

1. `k100_checkpoint_forward_smoke.py` and `verify_real_sample_cpu_k100.py` establish strict checkpoint loading and a fixed real-sample CPU/K100 comparison.
2. `evaluate_full_test_k100_fp32.py` reproduces the fixed 90-image FP32 task metrics.
3. `evaluate_full_test_k100_fp32_fp16.py` evaluates native FP16 task preservation.
4. The paired benchmark scripts preserve the original FP32/FP16 performance protocol.

These launchers reflect the archived environment and may contain frozen `/workspace` conventions. Provide paths through their CLI/environment interfaces; do not add passwords or private hosts.

## 3. K100 compatibility graph and 25-segment FP32 denominator

Relevant code is under `src/phase11/`:

```text
rewrite_layernorm_for_migraphx_ort119.py
rewrite_convtranspose_stride2_for_migraphx.py
add_phase11_fpn4_maxpool_barrier.py
build_phase11_fp32_segment25_head_fpn4_barrier.py
evaluate_phase11_fp32_segment25_headbarrier_single.py
evaluate_phase11_fp32_segment25_headbarrier_test90.py
benchmark_phase11_fp32_segment25_headbarrier_all_resident_trial.py
aggregate_phase11_fp32_segment25_headbarrier_performance.py
```

Run the single-sample gates first, then the 90-image task gate, then three fresh performance processes. The 25-segment FP32 result is the topology-matched denominator; the monolithic Full FP16 result is a separate deployment scope.

## 4. INT8 backbone segmentation and discrepancy localization

Use:

```text
build_phase11_int8_backbone_segment25.py
validate_phase11_int8_backbone_segment25_cpu.py
evaluate_phase11_int8_backbone_segment25_cached_single.py
evaluate_phase11_int8_backbone_segment25_cached_90_diagnostic.py
diagnose_phase11_int8_backbone_segment25_numeric_localization.py
```

The local and cumulative CPU--MIGraphX discrepancies are backend diagnostics. They are not pure quantization error and are not universal task-loss sensitivity scores.

## 5. Build and admit M0--M5

Follow the frozen instructions:

- `src/phase11/PHASE11_MIXED_PRECISION_EVAL_README.md`
- `src/phase11/PHASE11_MIXED_PRECISION_M5_README.md`

The sequence is build → compile FP16 caches → finalize manifest → single sample → three fresh 90-image processes → aggregation → performance → VRAM → Pareto summary.

M0--M4 and M5 have separate evidence roots. M5 must not overwrite M0--M4.

## 6. Kernel evidence

Use the direct-trace v3 protocol:

- `src/phase11/PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_V3_README.md`
- `protocols/PHASE11_MIXED_PRECISION_KERNEL_EVIDENCE_PROTOCOL_V3.json`

The untraced prepass materializes the actual boundary tensor; `hipprof` then traces only the target segment. Full acceptance requires all 33 unique traces and all 144 candidate/block mappings. I8II/HBH proves the intended block-level GEMM path only.

## 7. Immutable deployment bundles

The complete tools are in `src/phase11/phase11_minimal_deployment/`.

The acceptance sequence is:

1. bundle construction and static SHA verification;
2. five fresh cold starts;
3. three forced termination/reload cycles;
4. K100-2 and K100-3 smoke tests;
5. 60-minute stability with 5-second telemetry;
6. fail-closed final aggregation.

Use `FINAL_DEPLOYMENT_ACCEPTANCE_README.md` in that directory. Failed environmental attempts remain excluded evidence and never count as model failures or passed gates.

## 8. Production-style same-CLI benchmark

Use `src/phase11/phase11_deployment_cli_pair_benchmark/README.md`. It runs six fresh containers in an alternating order, with 30 warm-ups and 100 timed calls per container. This comparison combines precision and topology effects.

## 9. Paper

From `paper/`:

```bash
python C:/Users/<user>/.codex/skills/latex-paper-en/scripts/compile.py main.tex --recipe pdflatex-bibtex
python C:/Users/<user>/.codex/skills/latex-paper-en/scripts/compile.py main_CN.tex --recipe xelatex-bibtex
```

Any standard IEEEtran-compatible LaTeX toolchain can be substituted. The final English and Chinese PDFs are included for comparison.

