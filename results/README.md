# Compact result evidence

This directory contains small, Git-suitable copies of the authoritative summaries used by the paper. It intentionally excludes raw tensors, full latency arrays when embedded elsewhere, ONNX/MXR artifacts, Docker inspect archives, telemetry JSONL, and profiler logs.

## Human-readable tables

- `paper_metrics.csv`: fixed 90-image task metrics, strict diagnostic, logical capacity, topology-matched latency, and device-memory summary.
- `deployment_comparison.csv`: complete bundle capacity and production-style same-CLI comparison.

## Authoritative machine-readable summaries

- `fp32_25segment_performance_summary.json`
- `m0_m4_performance_summary.json`
- `M5_performance_summary.json`
- `M0_test90_summary.json` ... `M5_test90_summary.json`
- `m0_m5_combined_pareto.json`
- `kernel_evidence_summary.json`
- `candidate_block_kernel_matrix.csv`
- `final_deployment_acceptance.json`
- `M5_vs_FP16_same_cli_summary.json`
- 60-minute M5/FP16 result summaries

The CSV tables use the rounded values published in the manuscript. For re-analysis, use the corresponding JSON evidence and preserve its schema, status, and claim boundaries.

Do not infer a formal INT8 speedup from the `1.006` M5/FP32 factor. The difference is treated as latency parity. Do not infer strict logit equivalence from task-level passes.

