# Code map to paper evidence

| Paper evidence | Primary code | Compact result |
|---|---|---|
| Checkpoint migration and fixed real sample | `src/baseline/k100_checkpoint_forward_smoke.py`, `verify_real_sample_cpu_k100.py` | paper Method and baseline descriptions |
| Full FP16 task baseline | `src/baseline/evaluate_full_test_k100_fp32_fp16.py`, Phase11 FP16 compatibility scripts | `results/paper_metrics.csv` |
| 25-segment FP32 denominator | `build_phase11_fp32_segment25_head_fpn4_barrier.py`, single/test90 evaluators | `results/fp32_25segment_performance_summary.json` |
| INT8 25-segment compatibility | INT8 builder, CPU validator, cached single/test90 evaluators | `results/M0_test90_summary.json` |
| Backend-discrepancy localization | `diagnose_phase11_int8_backbone_segment25_numeric_localization.py` | paper Table IV/Figure 2 |
| M0--M4 construction | `build_phase11_sensitivity_mixed_precision_m0_m4.py` and cache finalizer | `results/M0_test90_summary.json` ... `M4_test90_summary.json` |
| M5 construction | M5 builder, incremental cache compiler, finalizer | `results/M5_test90_summary.json` |
| Topology-matched performance | mixed trial, aggregate, FP32 trial/aggregate | `results/m0_m4_performance_summary.json`, `M5_performance_summary.json` |
| Device VRAM and Pareto analysis | VRAM sampler, capacity/Pareto summarizers | `results/m0_m5_combined_pareto.json` |
| Kernel realization | v3 prepass, target profiler, planner, summarizer | `results/kernel_evidence_summary.json`, block matrix CSV |
| Operational deployment | minimal-deployment tool directory | `results/final_deployment_acceptance.json` |
| M5 versus Full FP16 same CLI | deployment CLI pair benchmark directory | `results/M5_vs_FP16_same_cli_summary.json` |
| IEEE paper | `paper/main.tex`, `paper/main_CN.tex`, figure generators | `paper/main_final.pdf`, `paper/main_final_CN.pdf` |

The copied shell launchers are frozen evidence-oriented entry points. Their hashes and fixed `/var/tmp` conventions are intentional. For a new experiment, create a new protocol/run root rather than silently modifying a frozen launcher and mixing outputs.

