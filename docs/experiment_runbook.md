# Experiment runbook

Run from the repository root, using new output directories. Preserve the packaged records. Begin with [experiment coverage](experiment_coverage.md); environment roles are described in `environment/`.

## Offline evidence reconstruction

These commands do not run inference or training. To execute all ten analyses with separate logs, use `python scripts/reproduce_offline.py --output-dir outputs/offline`:

```bash
python scripts/reproduce_tables.py --output-dir outputs/tables
python scripts/analyze_results.py flood --output-dir outputs/flood
python scripts/analyze_results.py cloud --output-dir outputs/cloud
python scripts/analyze_controls.py configuration --output-dir outputs/configuration
python scripts/analyze_controls.py flood-latency --output-dir outputs/flood-latency
python scripts/analyze_controls.py cloud-latency --output-dir outputs/cloud-latency
python scripts/analyze_supplementary.py clean89 --config configs/analysis/retained_records.json --output-dir outputs/clean89
python scripts/analyze_supplementary.py margins --config configs/analysis/retained_records.json --output-dir outputs/margins
python scripts/analyze_supplementary.py cloud-confusion --config configs/analysis/retained_records.json --output-dir outputs/cloud-confusion
python scripts/analyze_supplementary.py process-latency --config configs/analysis/retained_records.json --output-dir outputs/process-latency
```

The configuration-count reconstruction yields pooled mIoU, water/background IoU and pixel accuracy. Those confusion matrices do not contain boundary overlap or prediction-agreement counts; inspect the retained candidate table or rerun the evaluator for those quantities. The margin analysis uses the original exploratory `>=` display rule for both tasks; the retained flood descriptive gate uses strict `>` and remains separate.

Full-precision files retain original variant names. Paper labels C0–C5 correspond to M0-R–M5-R; Ctrl_D5 to Legacy-M5; Ctrl_early/late to Early-9/Late-9; Ctrl_U42–U46 to Random-9-seed-42–46. Runtime and task-sensitivity controls retain their explicit selected block lists in `configs/precision/candidates/`.

## Diagnostic and precision controls

```bash
python scripts/diagnose_blocks.py run --help
python scripts/diagnose_blocks.py aggregate --help
python scripts/build_deployment.py precision-map --aggregated-diagnostics results/flood/configuration/aggregated_block_diagnostics.csv --output-dir outputs/generated-maps
python scripts/build_deployment.py fp16-segments --help
python scripts/experiment_controls.py evaluate-precision --help
```

Use the fixed 64-scene configuration input, not the 90-image flood pool, for these commands. The evaluator's explicit FP32/INT8 manifests, FP16 build report, calibration/diagnostic records and input manifest must all be supplied. `docs/cli_arguments.json` records the source-declared arguments. Generated maps and retained paper assignments serve different purposes; do not replace the frozen candidate based on a new evaluation.

## Graph and compilation controls

```bash
python scripts/experiment_controls.py pair-probes --help
python scripts/experiment_controls.py multiblock-probes --help
python scripts/experiment_controls.py fp16-region-probes --help
python scripts/experiment_controls.py head-probe --help
python scripts/experiment_controls.py compile-matrix --help
python scripts/experiment_controls.py compile-passes --help
python scripts/build_deployment.py flood-graph --help
python scripts/build_deployment.py cloud-graph --help
python scripts/build_deployment.py cloud-split --help
python scripts/experiment_controls.py cloud-candidate --help
```

`compile-matrix` is the original cgroup-v1 controller: C0–C3 here denote compilation settings, not the precision candidates with similarly named paper labels. It retains a 48 GiB host budget, 49 GiB container ceiling, memory safety checks and the exact vendor image identity. It expects code/assets under `/var/tmp`, mounted at identical paths inside Docker. Pass `--tools-root` as the absolute `src/runtime` directory; the controller invokes the packaged compiler and memory samplers. Do not disable its safeguards on a smaller host.

Compile records distinguish session creation, total wall time, GNU-time maximum RSS, process-tree RSS and cgroup peak. Preserve units and scopes. Missing timings for reused sessions are missing values, not zero; their measured sum is a lower bound. Model/cache capacity is computed from actual byte sizes, without summing rounded GiB displays.

For the cloud screening constructor, the companion transfer builder is `src/precision/cloud_transfer_builder.py`; the exact FP16 compatibility utility and graph/report/input prerequisites are listed in [external requirements](external_requirements.md). A historical access ledger must not be fabricated or altered to pass an admission check.

## Deployment evaluation and timing

Follow `docs/reproduction.md` for full-scene evaluation and C++ plan construction. Follow [system measurements](system_measurements.md) for power, VRAM, initialization, recovery and stability. Always separate candidate task acceptance from performance characterization: the cloud mixed candidate remains rejected by agreement even when its runtime measurements succeed.

## Reporting

For each experiment, retain command arguments, input SHA256 values, actual runtime/provider, raw outputs, failure records and node scope. Compare independently recalculated quantities with the packaged full-precision records before comparing rounded manuscript tables. Repository preparation did not execute these research commands; the accompanying `prompt.md` requests independent execution and verification.
