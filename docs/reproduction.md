# Reproduction workflow

Run commands from the repository root and choose new output directories. The examples document interfaces; they are not execution logs or a claim of completed hardware validation.

## 1. Offline tables and statistics

Use Python with NumPy for `analyze_results.py`. Markdown table export uses only the standard library.

```bash
python scripts/reproduce_tables.py --output-dir outputs/tables
python scripts/analyze_results.py flood --output-dir outputs/flood-analysis
python scripts/analyze_results.py cloud --output-dir outputs/cloud-analysis
```

The statistical commands use the included derived records. They do not train, infer, select new configurations or change retained tables. Computational requirements include memory for 10,000 paired scene resamples.

## 2. Inputs and model export

```bash
python scripts/prepare_inputs.py flood --help
python scripts/prepare_inputs.py cloud --records manifests/datasets/cloud/formal_records.csv --source-root external/cloudsen12 --output-root outputs/cloud-inputs
python scripts/export_model.py flood --project-root external/flood
python scripts/export_model.py cloud --help
```

The flood materializer retains the original input-pack identity checks and requires its listed source manifests. Cloud preparation reads local TACO parts; allocate space for all 300 six-band scene tensors. The cloud exporter takes explicit task/backbone checkpoints, a data manifest and `src/model_export/cloud_model.py` as its trainer module. Neither export procedure should be interpreted as authorizing a new training experiment.

## 3. Diagnostics and precision

```bash
python scripts/diagnose_blocks.py run --help
python scripts/diagnose_blocks.py aggregate --help
python scripts/build_deployment.py precision-map --help
python scripts/build_deployment.py fp16-segments --help
```

Use the 64-scene configuration pack for diagnostics and the distinct calibration set for quantization. The packed aggregate S7 table is insufficient input for the diagnostic aggregator: that program needs complete per-scene records and their manifest.

For `fp16-segments`, the `--legacy-builder` parameter identifies the byte-preserved `src/precision/segment_builder.py`. For cloud quantization, `--base-quantizer` identifies `src/precision/cloud_quantizer_base.py`. These parameter names are retained for interface compatibility.

## 4. Graph construction and compilation

```bash
python scripts/build_deployment.py flood-graph --help
python scripts/build_deployment.py cloud-graph --help
python scripts/build_deployment.py cloud-split --help
python scripts/build_deployment.py compile-flood --help
python scripts/build_deployment.py compile-cloud --help
```

The cloud graph mapper uses `src/graph/rcs13_builder.py` and `configs/topology/flood_partition_locked.json` as its frozen builder and boundary dependencies. The final cloud split is applied to the corresponding decoder-containing session. Model weights and tensor boundaries must match the indicated source identities.

Compilation runs in the vendor K100 environment. The `--shim` argument refers to the study's explicit container compatibility helper. Compile one session with its matching feed tensors and preserve numerical, provider and resource checks.

## 5. Inference and statistics

```bash
python scripts/evaluate.py flood --help
python scripts/evaluate.py cloud --help
python scripts/evaluate.py cloud-statistics --help
```

The flood evaluator expects its fixed 90-image pack and runtime manifest. Its historical machine schema includes `formal` fields; the scientific interpretation remains descriptive. The cloud evaluator requires fourteen ordered model/cache pairs, the supplied input contract and a complete scene payload directory. The local reproduction payload status does not assert a new independent test.

`cloud-statistics` is the retained pixel-level analysis path requiring predictions and labels. The offline `analyze_results.py cloud` command instead reads the included compact derived records.

## 6. Timing

Build the appropriate C++ runner using `environment/k100_runtime.md`. Prepare its plan from matching tensor interfaces and artifact identities; `src/runtime/materialize_cpp_plan.py` provides the retained flood manifest-to-plan implementation. Cloud and flood output contracts differ.

```bash
python scripts/benchmark.py cpp --runner build/flood/flood/k100_pipeline_benchmark --plan external/plans/flood_fp32.tsv --input-raw external/inputs/flood.f32 --cpu 0 --output-dir outputs/flood-latency
```

Choose a CPU permitted by the target host affinity. Run under the same isolated accelerator and fixed software conditions as the intended comparison. The launcher starts five fresh processes with 50 warm-ups and 200 measured calls each; it does not manage remote nodes or terminate other users' processes.

The single-session flood Python benchmark has its own command interface:

```bash
python scripts/benchmark.py python-flood --help
```

Do not merge the Python and C++ timing populations into a precision-only comparison.

## Complete experiment inventory

See [experiment coverage](experiment_coverage.md), [the runbook](experiment_runbook.md), and [system measurement procedures](system_measurements.md) for controls, sensitivity, resources, startup, recovery and sustained execution.
