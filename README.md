# Prithvi deployment on the Hygon DCU architecture

[中文说明](README_zh.md)

Research code and compact derived records associated with:

**Onboard Deployment of Remote Sensing Foundation Models for the Hygon DCU Architecture**

Chengang Yan, Haining Tan, Longxiang Yin, Youwei Wang, Jie Luo and Jibing Qiu.
Correspondence: qiujibing@ict.ac.cn.

## Research scope

The study evaluates Prithvi-EO-2.0–UPerNet on Hygon K100 accelerators through ground-based flood and cloud segmentation. It links blockwise numerical diagnostics, precision assignment and graph restructuring to task accuracy, compilation resources and inference performance.

- Flood deployment uses a 25-session diagnostic graph and a common 13-session comparison topology. Results on the previously accessed 90-image public pool are descriptive.
- Cloud deployment uses a repaired 14-session graph. Evaluation on 300 previously unused, region-disjoint scenes accepts FP16 and rejects the mixed candidate because of prediction agreement.
- The measurements provide ground-based evidence relevant to future onboard deployment. They do not establish completed in-orbit operation.

## Contents

| Directory | Purpose |
|---|---|
| `src/` | Preprocessing, export, diagnostics, graph/precision operations, inference, metrics and statistics |
| `cpp/` | Separate flood and cloud C++ runner contracts |
| `scripts/` | Command entry points and offline table/statistical analysis |
| `configs/` | Task protocols, precision maps, topology records and runtime manifest templates |
| `environment/` | Analysis, model preparation, quantization and K100 environment requirements |
| `manifests/` | Dataset selection, external artifact identities and source identities |
| `results/` | Retained supplementary tables and scene-level derived records |
| `figures/` | Diagnostic plotting entry point and figure scope |
| `docs/` | Reproduction instructions, methods, availability and paper-to-code mapping |

## Start here

The commands below are usage instructions. This repository does not claim that its reorganized entry points have been executed on the reader's environment.

### Read retained tables without K100

```bash
python scripts/reproduce_tables.py --output-dir outputs/tables
```

This formats the retained CSV values as Markdown. It does not recalculate experimental results.

### Reanalyse derived records without inference

Use a separate Python environment with `environment/requirements-analysis.txt`:

```bash
python scripts/analyze_results.py flood --output-dir outputs/flood-analysis
python scripts/analyze_results.py cloud --output-dir outputs/cloud-analysis
```

The flood analysis separates pooled scores from mean paired scene differences. The cloud analysis uses scene confusion matrices and the retained full-precision agreement records. See [evaluation protocols](docs/evaluation_protocols.md).

### Prepare and run models

```bash
python scripts/prepare_inputs.py --help
python scripts/export_model.py --help
python scripts/diagnose_blocks.py --help
python scripts/build_deployment.py --help
python scripts/evaluate.py --help
python scripts/benchmark.py --help
```

Model execution requires the external data, task weights, graph/cache artifacts and vendor runtime described in [data and models](docs/data_and_models.md) and [K100 environment](environment/k100_runtime.md). Some identity-locked construction procedures additionally require their exact calibration manifests and build reports. The repository alone does not supply every prerequisite for end-to-end reproduction.

Read [reproduction instructions](docs/reproduction.md) and the [paper-to-code map](docs/paper_mapping.md) before selecting an experiment. Independent source inspection, offline reanalysis and hardware execution have different requirements.

## Numerical and execution contracts

- Preserve the task-specific preprocessing: normalization is embedded in the flood graph and external to the cloud graph.
- Keep FP32 external inputs and logits, tensor names, session order and feature taps consistent with the selected artifact identities.
- Compare latency within the same runner, node, topology and measurement scope.
- Retained schema identifiers are machine-readable provenance labels. Their historical use of words such as `formal` does not establish independence of the flood public pool.

## Citation and permissions

Use [CITATION.cff](CITATION.cff) for attribution and identify the repository commit used in a reproduction. The code is maintained in [Prithvi-K100-Auditable-Deployment](https://github.com/Undergraduate-Yan/Prithvi-K100-Auditable-Deployment). No article DOI is asserted here.

Code use and redistribution are governed by [LICENSE](LICENSE). No open-source license has been selected. Third-party components retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
