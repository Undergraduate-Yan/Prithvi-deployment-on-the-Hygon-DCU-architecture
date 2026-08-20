# Auditable Prithvi-EO-2.0 Deployment on a K100 AI Accelerator

[中文说明](README_zh.md)

This repository is the Git-ready source and compact evidence package for the paper:

> **Auditable Mixed-Precision Deployment of Prithvi-EO-2.0 on a K100 AI Accelerator: Accuracy, Kernel Paths, and System Trade-offs**

It contains the reproducible Python/Bash toolchain, immutable protocol files, compact result summaries, deployment utilities, and English/Chinese LaTeX sources. Multi-gigabyte checkpoints, ONNX graphs, MXR caches, datasets, container images, and raw profiler output are deliberately excluded from Git and are referenced by size/SHA256 instead.

## Main result

- **Full FP16** is the speed-oriented primary deployment for the evaluated checkpoint and K100 software stack.
- **M5** is a storage-oriented deployable alternative: its complete bundle is 23.463% smaller, but it is 73.642% slower than Full FP16 under the same production-style CLI.
- M5 is at latency parity with the topology-matched 25-segment FP32 baseline (22.080 ms versus 22.209 ms); this is **not** reported as a formal INT8 speedup.
- The historical strict CPU--MIGraphX logit diagnostic remains failed. Task-level acceptance does not overwrite that result.

## Repository layout

```text
src/baseline/                  Initial checkpoint, FP32, and Full-FP16 checks
src/phase11/                   Final K100 graph, evaluation, performance, and kernel tools
src/phase11/phase11_minimal_deployment/
                               Immutable bundle and operational-acceptance tools
src/phase11/phase11_deployment_cli_pair_benchmark/
                               Same-CLI M5 versus Full-FP16 benchmark
protocols/                     Copies of the frozen experiment protocols
artifacts/                     External-artifact identities and mounting instructions
results/                       Compact authoritative JSON/CSV summaries
paper/                         English/Chinese LaTeX, figures, audits, and final PDFs
docs/                          Experiment map, evidence boundaries, and release guidance
scripts/                       Repository and external-artifact verification utilities
```

## Reproduction levels

1. **Offline audit**: inspect paper tables, JSON summaries, hashes, and run static tests without K100 hardware.
2. **Artifact reconstruction**: provide the frozen checkpoint/ONNX inputs and rebuild compatible FP32, INT8, and M0--M5 segment manifests.
3. **K100 execution**: use the locked DTK/MIGraphX/ORT container, compile/load MXR caches, run task gates, kernel traces, performance, and deployment acceptance.

Start with [docs/reproduction.md](docs/reproduction.md) and [artifacts/README.md](artifacts/README.md).

## Quick offline check

```bash
python scripts/repo_check.py
python scripts/summarize_results.py
python -m pytest -q
```

Verify separately stored artifacts:

```bash
python scripts/verify_external_artifacts.py \
  --manifest artifacts/required_artifacts.example.json \
  --root /absolute/path/to/k100_artifacts
```

## Evidence boundaries

Provider placement, realized kernel precision, task accuracy, system latency, and operational reliability are separate claims. See [docs/evidence_boundaries.md](docs/evidence_boundaries.md) before reusing any number or making acceleration claims.

## Publication and licensing

The repository has been scrubbed of known credentials and excludes large/licensed model artifacts. A project license and author metadata have intentionally **not** been invented; resolve [LICENSE_PENDING.md](LICENSE_PENDING.md) and the release checklist before making a public repository.

