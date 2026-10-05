# External reproduction materials

The repository contains source code, configuration definitions and compact observations. It does not contain raw images/labels, task weights, prediction tensors, ONNX graphs, MXR caches or vendor runtime binaries. Code coverage and material availability must be reported separately.

| Material | Identification / acquisition | Used by |
|---|---|---|
| Sen1Floods11 images, labels and fixed splits | Original dataset distribution; selection IDs in `manifests/datasets/flood/` | Input preparation and flood evaluation |
| CloudSEN12+ TACO parts and labels | Exact dataset revision, URLs and byte offsets in `manifests/datasets/cloud/formal_records.csv`; obey original terms | Full-scene cloud preparation |
| Encoder and selected downstream checkpoints | Model family in `docs/data_and_models.md`; obtain exact study checkpoints and selection metadata from the corresponding author where permitted | Export or fixed-checkpoint inference |
| Ordered ONNX/MXR artifacts | `manifests/artifacts/required_artifacts.json`, task manifests; store as `<sha256>/<basename>` | Runtime, profiling and system tests |
| Flood system development input | SHA256 `7f0e2c08cf337dadf277a91976e6f9f7b3f8e022a2b06ad97547dcb562f5be80`, 1,204,352 bytes | Resource, initialization, recovery and stability measurements |
| Cloud normalized development input pack | Obtain the original development input manifest and matching payload; the pack is not the 300-scene formal dataset | Plan preparation, benchmarking and system measurements |
| 64-scene diagnostic/configuration pack | Original input manifest, normalization contract and source IDs | Block diagnostics and precision candidate evaluation |
| Calibration tensors and quantization reports | Exact construction manifests/selection ledger required by quantizer CLI | FP16/INT8 graph generation |
| Full diagnostic records | 1536 scene/block records and their input manifest, not the S7 aggregate | Recalculation of diagnostic distributions |
| Graph probes, compile feeds and construction reports | Original region/partition and numerical-admission reports required by the selected builder | Graph coarsening and rejected-candidate replay |
| Raw provider and hipprof traces | Trace the exact compiled regions using the included trace programs; retain original trace scopes | Kernel category evidence |
| Hygon image, DTK/HIP/ORT/MIGraphX and K100 | Authorized vendor distribution; versions in `environment/k100_runtime.md` | Accelerator execution and resource measurements |

Study-authored code and unrestricted derived materials may be requested reasonably from Jibing Qiu (`qiujibing@ict.ac.cn`). A dataset or vendor limitation is not permission to redistribute its payload. A hash identifies an artifact; it is not a download URL and cannot regenerate a missing file.

## Companion code arguments

- `--legacy-builder`: `src/precision/segment_builder.py` (byte-preserved).
- `--base-quantizer`: `src/precision/cloud_quantizer_base.py` (byte-preserved).
- `--base-builder` / `--rcs13-base-builder`: `src/graph/rcs13_builder.py` (byte-preserved).
- Cloud screening `--transfer-builder`: `src/precision/cloud_transfer_builder.py`.
- Cloud FP16 `--fp16-compat-tool`: `src/precision/fp16_compat.py`.
- Cloud FP16 `--fp16-resize-tool`: `src/precision/build_fp16_c.py`.
- Compilation `--tools-root`: absolute `src/runtime/`; includes `collect_cgroup_memory.sh` and links to the packaged process-tree sampler.

Preserve the expected identities and original data-role checks. An unavailable historical ledger or artifact must be listed as a missing prerequisite; do not invent an equivalent-looking ledger or replace expected hashes.

## What a complete independent run would need

Supply the exact materials above, establish the documented preparation/analysis/K100 environments, execute the applicable entries in `experiment_coverage.md`, and retain raw outputs. Ordinary CPU analysis can validate many numerical summaries without the accelerator. It cannot validate compiled-kernel execution, hardware resource use or long-duration behavior. Missing inputs are BLOCKED; unexecuted code is NOT_RUN.
