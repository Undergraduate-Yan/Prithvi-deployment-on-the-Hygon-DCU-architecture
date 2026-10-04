# Paper-to-code and input mapping

| Paper content | Implementation | Configuration / retained record | Additional inputs |
|---|---|---|---|
| Architecture and task models, Sections 2.1–2.2 | `src/model_export/`, `cpp/` | `environment/`, `configs/cloud/input_contract.json` | Pretrained and downstream weights, vendor SDK |
| Block diagnostics, Section 2.3; S4 / S7 | `src/diagnostics/diagnose_blocks.py`, `aggregate_blocks.py` | `results/flood/blockwise_discrepancy_64.csv` | 64-scene input pack, segment graphs, manifests and caches |
| Precision assignment; S4 / S8 | `src/precision/` | `configs/precision/flood_c5.json`, configuration candidate CSV | Calibration tensors, quantization reports and source graphs |
| Graph restructuring; S5–S7 | `src/graph/`, `src/runtime/` | `configs/topology/`, topology and rejection CSVs | Exact graph and cache identities, compile feeds |
| Flood accuracy, Table 3; S14–S16 | `src/evaluation/`, `src/statistics/flood_analysis.py`, `scripts/analyze_results.py` | `results/flood/` | Derived records suffice for compact analysis; inference requires inputs and models |
| Cloud accuracy, Table 7; S17–S20 | `src/evaluation/evaluate_cloud.py`, `src/statistics/cloud_analysis.py`, `scripts/analyze_results.py` | `results/cloud/`, `configs/cloud/` | Derived records suffice for compact analysis; full pixel-level analysis requires predictions and labels |
| Latency, Tables 4 and 8; S21–S24 | `cpp/flood/`, `cpp/cloud/`, `src/benchmarking/` | `configs/benchmark/`, retained latency CSVs | K100, compiled artifacts, runtime plans and fixed input |
| Compilation / runtime resources; S6 and S10 | Session compilation and memory collection modules | Retained compilation/runtime CSVs | Matching Linux/container/cgroup and device telemetry records |

`scripts/reproduce_tables.py` renders the 25 retained numerical/protocol tables indexed in `results/table_index.json`. It does not reproduce the supplementary correspondence table S1 or access-policy table S27.

The diagnostic figure entry point requires complete per-scene block records. It does not certify regeneration of every manuscript figure. Selected runtime source is provided for the measured inference paths; the private orchestration, remote-host management and all original long-duration reliability workflows are not bundled.

The compact statistical entry point preserves the task-specific estimands. Its output is separate from the retained published tabulations and is written only to a new output directory.
