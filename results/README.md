# Retained study records

The numerical and protocol CSVs indexed by `table_index.json` match the corresponding supplementary tables. They are copied records, not outputs of a new experiment. Full-precision scene records support separate offline analysis.

- `flood/per_scene_metrics.csv` retains source sample identifiers, confusion counts, metric values and prediction identities for six configurations. Runtime index 0 has the source-identity discrepancy explained in `docs/evaluation_protocols.md`.
- `cloud/per_scene/*_confusion.csv` retains the 300 scene confusion matrices for each model role.
- `cloud/per_scene_metrics.csv` retains full-precision scene metrics and agreement ratios.
- `cloud/bootstrap_full_precision.csv` and `cloud/metrics_full_precision.csv` retain the original numerical summaries.

The cloud role names map as follows: `Cloud-RCS-FP32-Compat` → Cloud-14S-FP32; `Cloud-RCS-FP16-Opt-v2` → Cloud-14S-FP16; `Cloud-RCS-MP-Task-v2` → Cloud-14S-MP-X. Original role identifiers remain intact for pairing.

Reading a table does not reproduce the underlying model run. Use each script's stated input requirements and keep regenerated output outside this directory.
