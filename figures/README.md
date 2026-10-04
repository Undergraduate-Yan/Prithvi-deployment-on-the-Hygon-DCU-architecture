# Diagnostic plots

`scripts/plot_block_diagnostics.py` delegates to the retained block-record aggregator. Supply its `--input`, `--input-manifest`, `--output` and optional `--plot` arguments. It requires the complete per-scene block records, not the rounded S7 summary.

The source contains the diagnostic plotting implementation used with those records. This directory does not assert regeneration of every final manuscript figure. The paper-to-code map identifies the numerical inputs that are included and those requiring separate access.
