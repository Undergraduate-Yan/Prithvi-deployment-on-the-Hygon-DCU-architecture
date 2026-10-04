# Deployment methodology

The implementation connects numerical compatibility and precision selection to compilation feasibility and task-level performance.

1. Prepare task-specific six-band inputs and preserve normalization, tensor shape and sample order.
2. Export the fixed task model. Flood and cloud use separate checkpoints and heads.
3. Compare intermediate CPU and K100 outputs through the diagnostic block graph.
4. Generate precision candidates using the configuration set and calibration records.
5. Restructure graphs under compilation-memory and numerical constraints.
6. Evaluate task predictions and benchmark the exact selected execution graph.

For block implementations `f_i^C` and `f_i^K`, and accumulated inputs `x_i^C` and `x_i^K`, the diagnostics are:

\[
L_i=\operatorname{MAE}(f_i^C(x_i^C),f_i^K(x_i^C)),
\]
\[
C_i=\operatorname{MAE}(f_i^C(x_i^C),f_i^K(x_i^K)),\qquad
P_i=\operatorname{MAE}(f_i^K(x_i^C),f_i^K(x_i^K)).
\]

Only the local comparison holds the input fixed. The 64-scene, 24-block study contains 1536 block records. The retained flood C5 configuration protects blocks 0, 8, 10, 14, 15, 17, 18, 19 and 20 in FP16. This is a deterministic feasible configuration; its results do not establish a globally optimal precision map.

The cloud compatibility repair splits the final decoder-containing session at retained feature boundaries without changing learned weights. It produces a 14-session reference used for the cloud precision comparison.

Source QDQ structure, provider placement, realized kernel precision, task accuracy, latency and reliability are distinct observations. A reduction in bundle size does not substitute for task acceptance.
