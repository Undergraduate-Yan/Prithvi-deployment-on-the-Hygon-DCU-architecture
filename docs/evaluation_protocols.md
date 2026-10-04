# Evaluation protocols and interpretation

## Flood

The 90-image public pool was exposed during deployment development. All associated scores, paired uncertainty estimates and margin checks are descriptive. The unused official validation reserve is listed separately in `manifests/datasets/flood/official_validation_reserve_ids.txt`.

Runtime index 0 is labelled `Ghana_313799` in retained records, while its actual input source is `Ghana_1078550`. The retained files preserve that observation rather than silently relabelling inputs. The clean-89 analysis omits this index post hoc and does not create an independent test set.

Flood metrics use valid pixels only. Boundary IoU uses the fixed symmetric two-pixel band defined by intersection of foreground and background dilations with a 5 × 5 neighborhood. Pooled confusion-derived mIoU differs from the mean of defined scene-level mIoU values.

The descriptive paired bootstrap uses 10,000 draws and seed 42, omitting undefined pairs only for the corresponding metric. The one-sided lower bound is the fifth percentile. The retained descriptive margin implementation uses a strict `lower > margin` comparison. The tolerated decreases are 0.005 for mIoU and 0.01 for water and boundary IoU; agreement is compared with a decrease of 0.005 from reference agreement 1.

## Cloud

The evaluation contains 300 region-disjoint scenes. Each 2000 × 2000 scene uses 81 tiles of 224 × 224 pixels; overlapping logits are averaged. The class order is clear sky, thick cloud, thin cloud and cloud shadow. Ignore label is 255.

The cloud bootstrap resamples scenes as paired units, sums the selected scene confusion matrices, then computes each pooled metric. It is not a bootstrap of unweighted per-scene mIoU means. Agreement uses the selected scenes' agreeing and valid pixel counts. The retained full-precision ratios and integer valid counts permit those agreement counts to be recovered; the compact analysis rejects a non-integral reconstruction.

The 10,000 draws use seed 42. The fifth-percentile lower bound must be at least the negative allowed decrease: 0.005 for mIoU, macro-F1 and agreement, and 0.01 for each class IoU. FP16 meets the retained requirements. The mixed candidate's agreement is 0.994727 and its lower bound on agreement difference is -0.005969, which fails the agreement requirement.

## Latency

Controlled measurements use five fresh processes per node and variant, 50 warm-ups and 200 measured calls per process. C++ Runner-A includes all sessions, final synchronization and FP32 logits transfer to the host. The single-session flood comparison uses a Python I/O-binding runner. Scene loading, cloud tiling, stitching and scoring are excluded from model-call timing.

Compare results within the same node, runner, graph topology and scope. A pooled-call median differs from a median of process medians. Diagnostic configuration timing and sustained-loop throughput are documented separately from the controlled benchmark.
