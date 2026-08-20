# Evidence boundaries

The project uses five independent evidence layers.

| Layer | What it supports | What it does not support |
|---|---|---|
| Task utility | Accuracy on the fixed 90-image Sen1Floods11 test set | Strict CPU/provider logit equivalence or broad generalization |
| Provider placement | Positive MIGraphX events and zero CPU provider events | Actual arithmetic datatype of every kernel |
| Realized kernels | I8II/HBH GEMM paths in traced blocks | Whole-graph INT8/FP16 execution or model-level acceleration |
| System performance | Latency/throughput under a named topology and timing scope | Precision-only hardware peak comparison across different graph topologies |
| Operations | Bundle integrity, cold start, recovery, cross-node smoke, stability | Statistical model quality outside the evaluated task and stack |

Permanent boundaries:

- The strict single-image CPU--MIGraphX logit diagnostic remains failed for Full FP16 and M0--M5 under its predefined tolerances.
- All mixed candidates pass the separate fixed application-level task criteria.
- M5 versus 25-segment FP32 is latency parity, not a formal INT8 speedup.
- M5 versus monolithic Full FP16 is a production-style system comparison that combines graph topology and precision.
- Results apply to one checkpoint, one task, batch 1, and the recorded K100 software stack.

