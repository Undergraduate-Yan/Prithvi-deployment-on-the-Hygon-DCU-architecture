# Phase 11 混合精度 kernel 证据边界

状态：`passed`。

- **Provider placement 与 kernel 精度分开。** ORT profile 中 MIGraphX 事件大于 0、CPU 事件为 0，只能证明该分段落在 MIGraphX，不能单独证明执行了 INT8 或 FP16 kernel。
- **跟踪范围。** 每个 block 先由未跟踪 prepass 用 device OrtValue 执行其前序段并生成带 SHA 与完整 manifest/model/MXR lineage 的 float32 边界输入；随后直接外部 `hipprof` 只启动目标单段进程，不使用 `--trace-off`、`--session` 或 start/stop/flush 控制。
- **INT8 证明口径。** 只有对应模型与 MXR 的 hipprof 记录中出现 `I8II`，并且调用数不少于被跟踪的重复次数，才把该 block 标记为原生 INT8 GEMM 已确认。
- **FP16 证明口径。** 同理，只有出现 `HBH` 才把该 block 标记为 FP16 GEMM 已确认。
- **开销口径。** Q/DQ、Cast、layout conversion 按 GPU kernel 时间分别统计；Memcpy 采用 hipprof 的 HIP/HSA API 调用时间并使用独立分母，二者不得相加为端到端时间。直接外部 trace 还包含目标 session 创建、边界 H2D 上传、输出分配，以及用于 finite/SHA 校验的最终输出 D2H，因此转换占比不表述为纯稳态占比。
- **复用口径。** 多个候选引用完全相同 SHA256 的静态 ONNX 与 MXR 时，只采集一次 kernel 证据；该证据不跨不同 artifact 身份复用。
- **不能推出的结论。** 本证据不覆盖既有 strict logits failure，不单独证明模型级加速，也不代表 full/task-head INT8 或部署稳定性已经通过。
