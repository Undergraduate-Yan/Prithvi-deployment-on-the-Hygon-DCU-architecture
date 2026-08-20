# Phase 11：M0–M5 混合精度 kernel 证据（v3）

## 为什么改为 v3

此前首个真实 trace 在动态 `hipprof --session ... --start` 控制处超时，外层 `hipprof` 也无法正常
收口。v3 回到 Phase 10 已验证成功的方式：由外层 `hipprof` 直接启动被测 Python 进程，全链路不再
使用 `--trace-off`、`--session`、`--start`、`--stop` 或 `--flush`。

为了避免直接外层 trace 把前序 block 也记入 kernel 统计，每个去重任务分成两个完全独立的容器：

1. **未跟踪 prepass**：用冻结样本和 canonical candidate，按 device OrtValue 顺序执行目标 block
   之前的分段，生成目标 block 的真实 float32 边界 `boundary_input.npy`；
2. **直接 hipprof target**：不挂载原始样本，不构建任何前序 session，只加载目标 block 的一个
   ONNX+MXR session，并对冻结边界输入执行 20 次。

边界 prepass 会登记原始样本文件 SHA、归一化输入 tensor SHA、candidate manifest SHA、每个前序
block 的 ONNX/MXR SHA、目标 block 的 ONNX/MXR SHA、边界 NPY SHA 与边界 tensor SHA。宿主机在
启动 trace 前再次生成 boundary/result size+SHA receipt，target 和 summarizer 都会复核。

## K100-2 执行命令

把当前目录工具上传到一个**新的**只读远端目录，例如
`/var/tmp/phase11_mixed_kernel_tools_v3`。必须给本次运行使用一个从未存在的新结果根；不得覆盖动态
session 失败的 v2/v3 结果或其中的 `tool_snapshot`。

```bash
PHASE11_M5_CANDIDATE_ROOT=/var/tmp/20260818-phase11-mixed-m5-node2-v2 \
bash /var/tmp/phase11_mixed_kernel_tools_v3/run_phase11_mixed_precision_kernel_evidence_node2.sh \
  /var/tmp \
  /var/tmp/20260818-phase11-mixed-m0-m4-node2 \
  /path/to/frozen_real_sample.pt \
  /var/tmp/20260818-phase11-mixed-kernel-evidence-m0-m5-direct-v3-node2 \
  M0,M1,M2,M3,M4,M5 \
  20
```

首条修复验证建议用 M0 的第一个 identity-sorted task 做 block 0 smoke：

```bash
PHASE11_KERNEL_TASK_LIMIT=1 \
bash /var/tmp/phase11_mixed_kernel_tools_v3/run_phase11_mixed_precision_kernel_evidence_node2.sh \
  /var/tmp \
  /var/tmp/20260818-phase11-mixed-m0-m4-node2 \
  /path/to/frozen_real_sample.pt \
  /var/tmp/20260818-phase11-mixed-kernel-evidence-m0-direct-v3-smoke-node2 \
  M0 \
  20
```

`PHASE11_KERNEL_TASK_LIMIT=1` 会选择按 identity 排序后的首个任务，即 M0 的 block 0 INT8 QDQ。
它会生成一条完整的 prepass、direct-hipprof 和 summary 链；若所选 block 的 placement 与 I8II 门槛
通过，脚本退出码为 0，但 summary 状态明确为
`selected_task_smoke_passed_no_full_mapping_claim`，不能据此声称 M0 的 24 个 block 已全部验证。确认
该 smoke 通过后，应取消这个环境变量并换用新的结果根执行完整 33 条 trace。

## 不变的约束

- 宿主机必须是 `machine2`，启动时没有其他运行中容器；
- 镜像必须完整匹配
  `sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01`；
- static `lsmod` 必须为 819664 B、SHA256
  `9328bce5211d360b0c7acaafdc9c4b927ac3773f7a7422600470afc5478f8e12`，且其目录是每个容器
  `PATH` 第一项；
- 原始样本挂载到 prepass 时必须保留 `.npy/.npz/.pt/.pth` 后缀；trace 容器不挂载原始样本；
- 每个 prepass/trace 容器默认最多等待 1800 秒，超时会保留日志与 inspect 后强制移除；确需调整时
  只允许通过 `PHASE11_KERNEL_CONTAINER_TIMEOUT_SECONDS` 设置 60–7200 秒；
- 完整 M0–M5 仍是 33 条去重 trace（24 条 INT8、9 条 FP16）以及 144 行 candidate/block
  映射，M5 支持不变。

## 结果解释

`RUN_STATUS=completed_all_required_kernel_gates_passed` 且退出码 0，才允许逐 block 报告预期 kernel：

- INT8 QDQ block：`I8II` 调用数不少于 target 重复数；
- FP16 block：`HBH` 调用数不少于 target 重复数；
- ORT placement 同时满足 MIGraphX 事件 > 0、CPU 事件 = 0；
- Docker inspect 确认直接外层 hipprof、无动态 session 参数、target 只接收一个边界输入。

直接外层 hipprof 会同时看到目标 session 创建、MXR 加载、边界 H2D 上传、首次输出分配、20 次
目标 block 调用，以及为 finite/SHA 校验进行的最终输出 D2H。因此 Q/DQ、Cast、layout 与 Memcpy
仍可用于识别该 direct-trace 范围内的开销来源，但不能称为“纯稳态推理占比”。Memcpy 的 HIP/HSA
API 时间也不能与 GPU kernel 时间相加。

本证据无论通过或失败，都不覆盖历史 strict logits failure，也不能单独推出模型级加速、full/task
head INT8 准入或部署稳定性。
