# Phase 11 条件候选 M5

M5 只在 M4 仍有明显数值问题时追加，不覆盖 M0–M4 的冻结证据。精度映射固定为：

- FP16：block 0、14–21；
- INT8 QDQ：block 1–13、22、23；
- Head：已修复的 FP32 `fpn4 MaxPool barrier`。

构建器从已终结的 M4 中逐字节复用 block 0、14、17、18 的 ONNX/MXR、其余 INT8 段及 head；只从同一冻结 FP16 full artifact 提取 block 15、16、19、20、21。新增 FP16 段保持图外 FP32 I/O、图内显式 Cast，并把 LayerNorm 统计保留在 FP32 island。

## 1. 构建与五段增量缓存

仅在 node2 没有运行 M0–M4 正式性能任务时执行。启动器与正式性能启动器共用 `flock`；冲突时会在任何容器启动前以退出码 75 结束。

两个启动器均锁定并优先使用证据 bundle 中的静态 `lsmod` shim（819,664 B，SHA256 `9328bce5...8f8e12`），避免官方镜像内 `lsmod` 递归派生进程。准入启动器还会保留输入样本的 `.npy`、`.npz`、`.pt` 或 `.pth` 后缀；没有受支持后缀时会在启动容器前失败。

```bash
bash run_phase11_mixed_precision_m5_build_and_cache_node2.sh \
  /var/tmp/20260818-phase11-mixed-m0-m4-node2 \
  <冻结FP16-full.onnx> \
  <FP16导出报告.json> \
  <INT8量化报告.json> \
  /var/tmp/20260818-phase11-mixed-m5-node2
```

成功标志：

```text
/var/tmp/20260818-phase11-mixed-m5-node2/RUN_STATUS
  cache_finalized_static_pass

/var/tmp/20260818-phase11-mixed-m5-node2/M5/mixed_precision_manifest.final.json
```

缓存编译只处理 15、16、19、20、21，且每段必须满足 MIGraphX 事件大于 0、CPU provider 事件等于 0、输出有限。严格数值列仍是诊断项，不是缓存准入前置。

## 2. 单样本与三轮固定 90 样本

```bash
bash run_phase11_mixed_precision_m5_admission_node2.sh \
  /var/tmp/20260818-phase11-mixed-m5-node2 \
  <冻结1x6x224x224样本> \
  <包含phase11_frozen_test90与FP32基线的项目根目录> \
  /var/tmp/20260818-phase11-mixed-m5-admission-node2
```

正式准入汇总位于：

```text
/var/tmp/20260818-phase11-mixed-m5-admission-node2/M5/three_run_summary/summary.json
```

它沿用 M0–M4 的同一门槛和通用评估器。三轮都通过后，才能进入 M5 性能测试；未通过则 M5 仅保留为失败消融。

## 3. 独立正式性能（仅在任务准入通过后）

M5 使用单候选 schedule，输出到全新目录，不合并或覆盖 M0–M4 的正式性能根目录：

```bash
bash run_phase11_mixed_precision_formal_performance_node2.sh \
  /var/tmp/20260818-phase11-mixed-m5-node2 \
  /var/tmp/20260818-phase11-mixed-m5-node2 \
  /var/tmp/20260818-phase11-mixed-m5-admission-node2 \
  <冻结样本> \
  <同协议25段FP32性能summary.json> \
  /var/tmp/20260818-phase11-mixed-m5-performance-node2 \
  M5
```

即使 M5 通过，也必须与 M0–M4 的精度、容量、时延和 kernel 证据一起做 Pareto 判断，不能仅凭 strict MAE 下降就替换 FP16 full 部署基线。
