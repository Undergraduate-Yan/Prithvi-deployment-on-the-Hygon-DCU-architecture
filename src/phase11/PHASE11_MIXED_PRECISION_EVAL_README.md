# Phase 11 M0–M4 验收与性能工具

这套工具只建立新的 `task-equivalence performance track`，不会覆盖 M0 已冻结的 strict-logits failed 结论。

## 前置条件

- 构建阶段使用 `mixed_precision_manifest.json`；正式验收统一使用缓存终结器产生的 `mixed_precision_manifest.final.json`。最终文件保持 schema `phase11_sensitivity_mixed_precision_candidate_v1`，并使用 status `cache_finalized_static_pass` 与 manifest_stage `phase11_sensitivity_mixed_precision_candidate_cache_final_v1`标记已具备25段缓存。
- `--bundle` 是文件系统授权根：manifest 及其所有相对 model/cache 目标都必须在该根下。路径仍以 manifest 所在目录为基准解析。
- M1–M4 新 FP16 分段的 `cache`/`cache_identity` 必须先编译并写入一份新的、身份锁定的 manifest。任何 null cache 都会在创建 ORT 会话前失败。
- 所有运行使用 ORT `1.19.2` + MIGraphX EP，并需可读的固定90样本项目目录。

## 候选验收顺序

以 M0 为例，其他候选只替换 manifest 和证据目录。每个 output 目录必须事先不存在。

```bash
python evaluate_phase11_mixed_precision_single.py \
  --bundle "$BUNDLE_ROOT" \
  --manifest "$CANDIDATE_ROOT/M0/mixed_precision_manifest.final.json" \
  --sample "$SAMPLE" \
  --output-dir "$EVIDENCE/single/M0/output"
```

单样本 operational gates 通过后，在三个新进程/容器中分别执行：

```bash
python evaluate_phase11_mixed_precision_test90.py \
  --bundle "$BUNDLE_ROOT" \
  --manifest "$CANDIDATE_ROOT/M0/mixed_precision_manifest.final.json" \
  --single-result "$EVIDENCE/single/M0/output/result.json" \
  --project "$PROJECT" \
  --output-dir "$EVIDENCE/test90/M0/run_1/output"
```

将 `run_1` 依次替换为 `run_2` 和 `run_3`，然后聚合：

```bash
python aggregate_phase11_mixed_precision_test90.py \
  --bundle "$BUNDLE_ROOT" \
  --manifest "$CANDIDATE_ROOT/M0/mixed_precision_manifest.final.json" \
  --run-dir "$EVIDENCE/test90/M0/run_1/output" \
  --run-dir "$EVIDENCE/test90/M0/run_2/output" \
  --run-dir "$EVIDENCE/test90/M0/run_3/output" \
  --output-dir "$EVIDENCE/test90/M0/three_run_summary"
```

90样本脚本的退出码：`0` 表示任务门槛通过，`3` 表示证据已完整产生但任务门槛失败，`2` 表示运行/身份/放置错误。

## 3×100 计时

只允许对三轮90样本聚合为 `passed` 的候选计时。默认旋转顺序为：

| trial | position 1→5 |
|---|---|
| 1 | M0, M1, M2, M3, M4 |
| 2 | M1, M2, M3, M4, M0 |
| 3 | M2, M3, M4, M0, M1 |

如有候选未通过三轮90样本，则 `--candidate-schedule` 必须使用“仅包含已通过候选、并按 M0→M4 排序”的有序子集；所有 trial 和最终聚合必须使用完全相同的子集字符串。

每个 trial/candidate 组合必须启动一个新进程/容器，例如 trial 2 position 1 的 M1：

```bash
python benchmark_phase11_mixed_precision_task_equivalence_trial.py \
  --bundle "$BUNDLE_ROOT" \
  --manifest "$CANDIDATE_ROOT/M1/mixed_precision_manifest.final.json" \
  --test90-three-run-summary "$EVIDENCE/test90/M1/three_run_summary/summary.json" \
  --sample "$SAMPLE" \
  --trial-index 2 \
  --candidate-position 1 \
  --candidate-schedule M0,M1,M2,M3,M4 \
  --container-image-id sha256:<64位完整image-id> \
  --runtime-fingerprint "$PERF/runtime_fingerprint.json" \
  --output-dir "$PERF/trial_02/position_01_M1/output"
```

`runtime_fingerprint.json` 必须由同一锁定镜像内的
`fingerprint_phase11_mixed_precision_performance_runtime.py` 生成。每个 trial 记录完整
image ID、Python PID、`started_at_utc`和运行时指纹身份。输出校验只读取第30次
warmup和第100次测量的结果，不会插入额外未计时推理。

全部15个组合完成后：

```bash
python aggregate_phase11_mixed_precision_task_equivalence.py \
  --run-root "$PERF" \
  --candidate-schedule M0,M1,M2,M3,M4 \
  --fp32-summary "$FP32_25SEG_PERF/summary.json"
```

聚合结果包含每个候选的 model-only 和 end-to-end logits `median/P95/P99/吞吐量`、三轮 median CV 及300次池化描述统计。只有 median CV `≤5%` 才允许绝对时延和同scope FP32加速比结论。
后续M5可使用新的run-root和 `--candidate-schedule M5` 独立追加，不覆盖已冻结M0–M4证据。

## 独立K100显存峰值

显存采样必须在全部计时容器退出后单独运行；运行期间不得有其他K100负载。

```bash
python measure_phase11_mixed_precision_k100_vram.py \
  --bundle "$BUNDLE_ROOT" \
  --manifest "$CANDIDATE_ROOT/M1/mixed_precision_manifest.final.json" \
  --test90-three-run-summary "$EVIDENCE/test90/M1/three_run_summary/summary.json" \
  --sample "$SAMPLE" \
  --container-image-id sha256:<64位完整image-id> \
  --runtime-fingerprint "$PERF/runtime_fingerprint.json" \
  --output-dir "$PERF/vram/M1/output"
```

这一轮只采样设备级 `mem_info_vram_used`：从加载会话前基线，经25个常驻会话、
固定设备OrtValue，对model-only和end-to-end各执行30次预热与100次不计时运行。它报告设备总峰值和相对基线增量，
不产生任何时延结论。

同协议FP32 25段显存使用 `run_phase11_fp32_segment25_same_scope_vram_node3.sh`。它用
`measure_phase11_k100_vram_around_command.py` 包裹一次丢弃性FP32 30+100运行；包裹轮的所有时延输出均不允许作结论，
只保留同一sysfs口径的显存峰值。

## 容量、Pareto与下一候选选择

完成性能聚合和每个候选的独立显存轮后，使用显式的
`CANDIDATE=PATH` 参数连接manifest、90样本、性能和显存证据：

```bash
python summarize_phase11_mixed_precision_capacity_pareto.py \
  --bundle "$BUNDLE_ROOT" \
  --performance-summary "$PERF/summary.json" \
  --fp32-performance-summary "$FP32_25SEG_PERF/summary.json" \
  --fp32-vram-result "$FP32_25SEG_VRAM/output/result.json" \
  --manifest "M0=$CANDIDATE_ROOT/M0/mixed_precision_manifest.final.json" \
  --manifest "M1=$CANDIDATE_ROOT/M1/mixed_precision_manifest.final.json" \
  --test90-summary "M0=$EVIDENCE/test90/M0/three_run_summary/summary.json" \
  --test90-summary "M1=$EVIDENCE/test90/M1/three_run_summary/summary.json" \
  --vram-result "M0=$PERF/vram/M0/output/result.json" \
  --vram-result "M1=$PERF/vram/M1/output/result.json" \
  --output-dir "$PERF/pareto"
```

实际命令需对性能schedule中每个已准入候选各提供一组参数。输出中会给出
`accuracy_best/file_smallest/latency_lowest/balanced_pareto`和推荐进入kernel与部署验收的下一候选。
这不会直接取代FP16 full。

## 证据边界

- provider profile 只证明25段 MIGraphX>0/CPU=0，不证明 I8II/HBH kernel。
- task-equivalence 通过不等于 strict logits 通过。
- 该性能聚合器不会计算FP32加速比；只有单独通过的同协议FP32摘要才能支持该结论。
- `process_max_rss_kib` 是主机进程内存，不是K100显存证据。
- K100显存证据来自单独、不计时的设备级sysfs采样，不得与timed loop同时运行。
- Pareto选择只确定后续kernel/部署验收顺序；在I8II、FP16-full同CLI对比和稳定性通过前，FP16 full仍是正式部署基线。
