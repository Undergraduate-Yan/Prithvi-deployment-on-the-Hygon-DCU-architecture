# Prithvi-EO-2.0 在海光 K100 AI 加速卡上的可审计部署

本仓库是论文对应的 Git 精简版源码与证据包，包含正式 Python/Bash 工具链、冻结协议、小型结果摘要、部署工具以及中英文 LaTeX 源文件。

## 为什么没有直接放入模型

原始归档约 5 GB，其中大部分是 checkpoint、ONNX、MXR 缓存、测试张量、日志和 profiler 输出。它们不适合普通 Git，也可能受到模型/数据许可证限制。因此：

- Git 只保存源码、协议、关键 JSON/CSV、论文和文件身份；
- 大文件放在对象存储、云盘、Release 或 Git LFS；
- 每个必要外部 artifact 通过字节数和 SHA256 验证；
- 仓库中不保存 SSH 指令、密码、私有主机地址或账号。

## 论文对应结论

| 结论层 | 冻结结果 |
|---|---|
| 任务精度 | Full FP16 与 M0--M5 均通过固定90样本应用门槛 |
| 严格 logits | CPU--MIGraphX 单样本严格诊断仍为 failed，不被任务准入覆盖 |
| 同拓扑性能 | M5 为 22.080 ms，25段 FP32 为 22.209 ms，只表述为时延持平 |
| 同 CLI 部署 | M5 为 26.309 ms，Full FP16 为 15.151 ms；M5 慢 73.642% |
| 完整部署容量 | M5 比 Full FP16 小 23.463% |
| 最终建议 | Full FP16 为速度优先主方案，M5 为存储优先备选 |

## 目录导航

- `src/baseline/`：真实 checkpoint 前向、FP32/FP16 精度和基础性能。
- `src/phase11/`：图兼容改写、25段拓扑、M0--M5、K100评估和性能。
- `src/phase11/phase11_minimal_deployment/`：不可变部署包、冷启动、恢复、双节点、60分钟验收。
- `src/phase11/phase11_deployment_cli_pair_benchmark/`：M5 与 Full FP16 同 CLI 配对性能。
- `results/`：论文使用的最终小型结果证据。
- `paper/`：中英文论文源文件、矢量图和终稿 PDF。
- `artifacts/`：不进入 Git 的模型、缓存与数据身份。
- `docs/`：复现顺序、代码映射和证据边界。

## 建议使用顺序

```bash
# 1. 检查仓库是否误放大文件、凭据或损坏 JSON/Python
python scripts/repo_check.py

# 2. 查看论文指标摘要
python scripts/summarize_results.py

# 3. 验证外部模型与缓存
python scripts/verify_external_artifacts.py \
  --manifest artifacts/required_artifacts.example.json \
  --root /你的/artifact/根目录
```

随后按 [复现指南](docs/reproduction.md) 执行。K100 部分必须使用匹配的厂商运行时，不能用普通 PyPI ONNX Runtime 代替 MIGraphX 环境后声称复现。

## 上传 Git 前

1. 确认使用私有仓库还是公开仓库。
2. 决定许可证和作者信息；当前仓库不会擅自替你选择许可证。
3. 运行 `python scripts/repo_check.py`。
4. 查看 `git status`，确认没有 `.env`、模型、数据或远端登录信息。
5. 大 artifact 使用独立对象存储或 Git LFS，并发布 SHA256 manifest。

完整清单见 [Git 发布检查表](docs/git_publish_checklist.md)。

