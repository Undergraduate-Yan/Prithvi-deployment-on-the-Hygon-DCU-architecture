# 海光 K100 部署全过程与结果

本目录是第三周 K100 实验的统一、本地、未压缩归档。内容按真实实验顺序整理，保留成功结果、失败诊断、复现工具、原始 ONNX、正式性能、kernel 证据及部署验收；临时缓存、凭据辅助程序、已完成重组的传输分片和不完整下载已清理。

## 建议先看

1. [最终收口报告](00_导航与最终结论/Phase11_混合精度与部署收口报告_20260818.md)
2. [实验结果总表](00_导航与最终结论/K100_Phase11_混合精度与部署收口总表.xlsx)
3. [关键结果速查](00_导航与最终结论/关键结果速查.md)
4. [本地归档与远端边界](00_导航与最终结论/本地归档与远端边界.md)
5. `00_导航与最终结论/文件清单_SHA256.csv`：归档内文件大小与 SHA-256

## 目录说明

| 目录 | 内容 |
|---|---|
| `00_导航与最终结论` | 最终报告、总表、关键结果、文件清单、移动与清理记录 |
| `01_环境迁移与基础前向` | 模型迁入、源 ONNX、运行时预检、真实样本前向与早期环境证据 |
| `02_FP32_FP16基线` | K100 FP32、FP16 精度和性能基线及独立确认 |
| `03_ONNX消融` | FP32、FP16、INT8 full/backbone/task-head 的 ONNX 预检、精度和性能消融 |
| `04_INT8兼容与敏感层` | full INT8 失败诊断、bias DQ fold、block 0/23/late-MLP 等早期敏感层实验 |
| `05_MIGraphX环境诊断` | MIGraphX toy、FP16 兼容性与 task-head 分支诊断 |
| `06_Phase11_FP32同协议基线` | FP32 25 段、barrier/重写/内存诊断、90 样本和正式性能 |
| `07_Phase11_INT8_Backbone` | INT8 backbone 25 段、数值定位、跨节点缓存与替代拓扑证据 |
| `08_Phase11_M0_M5混合精度` | M0–M5 任务、性能、显存、Pareto 与确认实验 |
| `09_Kernel证据` | hipprof 汇总和 candidate/block kernel 矩阵 |
| `10_部署包与验收` | 同 CLI M5/FP16 配对、双节点 smoke、冷启动、恢复、60 分钟稳定性及最终聚合 |
| `11_复现工具` | Day 1–11 早期脚本和 Phase 11 正式构建、评估、kernel、部署工具链 |
| `12_报告论文与汇报` | 分阶段报告、论文 TeX/PDF、答辩 PPT 及页面源码 |

## 推荐复现顺序

1. 从 `01_环境迁移与基础前向/Day5_INT8_migration_staging` 核对源模型及 sidecar。
2. 复核 `02_FP32_FP16基线` 和 `03_ONNX消融`，确认固定 90 样本与指标口径。
3. 阅读 `04_INT8兼容与敏感层` 和 `07_Phase11_INT8_Backbone`，理解 strict logits failure 及 25 段路线来源。
4. 用 `06_Phase11_FP32同协议基线` 作为匹配拓扑的 FP32 分母。
5. 用 `08_Phase11_M0_M5混合精度` 重现任务、性能、容量和 Pareto 结果。
6. 用 `09_Kernel证据` 验证 INT8 block 的 `I8II` 与 FP16 block 的 `HBH` 路径。
7. 按 `11_复现工具/Phase11正式工具链/phase7_10_dtk25042/phase11_minimal_deployment/FINAL_DEPLOYMENT_ACCEPTANCE_README.md` 构建 bundle 并执行部署验收。
8. 以 `10_部署包与验收` 的最终聚合 JSON 为部署结果依据。

## 最终结论

- FP16 full 是正式部署首选。
- M5 是可部署的容量优先混合精度候选，完整 bundle payload 比 FP16 full 小 `23.463061%`。
- 同一部署 CLI 下，M5 median 时延为 FP16 full 的 `1.736423×`，因此没有替换 FP16 的时延优势。
- M0–M5 均通过任务级精度门槛；历史 strict logits 等价仍保持 failed。
- M5 和 FP16 full 均通过 K100-2/K100-3 smoke、5 次冷启动、3 次恢复和 60 分钟稳定性。

## 重要边界

- 归档中未写入口令，也未保留会扫描本机凭据文件的环境专用 SSH/SFTP 辅助程序。
- JSON 中记录的 `/var/tmp/...` 等路径是实验发生时的远端历史路径，未修改，以保持证据真实性。
- 本地包含源 ONNX 与完整构建/评估工具，但没有最终 `.mxr` 缓存和两个远端 bundle 的全部 payload；详见“本地归档与远端边界”。
- strict logits failure、M0 原始性能 unstable 和被排除的失败尝试均属于正式证据，不应删除或改写。

