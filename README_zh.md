# 面向海光DCU架构的Prithvi部署

本仓库对应论文 **Onboard Deployment of Remote Sensing Foundation Models for the Hygon DCU Architecture**，提供分块数值诊断、精度分配、执行图重构、任务评价与性能测量代码，以及可用于结果复查的派生记录。

作者：严晨钢、谭海宁、尹龙祥、王有为、罗杰、邱吉冰。通讯邮箱：qiujibing@ict.ac.cn。

## 研究范围

洪水任务采用25会话诊断拓扑和13会话对照拓扑。90幅公共池影像曾在开发期间被访问，相关准确率和统计结果采用描述性解释。

云任务采用修复后的14会话拓扑，在300个此前未使用、区域不重叠的场景上进行评价。FP16满足保留的评价要求；混合精度候选的预测一致性未达要求。

所有部署结论均基于地面K100实验，适用范围不包括已完成在轨运行验证。

## 使用入口

1. 阅读 `docs/paper_mapping.md`，定位论文内容对应的代码与输入。
2. 查看 `environment/`，区分普通电脑分析环境、模型准备环境、离线量化环境与K100推理环境。
3. 使用 `scripts/reproduce_tables.py` 将保留的CSV表格转成Markdown。此操作只格式化已有结果。
4. 使用 `scripts/analyze_results.py` 从逐场景派生记录进行统计复算，无需重新推理。
5. 进行模型导出、编译和执行前，按照 `docs/data_and_models.md` 准备外部材料。

```bash
python scripts/reproduce_tables.py --output-dir outputs/tables
python scripts/analyze_results.py flood --output-dir outputs/flood-analysis
python scripts/analyze_results.py cloud --output-dir outputs/cloud-analysis
```

这些命令是使用说明，仓库不声称已在读者环境中执行通过。

## 目录说明

| 目录 | 内容 |
|---|---|
| `src/` | 研究方法及数值计算实现 |
| `cpp/` | 洪水与云任务的C++推理运行器 |
| `scripts/` | 统一命令入口 |
| `configs/` | 协议参数、精度配置与运行清单模板 |
| `environment/` | 软件与硬件环境要求 |
| `manifests/` | 样本标识、外部工件身份与源文件哈希 |
| `results/` | 论文和补充材料对应表格、逐场景派生记录 |
| `figures/` | 诊断绘图入口及其输入范围 |
| `docs/` | 方法、复现步骤和论文对应说明 |

模型权重、影像、ONNX图、MXR缓存及厂商运行库需要另外获取。一些构建程序要求精确匹配的校准清单和构建报告，仓库本身不包含端到端复现的全部输入。

洪水运行索引0的记录名称与实际来源不同，说明见 `docs/evaluation_protocols.md`。清单中的历史字段名仅用于数据格式兼容，不能据此将洪水评价解释为独立确认。

代码许可见 `LICENSE`；目前未指定开源许可证。第三方组件许可见 `THIRD_PARTY_NOTICES.md`。
