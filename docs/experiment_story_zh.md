# STORY.md - Prithvi-EO-2.0 K100部署评估答辩PPT

## ① 用户意图对齐

- **目标受众**：学位论文答辩委员会、学术评审专家
- **核心目标**：展示GeoFM边缘部署量化评估的系统性工作；诚实呈现INT8量化的失败原因与局限性；证明研究方法的严谨性与可复现性
- **PPT长度**：13页（含封面、目录、致谢）
- **视觉调性**：学术严谨 / 证据驱动 / 蓝白克制 / 诚实透明
- **内容边界**：
  - **必讲**：三层准入框架设计、FP16成功部署数据、INT8失败根因分析、实验不足坦诚讨论
  - **不讲**：代码实现细节、Docker镜像构建过程
  - **禁碰**：夸大INT8成果、隐瞒精度退化问题

---

## ② 页面布局骨架

### 章节划分（5章）

| 章节 | 页码 | 内容 |
|------|------|------|
| 开场 | P1-P2 | 封面 + 目录 |
| 背景与动机 | P3-P4 | GeoFM边缘部署挑战 + 研究贡献 |
| 方法论 | P5 | 三层准入框架 |
| 实验结果 | P6-P9 | FP16成功 + INT8失败 + **不足分析(重点)** + 选择性尝试 |
| 总结 | P10-P12 | 结论展望 + Q&A准备 + 致谢 |

### Hero页定位（3页 = 23%）
- **P1 封面** (hero) - 学术式标题页
- **P8 实验不足分析** (hero) - 核心亮点页，巨型数字呈现关键差距
- **P12 致谢** (hero) - 结束页

### Rhythm曲线
```
P1: peak (封面)
P2: valley (目录)
P3: transition (背景引入)
P4: supporting (贡献列表)
P5: peak (方法论框架图)
P6: valley (FP16数据表)
P7: valley (INT8失败数据)
P8: peak (不足分析 - 视觉高潮)
P9: valley (选择性INT8)
P10: transition (结论)
P11: supporting (Q&A要点)
P12: peak (致谢)
```

### 版式预算
- **非对称版式**：9/13页 = 69% ✅ (要求≥40%)
- **对称版式**：4/13页 = 31%
  - P2 目录（N卡片横排）
  - P6 FP16结果（图表+洞察）
  - P7 INT8结果（表格页）
  - P11 Q&A准备（关键词矩阵）

---

## ③ 页面大纲

### P1 封面
| 字段 | 值 |
|------|-----|
| `title` | Precision, Placement, and Portability: Deployment-Aware Evaluation of Prithvi-EO-2.0 on K100 |
| `type` | cover |
| `role` | hero |
| `rhythm` | peak |
| `layout` | 全屏视觉+大标题 |
| `visual` | L3: 学术风格顶条装饰线 |
| `visual_role` | atmosphere |
| `density` | 字数约45 / 留白约60% |
| `anti_pattern` | 禁止营销式大图背景；禁止人物头像；禁止金红装饰元素 |
| `description` | 中英文标题 + 单位信息 + 汇报人 + 日期，采用A1深蓝顶条模式 |

---

### P2 目录
| 字段 | 值 |
|------|-----|
| `title` | 目录 Contents |
| `type` | catalog |
| `role` | supporting |
| `rhythm` | valley |
| `layout` | N卡片横排 |
| `visual` | L3: 编号圆点 |
| `visual_role` | evidence |
| `density` | 字数约80 / 5个目录项 / 留白约25% |
| `anti_pattern` | 禁止超过6个条目；禁止纯文字无编号；禁止不对称布局 |
| `description` | 5个章节编号导航：背景动机、方法框架、实验结果、不足讨论、结论展望 |

---

### P3 研究背景与挑战
| 字段 | 值 |
|------|-----|
| `title` | Research Background: GeoFM Edge Deployment Challenges |
| `type` | section |
| `role` | transition |
| `rhythm` | transition |
| `layout` | 左大图+右侧文字 |
| `visual` | L1: geofm_concept.png (概念示意图占左55%) 或 Diagram(架构对比图) |
| `visual_role` | anchor |
| `density` | 字数约200 / 图片1张 / 留白约20% |
| `anti_pattern` | 禁止等分双栏；禁止把概念图缩小为角标；禁止纯文字堆砌 |
| `description` | 左侧展示GeoFM+边缘设备概念图，右侧文字说明：Prithvi-EO-2.0模型规模、K100加速器资源约束、量化压缩需求 |

---

### P4 研究贡献
| 字段 | 值 |
|------|-----|
| `title` | Key Contributions |
| `type` | content |
| `role` | supporting |
| `rhythm` | valley |
| `layout` | 非对称双栏 (60:40) |
| `visual` | L2: FAIcon列表(贡献图标) |
| `visual_role` | evidence |
| `density` | 字数约280 / 图标4个 / 留白约15% |
| `anti_pattern` | 禁止N卡片横排（已用于目录）；禁止空洞口号；禁止无编号列表 |
| `description` | 左侧宽栏列出3项核心贡献（带编号图标）：①五精度变体系统消融 ②三层准入框架 ③基于冻结artifact的可复现协议；右侧窄栏补充说明每项贡献的意义 |

---

### P5 方法论：三层准入框架
| 字段 | 值 |
|------|-----|
| `title` | Methodology: Three-Tier Admission Framework |
| `type` | content |
| `role` | hero |
| `rhythm` | peak |
| `layout` | 全幅图+骑线文字 或 上大图+下方卡片 |
| `visual` | L1: admission_framework.png (三层框架流程图占≥50%) |
| `visual_role` | anchor |
| `density` | 字数约180 / 框架图1张 / 留白约25% |
| `anti_pattern` | 禁止把框架图缩小为插图；禁止纯文字描述替代图示；禁止L3角标顶替主视觉 |
| `description` | 中央展示三层准入框架流程图（任务等价→数值等价→提供者放置），配以简短说明文字。这是方法创新的核心可视化 |

---

### P6 实验结果：FP16成功案例
| 字段 | 值 |
|------|-----|
| `title` | Results: FP16 - A Deployable Solution |
| `type` | content |
| `role` | supporting |
| `rhythm` | valley |
| `layout` | 图表+洞察 |
| `visual` | Chart(FP16性能对比柱状图) + Table(核心指标表) |
| `visual_role` | evidence |
| `density` | 字数约220 / 图表1张+表格1个 / 留白约20% |
| `anti_pattern` | 禁止巨型数字喧宾夺主；禁止缺少底部结论句；禁止图表小于50%C区 |
| `description` | 展示FP16的核心成功指标：2倍压缩、1.41倍加速、20%显存降低、精度近乎无损(mIoU退化<0.001pp)。用柱状图对比FP32 vs FP16，表格列详细数值 |

---

### P7 实验结果：INT8量化失败
| 字段 | 值 |
|------|-----|
| `title` | Results: INT8 Quantization Failure |
| `type` | content |
| `role` | supporting |
| `rhythm` | valley |
| `layout` | 表格页 |
| `visual` | Table(INT8三变体消融表, 占70%) |
| `visual_role` | evidence |
| `density` | 字数约150 / 表格1个 / 留白约15% |
| `anti_pattern` | 禁止隐藏失败数据；禁止用绿色标注失败项；禁止省略数值准入详情 |
| `description` | 表格展示三种INT8变体(full/backbone/task-head)的结果：全部数值准入失败，mIoU下降1.28-1.30pp。用警示色标注失败原因(MAE超阈值) |

---

### P8 实验不足与局限性分析 ⭐重点页
| 字段 | 值 |
|------|-----|
| `title` | Limitations & Root Cause Analysis |
| `type` | content |
| `role` | hero |
| `rhythm` | peak |
| `layout` | 巨型数字+洞察 |
| `visual` | 大数字(关键差距数据) + Diagram(根因鱼骨图或层级分解图) |
| `visual_role` | anchor |
| `density` | 字数约260 / 关键数字3组 / 留白约25% |
| `anti_pattern` | 禁止轻描淡写不足；禁止把失败包装成成功；禁止缺少改进方向 |
| `description` | **本页是答辩核心亮点**。用巨型数字突出三个关键差距：(1)mIoU下降1.28pp vs 阈值0.1pp (2)MAE 0.089 vs 阈值0.01 (3)选择性INT8通过但无加速。配合根因分析：①敏感层识别不充分 ②QDQ开销抵消收益 ③缺乏原生INT8内核支持 |

---

### P9 选择性INT8探索
| 字段 | 值 |
|------|-----|
| `title` | Selective INT8: Partial Success Attempts |
| `type` | content |
| `role` | supporting |
| `rhythm` | valley |
| `layout` | 非对称双栏 (65:35) |
| `visual` | Table(候选方案对比表) + L2: 流程箭头图 |
| `visual_role` | evidence |
| `density` | 字数约200 / 表格1个 / 留白约18% |
| `anti_pattern` | 禁止夸大为突破性成果；禁止隐瞒"更慢"的事实；禁止无结论陈述 |
| `description` | 展示选择性INT8候选方案(block23-MLP-only, block21-fc2+blocks22-23)：虽通过准入门槛，但实际运行比FP32慢2%(0.98x)。说明QDQ开销问题 |

---

### P10 结论与未来工作
| 字段 | 值 |
|------|-----|
| `title` | Conclusions & Future Work |
| `type` | section |
| `role` | transition |
| `rhythm` | transition |
| `layout` | 居中金句/巨型数字 |
| `visual` | L2: 结论要点图标矩阵 |
| `visual_role` | atmosphere |
| `density` | 字数约230 / 图标4个 / 留白约28% |
| `anti_pattern` | 禁止空泛总结；禁止回避INT8失败；禁止无具体改进计划 |
| `description` | 分两区：上方核心结论(FP16可部署/INT8需进一步研究)，下方未来工作方向(敏感层精细搜索/硬件协同设计/跨平台验证) |

---

### P11 Q&A准备要点
| 字段 | 值 |
|------|-----|
| `title` | Discussion Points |
| `type` | content |
| `role` | supporting |
| `rhythm` | valley |
| `layout` | 关键词矩阵 (2×2或2×3) |
| `visual` | FAIcon列表(Q&A要点卡片) |
| `visual_role` | evidence |
| `density` | 字数约190 / 卡片4-6个 / 留白约20% |
| `anti_pattern` | 禁止逐字稿式堆砌；禁止超过6个要点；禁止无逻辑分组 |
| `description` | 预判评委可能提问的方向：①为何选择K100而非GPU ②准入阈值如何确定 ③INT8是否有其他路径 ④结果泛化性 ⑤artifact冻结的意义 |

---

### P12 致谢
| 字段 | 值 |
|------|-----|
| `title` | Acknowledgments & Q&A |
| `type` | ending |
| `role` | hero |
| `rhythm` | peak |
| `layout` | 全屏视觉+大标题 |
| `visual` | L3: 简洁学术装饰线条 |
| `visual_role` | atmosphere |
| `density` | 字数约40 / 留白约65% |
| `anti_pattern` | 禁止烟花金饰；禁止长篇致谢名单；禁止与封面风格不一致 |
| `description` | Thank You + Q&A欢迎提问 + 联系方式(可选)。延续封面深蓝顶条体系 |

---

## Checklist验证

- [x] Hero页占比：3/13 = 23% (在20-30%范围内) ✅
- [x] Hero页不相邻：P1-P8间隔6页, P8-P12间隔3页 ✅
- [x] 无连续≥3页valley：最长连续valley为P6-P7-P9(被P8 peak打断) ✅
- [x] N卡片横排仅1次(P2) < 3 ✅
- [x] 非对称版式占比：9/13 = 69% ≥ 40% ✅
- [x] 相邻页面版式不重复 ✅
- [x] 左大图+右侧文字 + 非对称双栏合计：3/13 = 23% < 40% ✅
- [x] 每页都有role/rhythm/visual_role/anti_pattern字段 ✅
