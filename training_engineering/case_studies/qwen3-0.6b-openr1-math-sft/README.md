# Qwen3-0.6B OpenR1-Math SFT 工程案例

## 案例定位

这是一次真实的数学领域 SFT 工程，不是单纯的算法笔记，也不是为了追求最高分数的
训练 recipe。它的主要目标是建立一条可以审计、恢复、比较和解释的训练链路：

```text
真实模型 + 真实数据 + 真实 GPU
-> 可复现训练
-> 同合同评测
-> 有边界的实验结论
```

实验仓库：`D:/pythonlearning/small_model_post_training/independent_implementation`

当前状态：正式 S1 训练与工程收尾已完成，结果决策为 `ITERATE`。目标 validation NLL
和 GSM8K 有改善，59 项通用回归面板下降；完整 MATH-500 因 S1 生成长度和付费预算异常
被主动中止并延期。因此该模型可以作为可复现实验 checkpoint 发布，不能作为“数学能力已
全面提升”的最终模型发布。

## 阅读顺序

1. [`01_ENGINEERING_OVERVIEW.md`](01_ENGINEERING_OVERVIEW.md)：先建立完整工程地图。
2. [`02_OBJECTIVE_CONTRACT_AND_GATES.md`](02_OBJECTIVE_CONTRACT_AND_GATES.md)：把目标变成可验证合同。
3. [`03_DATA_AUDIT_AND_ARTIFACT_FREEZE.md`](03_DATA_AUDIT_AND_ARTIFACT_FREEZE.md)：从原始数据得到冻结训练 artifact。
4. [`04_IMPLEMENTATION_AND_NUMERICAL_VERIFICATION.md`](04_IMPLEMENTATION_AND_NUMERICAL_VERIFICATION.md)：验证 loss、batch、累积和 TRL 对齐。
5. [`05_BASELINE_AND_EVALUATION_CONTRACT.md`](05_BASELINE_AND_EVALUATION_CONTRACT.md)：建立训练前 B0。
6. [`06_SERVER_PREFLIGHT_AND_MEMORY.md`](06_SERVER_PREFLIGHT_AND_MEMORY.md)：在租用 GPU 后先验证环境与容量。
7. [`07_SMOKE_PILOT_AND_RESUME.md`](07_SMOKE_PILOT_AND_RESUME.md)：用 smoke/pilot 逐级扩大风险。
8. [`08_PAIRED_EVALUATION_AND_ANALYSIS.md`](08_PAIRED_EVALUATION_AND_ANALYSIS.md)：做同合同评测与错误分析。
9. [`09_FORMAL_S1_RUNBOOK.md`](09_FORMAL_S1_RUNBOOK.md)：正式 S1 的启动、观察、恢复和收尾手册。
10. [`10_INCIDENTS_AND_REUSABLE_LESSONS.md`](10_INCIDENTS_AND_REUSABLE_LESSONS.md)：真实波折、诊断过程和复用规则。
11. [`11_FORMAL_S1_RESULTS_AND_RELEASE.md`](11_FORMAL_S1_RESULTS_AND_RELEASE.md)：正式结果、结论边界和发布记录。

## 文档时间视角

这个目录同时保存了事前合同、执行手册、历史阶段记录和事后结论。文档中的未来时或
`NO-GO` 不一定表示案例当前仍未完成，应按下表理解：

| 文档 | 性质 | 当前应如何读取 |
| --- | --- | --- |
| `01` | 最终工程总览 | 当前事实入口，已同步到正式 S1 和后续分析状态 |
| `02` | 事前目标合同 | 保留当时冻结的目标和门禁；检查表已按实际执行回填 |
| `03` | 数据阶段方法与记录 | 本案例已完成，artifact 已冻结并用于正式训练 |
| `04` | 独立实现与对齐记录 | 本地数值验证、TRL 对齐和服务器验证均已完成 |
| `05` | B0 合同与结果 | B0 已完成；这些数值是训练前冻结基线 |
| `06` | 服务器预飞行记录 | 32K 探针失败后改为 16K，最终预飞行已通过 |
| `07` | smoke/pilot 历史快照 | 记录当时的 `NO-GO` 和恢复过程，不代表当前状态 |
| `08` | 配对评测与分析入口 | 已同步正式结果；MATH v2 仍为显式延期项 |
| `09` | 可复用运行手册 | 命令使用未来时是刻意保留；开头说明了实际执行偏离 |
| `10` | 事后事故复盘 | 当前最终记录 |
| `11` | 正式结果与发布 | 当前结果、证据边界和工程终态的唯一权威入口 |
| `analysis/gsm8k_target_capability` | 事后分析链 | 54 条固定分层样本复核已完成，以其中 `README` 和 `07` 为最终入口 |

如果历史阶段记录与最终状态看似冲突，以 `11` 为结果事实，以 `10` 解释过程；不要删除
历史门禁，否则会丢失“当时为什么不能继续”和“后来如何关闭风险”的证据。

## 每篇文档的使用方式

代码学习配套入口：[code-review/README.md](code-review/README.md)。其中六个主题按实际
调用链组织，分别覆盖入口、数据、计算、TRL、恢复显存和评测；包含代码定位、测试与小练习。
这些主题当前是待开展的学习计划，不代表学习者已经通过验收。

每个流程节点都按同一套结构组织：

```text
目的
-> 前置输入
-> 操作步骤
-> 验收门
-> 必须落盘的证据
-> 常见失败及判断
-> 本案例结果
-> 可复用边界
```

第一次阅读时先读 `01`，不要马上复制命令。真正执行某个节点时，再打开对应章节，
逐项核对输入、输出和验收门。
