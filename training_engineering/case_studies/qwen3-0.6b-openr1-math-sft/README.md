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

当前状态：正式 S1 为 `NO-GO`。CPU 数据与合同预检已完成；GPU allocator telemetry
预飞行和 MATH-500 评测入口预飞行尚未关闭。

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

## 每篇文档的使用方式

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
