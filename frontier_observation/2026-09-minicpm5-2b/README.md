# MiniCPM5-2B 案例

## 当前状态

| 项目 | 当前值 |
| --- | --- |
| 首次记录 | 2026-W37，2026-09-12 |
| 观察类型 | 开放小模型完整训练链路 |
| 当前标签 | `深读` |
| 是否启动训练 | 否 |
| 是否改变当前主线 | 否，当前仍先完成 Qwen3-0.6B 正式 S1 |

MiniCPM5-2B 作为小模型训练的 Gold Case 使用。当前重点不是比较它与 Qwen3-0.6B
谁的绝对分数更高，而是理解 Base、Midtrain、SFT 和最终 RL + OPD checkpoint 之间的
阶段关系、数据职责、训练目标和评测证据。

## 文档索引

1. [`01_CASE_OVERVIEW.md`](01_CASE_OVERVIEW.md)：模型定位、阶段树、证据边界和研究问题。
2. [`02_OFFICIAL_SOURCE_SCOUT.md`](02_OFFICIAL_SOURCE_SCOUT.md)：官方来源清单、配置核验、矛盾点和待补证据。
3. [`03_BASE_TRAINING_STAGE.md`](03_BASE_TRAINING_STAGE.md)：Base Training 的阶段目标、完整数据流、
   与 SFT 的区别、工程模块和公开证据边界。

## 下一步

Base Training 的阶段地图已经完成。接下来沿训练工程逐项回答：

1. 预训练原始文本如何变成可消费的 token sequence；
2. 文档切分、packing 和边界 mask 如何实现；
3. 4K 到 32K 后数据管道、batch 和显存为什么必须调整；
4. sampler、训练稳定性、checkpoint/resume 和评测如何组成完整闭环；
5. 哪些是官方明确披露，哪些只能作为待验证的工程假设。
