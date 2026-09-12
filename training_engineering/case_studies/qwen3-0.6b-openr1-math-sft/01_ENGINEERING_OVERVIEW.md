# 01 真实 SFT 工程总览

## 1. 这次工程到底在做什么

模型是 `Qwen/Qwen3-0.6B-Base`，数据来自固定 revision 的
`open-r1/OpenR1-Math-220k`。任务是进行一次数学领域、assistant-completion-only
的 SFT，并严格比较训练前 B0 与训练后 S1。

目标不是证明“训练 loss 能下降”。真正目标是同时回答：

1. 数据是否可信、可重建、可追溯？
2. completion mask、token loss 和 gradient accumulation 是否正确？
3. 训练是否能在真实 GPU 上稳定执行和恢复？
4. S1 是否更贴合目标 SFT 数据？
5. 数学任务是否改善，原有通用能力是否回退？
6. 所有结论是否来自相同评测合同，而不是配置漂移？

## 2. 完整流程链路

```text
问题定义与假设
-> 实验合同和 GO/NO-GO 门禁
-> 原始数据获取与 revision 固定
-> 自动数据审计
-> 人工抽样复核
-> 冻结 train/validation/rejected artifact 与 hash
-> 本机核心计算、单元测试和真实 tokenizer 集成
-> TRL/Transformers 数值与行为对照
-> B0 validation NLL、数学能力与通用回归基线
-> GPU 环境 preflight
-> longest-sample 显存探针
-> first-batch shadow
-> 1/5/20-step smoke
-> 50 -> 100-step checkpoint/resume pilot
-> 正式 S1 数据、顺序、源码和依赖 dry-run 冻结
-> GPU telemetry 与评测入口最终预飞行
-> GO/NO-GO 决策
-> 从 B0 启动完整 479-step S1
-> B0/S1 同合同配对评测
-> 逐样本错误分析
-> 下一轮实验假设
```

这条链路不是“SFT 专属脚本列表”，而是一个可靠实验框架。后续 mid-training、DPO、
GRPO 或 Agent RL 可以复用外层骨架，但必须替换数据合同、训练目标、在线轨迹、reward
和评测语义。

## 3. 每个阶段解决的风险

| 阶段 | 主要问题 | 不做的后果 |
| --- | --- | --- |
| 目标与合同 | 到底要改善什么、如何判断成功 | loss 下降后仍不知道实验是否成功 |
| 数据审计 | 数据格式、内容、重复、泄漏、长度是否可靠 | 模型可能学到脏数据或评测答案 |
| artifact 冻结 | 每次运行是否使用同一输入 | B0/S1 不能比较，也无法复现 |
| 核心计算验证 | shift、mask、归约、累积是否正确 | 训练可运行但目标函数错误 |
| B0 | 训练前能力是什么 | 没有因果对照，只剩训练后孤立分数 |
| GPU preflight | 环境、显存、最长样本是否能跑 | 昂贵正式训练在启动后才失败 |
| smoke | 一次更新、保存、生成能否闭环 | 小错误被拖到长任务后才暴露 |
| pilot | 吞吐、恢复、趋势和成本是否合理 | 无法判断正式预算和恢复可靠性 |
| formal S1 | 在冻结合同下完整训练 | pilot 被误当成正式效果实验 |
| paired eval | 变化是否来自模型而非评测条件 | 汇总分数变化没有可信解释 |
| error analysis | 为什么变化、下一步改什么 | 只能盲目换数据或调参 |

## 4. 数据流与控制流

训练的数据流是：

```text
OpenR1-Math 原始记录
-> 选择通过 verifier 的候选回答
-> Qwen chat template
-> tokenized input_ids
-> assistant-only labels，prompt/padding 为 -100
-> IndexedTokenizedSFTDataset
-> 固定 seed 的 sample order
-> DataLoader / dynamic padding collator
-> input_ids / attention_mask / labels
-> Qwen causal LM forward
-> logits [B, T, V]
-> causal shift + completion-only cross entropy
-> accumulation window 内按有效 token 全局平均
-> backward
-> gradient clipping
-> AdamW step
-> scheduler step
-> zero_grad
-> checkpoint / metrics / memory telemetry
```

控制流是：

```text
config + data hash + source hash + sample-order hash
-> run identity
-> fresh run 或合法 resume 校验
-> 每个 optimizer step 记录状态
-> 定期 checkpoint
-> 成功、暂停或失败都写终态证据
```

数据流决定“模型学了什么”；控制流决定“我们能否证明它确实这样学了”。两者缺一不可。

## 5. 这次哪些是自己的，哪些来自框架

我们自己实现和控制：

- 数据审计、人工复核、长度策略和 artifact 冻结；
- assistant-only labels、核心 masked SFT loss 和 token 级累积测试；
- 确定性样本顺序、run identity、hash 与 resume 合同；
- first-batch shadow、NLL 配对、评测比较和错误输出；
- CUDA allocator telemetry、失败快照和执行门禁。

成熟框架负责：

- Transformers 加载模型和 tokenizer；
- TRL `SFTTrainer` / `SFTConfig` 执行正式训练主循环；
- PyTorch autograd、AdamW、CUDA 和张量计算；
- LightEval/vLLM 执行冻结评测。

OpenR1 提供数据来源和成熟后训练工程的参考，但正式 S1 不是直接调用 OpenR1 的训练
脚本。我们的入口是实验仓库中的 `scripts/train_sft_trl.py`。

## 6. 当前已经拿到的证据

| 证据 | 结果 | 能说明什么 |
| --- | --- | --- |
| 冻结训练集 | 61,224 条，<= 16,384 token | 正式 S1 输入已固定 |
| 冻结验证集 | 1,968 条，11,367,594 个有效 token | B0/pilot NLL 可直接配对 |
| B0 validation NLL | 0.7804831795 | 训练前目标分布拟合基线 |
| B0 MATH-500 | pass@1:1 0.432；pass@1:4 0.449 | 数学主评测基线 |
| B0 GSM8K | QEM 0.4761182714 | 补充数学基线 |
| B0 regression | aggregate acc 0.5407047139 | 通用能力回归基线 |
| 20-step smoke | NLL 从同子集 0.8050 降至 0.6754 | 训练路径能学，非正式能力结论 |
| 100-step pilot | 全量 validation NLL 0.5952246359 | 相对 B0 下降约 23.74% |
| CPU S1 dry-run | 61,224 records、479 steps、15 warmup | 正式数据与训练计划已冻结 |

100-step pilot 的 NLL 是可信训练行为证据，但 pilot 的 MATH-500 没有形成可靠终态；
GSM8K 50 样本和 regression smoke 也不能和完整 B0 直接比较。

## 7. 当前状态与下一步

当前正式 S1 仍为 `NO-GO`，不是因为训练代码缺失，而是还剩两类实机门禁：

1. 用最终 S1 配置完成 1 至 2 个 optimizer update 的 CUDA allocator telemetry，
   确认最长保留样本、首次 AdamW `m/v` 分配和 checkpoint 均安全。
2. 让 B0 与 pilot/final-like checkpoint 通过同一个 MATH-500 小规模评测入口，
   确认 vLLM context、KV cache 和失败终态记录可靠。

两个门禁关闭后，才从 Base 新建正式 S1；不能从 smoke 或 pilot checkpoint 接着训练。

## 8. 如何把本案例复用到下一次训练

保持不变：

```text
目标 -> 合同 -> 数据 -> B0 -> preflight -> smoke -> pilot
-> formal run -> paired eval -> error analysis
```

必须重新定义：

```text
模型 revision
数据来源与许可
训练目标和 mask
长度与混合策略
预算和资源
主指标与回归指标
成功/停止门槛
算法特有状态
```

不要复制这次的 `16K`、`gradient_accumulation=128` 或 `learning_rate=4e-5` 作为
普遍答案。可复用的是做决策和保留证据的方法，不是具体超参数。
