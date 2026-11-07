# 参数适配

本目录回答：在已有模型上训练时，更新哪些参数，怎样表示调整量，以及怎样降低存储与训练开销。

## 1. 阅读顺序

| 顺序 | 文章 | 核心问题 |
| --- | --- | --- |
| 1 | [LoRA：结构、低秩与参数更新](01_LORA_STRUCTURE_AND_UPDATES.md) | 外挂矩阵如何接入线性层，数据形状、梯度和 rank 分别意味着什么？ |
| 2 | [QLoRA：量化存储与训练计算](02_QLORA_QUANTIZATION_AND_TRAINING.md) | 基模量化后怎样参与前向与反向，NF4、双重量化和分页优化器分别解决什么？ |

前置知识：

- [向量与线性层](../01_model_architecture/foundations/02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md)
- [自动求导](../01_model_architecture/foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)
- [数值精度](../03_optimization/02_NUMERICAL_PRECISION_AND_MIXED_PRECISION.md)

## 2. 与研究链路的关系

- 对应“训练配方”中的参数更新范围、参数化方式和精度策略，不替代目标能力、数据设计与评测。
- 两篇根据 2026-11-07 的讨论整理，为独立技术说明，不是已经完成的 LoRA/QLoRA 训练记录。
- 下一篇案例仍从[第二阶段阅读清单](../../frontier_observation/NEXT_CASES_2026-11.md)进入；不会因为读完原理而自动启动实验。

## 3. 文档边界

- 本目录保存通用机制与工程判断依据；具体配置、依赖、设备、结果与恢复记录放在训练工程案例。
- 既有 [MiniMind 第八周笔记](../../training_engineering/case_studies/minimind/theory_notes/week_08_lora.md)是历史实践上下文，本轮不重写；通用定义以本目录为阅读入口。
- 特别区分 SFT 的数据与监督规则、LoRA 的参数更新方式、QLoRA 的量化存储与计算方案。
