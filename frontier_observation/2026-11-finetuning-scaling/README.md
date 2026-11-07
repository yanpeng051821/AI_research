# When Scaling Meets LLM Finetuning

首次记录：2026-11-05。更新：2026-11-06。第二阶段第一篇，标签：记录（本轮主线阅读完成）。

## 为什么读

承接 AnaDroid 的模型、数据量和方法选型缺口，学习控制变量、比较收益和判断证据边界。
本篇研究翻译与摘要，不是测试领域模型配方，不直接套用其效果结论。

## 文档入口

[案例总览](01_CASE_OVERVIEW.md)按我们的研究链路整理：

1. 研究目标：模型、数据和方法之间的关系。
2. 评测标准：指标、隔离和未微调 baseline。
3. 模型起点：作者的预训练基模与我们使用开源模型的差异。
4. 数据设计：样本规模、抽样与累计训练量。
5. 训练配方：监督、参数更新、训练上限与 checkpoint 选择。
6. 对照实验与结论：图 1 至 6、图 11、12 及对我们的启发。

讨论中的判断与待验证问题保留在对应环节，供后续案例和实验承接，不另堆问答记录。

## 当前状态与下一步

- 本轮主线带读结束，不代表逐页读完所有附录或完成复现。
- 主要收获是实验设计方法；尚未得到适用于我们测试业务的模型、数据量和训练方式答案。
- 下一步先单独讨论 LoRA / QLoRA，再读第二阶段清单中的后续案例。
- 未启动训练，未形成完整团队实验计划。

## 原始资料

论文：*When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method*，ICLR 2024；Biao Zhang、Zhongtao Liu、Colin Cherry、Orhan Firat。

- [论文 PDF（含附录）](sources/When-Scaling-Meets-LLM-Finetuning-2402.17193v1.pdf)。
- [官方会议海报](sources/ICLR2024-poster-19432.png)。
- [来源与版本记录](sources/README.md)：官方在线入口、文件 hash 及开放材料边界。

本实验完整训练代码、模型与原始实验结果的开放范围尚未确认，不把资料下载当作复现完成。

## 返回入口

- [第二阶段阅读清单](../NEXT_CASES_2026-11.md)。
- [上一案例：AnaDroid](../2026-11-anadroid/README.md)。
