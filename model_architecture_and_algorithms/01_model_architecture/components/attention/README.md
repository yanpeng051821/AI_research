# Attention：从信息读取到上下文表示

> 先弄清一个位置怎样汇总其他位置的信息，再理解多头怎样并行组织这件事，最后加入位置关系。
> 更新：2026-10-06。第一篇已写成正文；后两篇是明确的写作安排，尚未创建空白文章。

[模型架构总导航](../../README.md) · [Decoder 整体结构](../../models/decoder_only_transformer/01_DECODER_ONLY_TRANSFORMER.md)

## 主线

此前的 FFN 能加工每个位置的特征，但不能直接从另一个位置读取表示。
Attention 补上这条信息路径：根据当前输入计算匹配权重，再汇总允许读取位置的内容。
本专题沿着同一个计算过程展开，逐步解释为什么需要这些步骤。

| 顺序 | 文章 | 内容与完成边界 |
| --- | --- | --- |
| 01 | [注意力机制](01_ATTENTION_MECHANISM.md) | 已完成：动机、Q/K/V、单头计算、缩放、mask、梯度、验证与完整模块出口 |
| 02 | 多头与 GQA | 待写：多组投影、拆头、换轴、拼接、o_proj、MHA/MQA/GQA 与资源代价 |
| 03 | 位置与 RoPE | 待写：置换等变、位置与可见性的区别、二维旋转、相对位置、多频率与实际实现 |

当前在第一篇先解释完整模块的出口，避免把 AV 的宽度误认为模型输出宽度；
第二篇再以多头完整例子展开全部张量轴变化。第三篇解释顺序如何进入匹配计算。
后续 KV cache 和 FlashAttention 的工程细节按实践需要独立展开，本专题不把效率技术混成同一种架构变化。

## 读法

先看第一篇的整体图，再顺着三个 value 的例子走到公式。每节先说明上一节还缺什么，再补上一个机制。
正文中的观察代码可以独立运行，作用是核对局部数学关系；不代替学习者亲手实现 Attention 模块。

2026-10-06 已在 Windows、Python 3.12.13、PyTorch 2.14.0+cu130 环境的 CPU 上运行第一篇全部 5 个 Python 示例。
覆盖手算与矩阵结果、未来位置扰动、投影梯度、同宽度输出投影，以及 SDPA 前向和 Q/K/V 梯度对照。
这些是文档示例的验证，不作为学习者实践模块的完成记录。

建议熟悉[线性层](../../foundations/02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md)与
[自动求导](../../foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)，但不要求先学完所有优化算法。
[第四周实践记录](../../../../training_engineering/case_studies/minimind/practice_notes/week_04_attention_and_decoder.md)
保存实际完成的代码、结果和疑问。

## 依据

历史动机参考 [Bahdanau 等人的机器翻译注意力论文](https://arxiv.org/abs/1409.0473)；
标准缩放点积与多头定义参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；
框架行为以项目 PyTorch 版本的接口和实际数值检查为准。
用户提供的 AI 对话用于识别学习疑问，正文不把其中未经核实的例子或结论作为证据。

具体 MiniMind 对照使用[固定提交 3f1a7cc](https://github.com/jingyaogong/minimind/blob/3f1a7cc25b19a861cd1bd6ed313be526b9ecdaf8/model/model_minimind.py)。
该版本是本次 Windows 源码观察对象，不替换此前实践路线记录的参考提交。
