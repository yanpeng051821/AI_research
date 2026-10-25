# 优化：从梯度到参数变化

本目录解释怎样组织并可靠执行参数优化，包括梯度、优化器、学习率与数值稳定性。先连接计算图与梯度，再研究更新规则及其计算条件。

先读[从梯度到参数更新：SGD 与 AdamW](01_SGD_ADAMW_AND_PARAMETER_UPDATES.md)。它从一个标量实验出发，依次说明普通 SGD、历史梯度统计、偏差修正、更新尺度和权重衰减，让每组公式都有明确输入与输出。

[训练数值精度与混合精度](02_NUMERICAL_PRECISION_AND_MIXED_PRECISION.md) 接着解释同一次更新中的参数存储、前向运算、敏感统计、梯度与优化器状态如何采用不同精度，并串起 RMSNorm、RoPE、Softmax 和 autocast。这里讨论的是训练的数值计算条件，不把它归为某个独立模型组件。

阅读前可回顾[自动求导与参数更新](../01_model_architecture/foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)，其中解释梯度的范围、链式法则、叶子张量、梯度保存与计算图关系。

通用原理在这里沉淀；具体脚本、报错、运行数值和待完成事项在[MiniMind 第五周实践](../../training_engineering/case_studies/minimind/practice_notes/week_05_training_system.md)中维护。阅读正文不等于已完成对应实践。

后续学到学习率调度、梯度累积与裁剪时，按实际内容扩展，不提前创建空白文章或增加另一份学习路线。
