# 模型组件

> 组件文章解释局部机制，并始终交代它在整个网络中的位置。

[模型架构总导航](../README.md) · [完整模型](../models/README.md)

从输入开始，Embedding 提供初始表示；归一化调整分支收到的尺度；
Attention 汇入其他位置的信息；FFN 加工当前位置已有的特征；残差把分支结果加回已有表示。
这些职责通过连接方式共同起作用，不能仅靠记住各自公式来理解整个网络。

| 组件 | 正文 | 主要问题 |
| --- | --- | --- |
| Embedding | [表示学习](embeddings/01_EMBEDDINGS_AND_REPRESENTATIONS.md) | 查表如何参与训练，初始表示与上下文表示怎样区分？ |
| 归一化 | [计算尺度](normalization/01_NORMALIZATION_AND_TRAINING_SCALE.md) | RMSNorm 改变什么、保留什么？ |
| FFN | [SwiGLU](feedforward/01_FEEDFORWARD_AND_SWIGLU.md) | 升维、激活、门控和降维怎样协作？ |
| 残差 | [表示更新](residual/01_RESIDUAL_CONNECTIONS.md) | 为什么分支输出要加回输入，梯度路径怎样变化？ |
| Attention | [专题入口](attention/README.md) | 内容如何匹配与汇总，多头和位置机制怎样组织计算？ |

组件正文承担主要推导，具体模型文章引用它们并说明选型与连接。
归一化、激活与残差的合作见残差篇；Attention 与 FFN 的合作见 Decoder 整体结构。
