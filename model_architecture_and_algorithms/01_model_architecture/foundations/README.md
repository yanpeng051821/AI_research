# 基础原理

> 从预测任务、张量计算到学习过程，为模型组件提供共同的解释基础。

[模型架构总导航](../README.md)给出跨目录的完整阅读顺序。本目录专注于理解各类组件都会用到的概念。

| 正文 | 核心问题 |
| --- | --- |
| [自回归语言建模](01_AUTOREGRESSIVE_LANGUAGE_MODELING.md) | 预测什么？条件概率、loss、模型与训练阶段怎样关联？ |
| [向量、矩阵与线性层](02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md) | 数值怎样表示特征，线性层怎样组合它们？ |
| [线性、非线性与激活](03_LINEARITY_NONLINEARITY_AND_ACTIVATIONS.md) | 为什么线性层需要非线性，激活怎样影响响应和梯度？ |
| [自动求导与参数更新](08_AUTOGRAD_AND_PARAMETER_UPDATES.md) | 参数、激活、计算图和 optimizer 分别承担什么职责？ |

原来的 Embedding、归一化、SwiGLU 与残差详解已归入[组件](../components/README.md)；
完整 Decoder 的数据流归入[模型](../models/README.md)。正文和实践链接随迁移同步更新。

阅读基础并不要求先记住全部公式。遇到组件中的具体疑问时，可以回到这里：
例如不清楚拆头与换轴，先复习矩阵篇的轴与数值；不清楚 Q/K/V 怎样学习，再复习自动求导篇。
