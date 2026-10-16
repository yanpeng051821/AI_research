# 完整模型

> 从整体输入输出出发，跟踪组件如何共同形成预测，并说明具体架构的选择。

[模型架构总导航](../README.md) · [组件详解](../components/README.md)

当前正文：[Decoder-only Transformer](decoder_only_transformer/01_DECODER_ONLY_TRANSFORMER.md)。
它以 MiniMind 风格的 pre-norm dense decoder 为例，串起 Embedding、Attention、FFN、残差与 LM head。
原始 Transformer 的 encoder-decoder 与 post-norm 结构不直接等同于这个例子。

后续具体模型分析在实际阅读源码时建立对应子目录。每篇应回答输入输出、组件选择、连接顺序、
关键配置、参数分布以及相对于通用结构的区别；组件公式通过链接复用。
训练 recipe、数据与实验产物仍归训练工程案例，避免同一份状态出现在多处。
