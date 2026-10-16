# 模型架构：从基础原理到完整网络

> 这里沿着“表示如何形成、信息如何交流、模型如何预测与学习”理解架构。
> 更新：2026-10-06。目录按知识归属组织，阅读顺序由本页串联；实践进度仍见 MiniMind 案例路线。

## 地图

一个语言模型收到 token ID 后，先查出可训练表示，再反复交流上下文和加工特征，最后为下一个 token 生成词表分数。
为了理解这条链，我们需要三种尺度：计算的基础规则、单个组件的机制，以及整个模型的组织方式。

| 目录 | 讨论对象 | 阅读时的问题 |
| --- | --- | --- |
| [foundations](foundations/README.md) | 通用数学与学习原理 | 矩阵怎样处理特征？非线性和梯度分别做什么？ |
| [components](components/README.md) | 可复用的模型组件 | 这个组件收到什么、为何这样计算、输出交给谁？ |
| [models](models/README.md) | 完整架构与具体模型 | 这些组件如何连接？配置怎样决定整条计算链？ |

例如，“矩阵乘法沿哪些轴进行”属于基础，“Attention 为什么把头轴移到前面”属于组件，
“MiniMind 使用多少个 Q/KV 头”属于具体模型。后两者引用前者，不需要重复写三套矩阵教程。

## 阅读顺序

已经读过的文章无需从头重读。遇到局部问题，先在整体结构中定位，再回到对应组件。

| 顺序 | 正文 | 这一步建立什么理解 |
| --- | --- | --- |
| 1 | [自回归语言建模](foundations/01_AUTOREGRESSIVE_LANGUAGE_MODELING.md) | 预测任务、条件概率、loss 与训练阶段的关系 |
| 2 | [向量、矩阵与线性层](foundations/02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md) | 特征组合、形状与批量计算 |
| 3 | [线性、非线性与激活](foundations/03_LINEARITY_NONLINEARITY_AND_ACTIVATIONS.md) | 网络怎样超出线性映射的表达范围 |
| 4 | [Embedding](components/embeddings/01_EMBEDDINGS_AND_REPRESENTATIONS.md) | 离散编号怎样获得可训练表示 |
| 5 | [归一化](components/normalization/01_NORMALIZATION_AND_TRAINING_SCALE.md) | 分支输入的尺度如何调整 |
| 6 | [FFN 与 SwiGLU](components/feedforward/01_FEEDFORWARD_AND_SWIGLU.md) | 每个位置的特征怎样被加工 |
| 7 | [残差连接](components/residual/01_RESIDUAL_CONNECTIONS.md) | 分支怎样更新已有表示，梯度怎样回传 |
| 8 | [Decoder 整体结构](models/decoder_only_transformer/01_DECODER_ONLY_TRANSFORMER.md) | 先获得全局图，再定位下一步要深入的 Attention |
| 9 | [自动求导与参数更新](foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md) | 前向依赖如何形成学习信号 |
| 10 | [Attention 专题](components/attention/README.md) | 系统展开跨位置交流，再回到完整 Decoder |

前四篇基础文件保留原编号，避免只为连续编号而改动文件身份。编号不是全局分类规则。
原“残差与整体结构”已拆成两篇；原空白 Decoder 占位已合并到有正文的完整架构文章中。

## 讲解方式

正文先给出本节要解决的问题和整体结论，再带读者经历一个小例子。
公式是对已经理解的过程做精确表达，形状帮助定位每个数从哪里来，代码用于复核推导。
每一篇都连接到前后模块，说明哪些量是参数、哪些量由当前输入计算，以及测试能排除什么错误。

“特征”不预设为一个人类词义，“某个头的作用”也不预设为固定语法标签。
类比只帮助入门，真正的解释要回到数值依赖、训练目标和证据。

## 符号

| 符号 | 含义 |
| --- | --- |
| B、T、D | batch 大小、序列长度、模型隐藏宽度 |
| H 或 D_ff | 早期 FFN 文章的中间宽度；Attention 专题统一使用 D_ff，避免和头数混淆 |
| n_q、n_kv、d_head | Q 头数、KV 头数、每个头的宽度 |
| V_vocab | Attention 专题中的词表大小，避免与 value 矩阵 V 混淆 |
| X、Q、K、V、A | 隐藏状态、query、key、value、当前输入的注意力权重 |

批量公式采用行向量写法 XW；PyTorch Linear 保存的是 [out_features, in_features]，
对应代码 X @ linear.weight.T。具体推导见矩阵篇。

## 实践与版本

[第三周教程](../../training_engineering/case_studies/minimind/practice_notes/week_03_model_components.md)记录已完成的基础组件实践；
[第四周教程](../../training_engineering/case_studies/minimind/practice_notes/week_04_attention_and_decoder.md)从单 query 观察开始持续补充。
读过理论、复核文中示例、亲手实现模块与通过实践验收分别记录。

早期材料在 Mac 上记录的参考提交为 f659b55761b754d306bd140573493a6543cafd7f。
本次 Windows 实际检查的是 [MiniMind 提交 3f1a7cc](https://github.com/jingyaogong/minimind/blob/3f1a7cc25b19a861cd1bd6ed313be526b9ecdaf8/model/model_minimind.py)。
新补充的具体结构以这个固定链接为依据，不把两份代码视作同一版本。理论整理不改变实践的参考版本合同；
后续实现前再核对差异。

模型数据、实验配置、运行日志与结果分析留在训练工程案例中，这里保存可跨实验复用的原理。
