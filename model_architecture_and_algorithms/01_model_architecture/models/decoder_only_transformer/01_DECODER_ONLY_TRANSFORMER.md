# Decoder-only Transformer：把组件连成完整模型

> token ID 经过 Embedding 得到初始表示；多个 Decoder block 交替进行上下文交流和逐位置加工；最终表示经过 LM head 变成下一 token 的词表分数。

[系列导航](../../README.md)

## 整体定位

此前按学习顺序接触 Embedding、RMSNorm、SwiGLU、残差，不等于实际模型依次把它们各调用一次就结束。
MiniMind dense decoder 的主要结构是：

```text
token ID [B,T]
  |
Embedding
  |
X0 [B,T,D]
  |
重复多个 decoder block：
  X -> RMSNorm_1 -> Causal Attention -> 增量 A
  H = X + A
  H -> RMSNorm_2 -> FFN              -> 增量 F
  X_next = H + F
  |
最终 RMSNorm
  |
LM head [B,T,V]
  |
词表 logits -> 训练时计算 loss / 生成时选择下一个 token
```

用两条公式写一个 pre-norm block：

\[
H=X+\operatorname{Attention}(\operatorname{Norm}_1(X))
\]
\[
Y=H+\operatorname{FFN}(\operatorname{Norm}_2(H))
\]

注意两个 Norm 通常拥有各自的参数，不是同一对象反复使用。
一个 block 也不等于一个 Linear：它包含两种子层、归一化与残差连接。
具体结构见[固定版本的 MiniMindBlock](https://github.com/jingyaogong/minimind/blob/3f1a7cc25b19a861cd1bd6ed313be526b9ecdaf8/model/model_minimind.py)。

当前实践尚未完成 attention 与整模型实现，本篇先给出定位，避免局部计算失去上下文。

## 分支协作

每个 block 的两条分支都接收当前表示，在内部完成加工，再输出相同宽度的更新量。RMSNorm 调整分支读到的尺度；Attention 让不同位置交流；FFN 加工每个位置已经获得的信息。两次残差相加分别完成一次表示更新。

这个结构解释了为什么 Attention 的输出投影、FFN 的 down projection 都要回到 hidden_size。模块内部可以改变宽度，block 之间传递的表示维度则保持一致。残差的数值例子、梯度推导和零分支验证见[残差组件](../../components/residual/01_RESIDUAL_CONNECTIONS.md)。

## 上下文交流

FFN 加工每个位置已有的信息，但若始终只做逐位置计算，
“我喜欢”最后一个位置就无法通过这些操作读取前一个位置的表示。
Transformer 使用 attention 在允许的位置之间交流信息。

### 匹配与汇总

对一个头，忽略 batch，输入 [T,D] 经投影形成 Q、K、V。
这里 V 表示 value 矩阵，不是本系列其他场景中的词表大小；必须根据上下文区分符号。

\[
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right),
\qquad O=AV
\]

Q 的每一行代表当前位置的查询，K 为各位置提供匹配依据，V 为各位置提供被汇总的内容。
这些是功能名称，不意味着它们已经是自然语言形式的问题、关键词和答案。

QK^T 得到 [T,T]：行是“谁在读取”，列是“从哪里读取”。
M 将未来位置排除，softmax 沿可读取位置形成权重，
AV 则把不同位置的 value 按权重汇总。除以根号 d_k 用于控制点积分数随维度增长的尺度；
它不是 RMSNorm，二者处理对象不同。
[Attention 的原始定义](https://arxiv.org/html/1706.03762v7#S3.SS2)

例如“喜欢”所在位置可以从“我”读取相关表示，并与自身信息结合。
FFN 随后对这份已经包含上下文的表示做非线性加工。
多层重复后，后层读取的是前层已经加工过的各位置表示，而不是始终读取最初 Embedding。

### 多头与 GQA

多头将投影后的特征组织成多个子空间，使不同头分别进行匹配和汇总，再拼接并投影。
拆分最后一维只是形状准备，核心信息交流发生在 [T,T] 权重与跨位置加权求和中。
因此不能把“多头 attention”仅理解成“像 FFN 一样分组处理最后一维”。

GQA 让多个 query 头共享较少的 key/value 头，是 head 之间的参数与缓存组织方式，
不是简单把 token 序列切成互不交流的几组。
这解释了它与当前特征维操作的联系，也保留了两者的区别。
[GQA 原论文](https://arxiv.org/abs/2305.13245)

本地 MiniMind 还在 Q、K 上应用 RoPE，引入位置关系。
匹配、mask、维度与梯度的逐步推导见[Attention 第一篇](../../components/attention/01_ATTENTION_MECHANISM.md)。
具体旋转、多头组织与缓存实现属于后续实践，本篇不把尚未实现的细节写成验收结果。

## 一条序列

现在把局部组件重新串起来。输入 token 序列是：

```text
BOS, 我, 喜欢, 数学, EOS
```

Embedding 给每个 ID 一个初始向量。“喜欢”的向量尚不负责理解整句。
进入第一层，attention 在因果范围内汇入 BOS、我、喜欢的信息，再把结果加回“喜欢”原来的表示；
FFN 随后加工这个位置的新表示，并再次残差相加。

第二层继续使用更新后的所有位置表示，而不是从原文重新查表开始。
重复多层后，最后的“喜欢”位置隐藏状态已经是依赖前文的特征向量。
最终 Norm 与 LM head 将它转换成词表分数，用来预测“数学”。

训练时，正确目标“数学”的 NLL 会通过 head 传回这份隐藏状态，再经过 FFN、attention 和各条残差路径，
影响参与预测的参数，包括前文的输入表示。
这说明 prompt 位置即使不直接计入 loss，也仍能通过上下文路径参与学习。

同一架构的生成阶段会从最后位置的分布选出新 token，追加输入并继续。
中间计算学习到什么，是数据和目标共同塑造的；不能把某一层固定命名为“语法层”或“推理层”。

## 复习

**归一化、激活、残差究竟是什么关系？**
先调整分支输入尺度，在分支内部形成非线性特征响应，再把分支结果加回原表示。
它们职责不同，连接位置明确，最终共同参与表示的逐层更新。

**为什么 FFN 不跨位置却能处理上下文？**
因为它接收的隐藏状态已经由 attention 汇入上下文；直接依赖与间接依赖需要分开看。

**残差为什么不保证梯度总是正常？**
它提供直接项，但分支项仍可能抵消或放大；训练还受初始化、尺度、深度与优化影响。

**模型理解只是查表吗？**
初始表示用查表实现，但后续结果由输入相关的交流与非线性计算产生，不是直接查找整句答案。

下一篇沿反方向走一遍：这些前向依赖如何变成[梯度与参数更新](../../foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)。
