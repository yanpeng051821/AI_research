# 残差与 Transformer 整体结构

> 归一化调整分支输入的尺度，激活参与分支内部的非线性加工，残差把加工结果加回已有表示。
> Attention 与 FFN 分别侧重跨位置交流和逐位置加工，多个 block 反复交替，最终形成可用于预测的表示。

[系列导航](README.md) · 前篇：[SwiGLU](06_FEEDFORWARD_AND_SWIGLU.md) · 后篇：[自动求导](08_AUTOGRAD_AND_PARAMETER_UPDATES.md)

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
本地官方结构见[MiniMindBlock](../../../../minimind/model/model_minimind.py)。

当前实践尚未完成 attention 与整模型实现，本篇先给出定位，避免局部计算失去上下文。

## 残差表示

### 学习增量

普通串联写作 \(y=F(x)\)，残差写作：

\[
y=x+F(x)
\]

两者都能表达复杂变换，但残差参数化让分支输出成为加在已有表示上的增量。
如果当前层不需要明显改变输入，只需要 F(x) 接近0，就能使 y 接近x。
这是一种计算组织方式，不是训练标签真的提供了一个“标准残差答案”。

“保留原表示”也不是绝对保证信息不会损失。
相加会把两项混合，若 F(x)=-x，输出就是0；网络仍需要学出有用的增量。
残差的确提供一条直接输入路径，但不能把这句话扩大为任意信息都可完整恢复。

### 形状约束

x 与 F(x) 必须能按预期逐位置、逐特征相加。
在当前 block 中二者都是 [B,T,D]。FFN 中间虽然升到 H，最终仍要投回 D。
这解释了 down projection 的结构职责。

其他网络可以使用投影捷径处理维度变化，但当前 MiniMind 子层是同宽度的恒等捷径。
不要因为广播机制允许某些 shape 相加，就把它们当成正确残差；
例如 [B,T,1] 被广播到 D 维可能运行成功，语义却不是预期的 D 维增量。

## 梯度路径

### 标量观察

先看已经做过的小实验。若 F(x)=2x：

\[
y_{\mathrm{plain}}=2x,\qquad y_{\mathrm{residual}}=x+2x=3x
\]

对输出求和时，输入梯度分别为2和3：

```python
import torch

plain_x = torch.tensor([-1., 2.], requires_grad=True)
residual_x = plain_x.detach().clone().requires_grad_(True)
plain = 2 * plain_x
residual = residual_x + 2 * residual_x
plain.sum().backward()
residual.sum().backward()
torch.testing.assert_close(plain, torch.tensor([-2., 4.]))
torch.testing.assert_close(residual, torch.tensor([-3., 6.]))
torch.testing.assert_close(plain_x.grad, torch.tensor([2., 2.]))
torch.testing.assert_close(residual_x.grad, torch.tensor([3., 3.]))
```

这里增加的1来自直接路径，不是 autograd 特意给残差“额外奖励”。
这个例子展示一层的导数结构，不证明任意深层模型都不会梯度消失或爆炸。

### 向量与深层

向量情形中，若 J_F 是分支输出对输入的 Jacobian：

\[
J_y=I+J_F,\qquad
\frac{\partial L}{\partial x}=g+J_F^Tg
\]

g 是从下游传回的梯度。反向有直接项 g，也有经过分支的项。
多层残差相乘的结构中包含直接路径，有助于信息和学习信号传播；
但各项仍可能相互抵消或放大，所以不是无条件保证。
[恒等映射残差网络论文](https://arxiv.org/abs/1603.05027)

## 三者协作

以已经实现组件所组成的前馈子层为中心：

\[
y=x+\operatorname{SwiGLU}(\operatorname{RMSNorm}(x))
\]

一次计算可以分成四步：

1. 主路径保留 x，尚未改变。
2. RMSNorm 根据当前向量的均方根调整尺度，乘可训练 γ，交给分支。
3. SwiGLU 通过投影、SiLU、逐元素相乘和 down 投影产生 D 维加工结果。
4. 将这一结果加到主路径 x，得到后续子层使用的新表示。

三者不是三个并列的“让训练更好”的技巧，而是处在不同计算位置：

| 组件 | 作用对象 | 在这条链中的职责 |
| --- | --- | --- |
| RMSNorm | 分支收到的整个特征向量 | 调整输入尺度与逐特征缩放 |
| SiLU | gate 投影后的各个数值 | 产生输入相关的非线性响应 |
| 残差相加 | 原表示与分支输出 | 让分支以增量形式更新表示，并提供直接梯度路径 |

激活并不直接接在残差相加后；它位于 FFN 内部。
归一化也没有替换主路径 x；它只改变分支读到的版本。
顺序和位置是结构的一部分，不能只记成“归一化 + 激活 + 残差”。

pre-norm 指归一化在子层计算前；post-norm 则把归一化放在残差相加后。
原始 Transformer 论文与本地 MiniMind 的放置方式不同，不宜用一张图代表所有 Transformer。
本篇以 MiniMind 的 pre-norm 为当前理解对象，不在这里展开两者的训练比较。

此外，归一化不是正则化的别名。权重衰减、dropout、RMSNorm 各自作用于不同环节，
不能把它们统称为“正则层”后丢失其具体机制。

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
具体旋转、头维度与缓存实现属于后续 attention 实践，本篇不把尚未实现的细节写成验收结果。

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

## 检验

残差观察之外，装配时应检查分支与主路径真的按预期连接：
若将一个前馈分支的输出固定为0，整个残差子层应输出原 x。
这验证的是“原路径没有丢失”，不要求把整个模型所有参数清零。

下面用普通线性分支演示这个验证，不替代待完成的 ResidualFeedForward 实现：

```python
import torch
from torch import nn

torch.manual_seed(0)
branch = nn.Linear(3, 3, bias=False)
with torch.no_grad():
    branch.weight.zero_()
x = torch.randn(2, 4, 3, requires_grad=True)
y = x + branch(x)
torch.testing.assert_close(y, x)
y.sum().backward()
torch.testing.assert_close(x.grad, torch.ones_like(x))
```

shape 检查保护相加的结构合同；零分支验证保护直接路径；梯度验证保护反向依赖。
之后完整 attention 还需要独立检查因果性，不能因为残差块测试通过就宣称完整 Transformer 正确。

小组件的数值正确、block 的连接正确、完整模型能拟合小批数据、真实任务表现改善，是递进的不同证据。
没有任何一个测试能独自代替后面的全部层次。

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

下一篇沿反方向走一遍：这些前向依赖如何变成[梯度与参数更新](08_AUTOGRAD_AND_PARAMETER_UPDATES.md)。
