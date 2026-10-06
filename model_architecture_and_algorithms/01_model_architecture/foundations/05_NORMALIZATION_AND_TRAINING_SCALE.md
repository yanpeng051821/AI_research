# 归一化与计算尺度

> RMSNorm 调整每个位置的特征向量尺度，再施加可学习的逐特征缩放。
> 它服务于后续计算的数值条件，不负责引入上下文，也不把不同内容变成相同语义。

[系列导航](README.md) · 前篇：[Embedding](04_EMBEDDINGS_AND_REPRESENTATIONS.md) · 后篇：[SwiGLU](06_FEEDFORWARD_AND_SWIGLU.md)

## 需求

隐藏表示经过投影、激活与残差相加，数值尺度会不断变化。
下游模块不仅关心方向，也会受到幅度影响：线性输出的大小随输入变化，
激活可能进入不同响应区域，attention 的匹配分数也可能变得更尖锐。

这不是说“大数一定坏、小数一定好”，也不是所有网络都必须用同一种归一化。
这里的目的，是在本地 MiniMind 的 pre-norm 结构中，让计算分支读取经过尺度调整的输入，
减少它对残差主路径整体幅度变化的直接敏感性。

先校正名称：我们这里讨论的是**归一化 normalization**，不是**正则化 regularization**。
前者是前向中的数值变换；后者通常指约束或偏好某类解、降低过拟合的办法，例如权重衰减或 dropout。
归一化也可能影响泛化，但不能因此把两个术语混用。

## 机制

### 均方根

对一个位置的 D 维向量 x，定义：

\[
r(x)=\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}
\]

先平方使正负不相互抵消，再平均得到整体平方幅度，开方使量纲回到原数值尺度。
RMSNorm 输出：

\[
y_i=\gamma_i\frac{x_i}{r(x)}
\]

γ 是长度 D 的可训练缩放向量，通常初始化为全1；ε 是很小的正数，避免零分母并影响小尺度区域。
本篇与当前练习使用 ε=1e-5。官方配置或框架默认可能不同，做对照时必须显式统一。
[RMSNorm 原论文](https://arxiv.org/abs/1910.07467)与[PyTorch 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html)

### 数值过程

取两行：

\[
x^{(1)}=[3,4],\qquad x^{(2)}=[30,40]
\]

忽略微小 ε，均方分别12.5、1250，均方根分别约3.5355、35.3553。
除完得到的两行都约为 [0.8485,1.1314]。

原来两行差10倍，归一化后接近，不是因为它“让任意两行相同”，
而是两行原本方向相同，只差正的整体倍数。
若一行是 [3,4]，另一行是 [4,3]，归一化仍保留其相对分量差异。

γ=[2,0.5] 时，输出约为 [1.6971,0.5657]。
所以最终输出不必保持 RMS=1；归一化和可学习缩放是两个相接步骤。
前者调整当前向量的整体尺度，后者允许模型学习各特征的相对缩放。

### 归一化轴

输入 [B,T,D] 时，分母按每个位置的最后一维计算：

```text
[B,T,D] -> 平方 -> 最后一维平均并保留维度 [B,T,1]
        -> 加 eps、开方
        -> 广播相除 [B,T,D]
        -> 乘 gamma [D]
```

每个位置有自己的分母，同一个 γ 则被所有位置共享。
没有混合 batch，也没有用不同 token 的平均值替代当前 token 的统计量。
这一点既解释了 dim=-1 和 keepdim=True，也说明归一化不是跨位置的信息交流。

## 保留与改变

忽略 ε 且 c>0 时：

\[
\frac{cx}{\operatorname{RMS}(cx)}
=\frac{x}{\operatorname{RMS}(x)}
\]

这体现对正整体缩放的不敏感性。ε 非零时只是近似成立，小数值区域影响尤其明显。
c<0 时符号会翻转，不能直接套用上式。

在乘 γ 前，x 的各分量除以同一个正数，方向保持，绝对尺度发生改变。
再乘逐维 γ 时，方向也可以改变。
因此“只改变长度不改变方向”最多描述归一化的第一步，不能无条件描述整个带权重的模块。

RMSNorm 不减均值。LayerNorm 则先计算均值 μ，再用中心化后的方差：

\[
\operatorname{LayerNorm}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\frac1D\sum_j(x_j-\mu)^2+\epsilon}}+\beta_i
\]

例如全1向量经无偏置 LayerNorm 会变为零，而 RMSNorm 大致仍为全1。
二者不是同一个公式的不同名称，也不能说 RMSNorm 一定“使均值0、方差1”。
[LayerNorm 定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)

## 学习

γ 使模块有可训练参数，但即使 γ 暂时固定，输入仍需要参与反向传播。
分母依赖全部输入分量，所以一个分量变化会影响同一向量中多个输出。
令 \(s=\frac1D\sum_jx_j^2+\epsilon\)，\(r=\sqrt{s}\)。先分两步求导：

\[
\frac{\partial s}{\partial x_j}=\frac{2x_j}{D},
\qquad
\frac{\partial r}{\partial x_j}
=\frac{1}{2\sqrt{s}}\frac{2x_j}{D}=\frac{x_j}{Dr}
\]

再对 \(y_i=\gamma_i x_i/r\) 求导：x_i 在 i=j 时直接变化，1/r 则通过上面的分母变化。
合在一起得到：

\[
\frac{\partial y_i}{\partial x_j}
=\gamma_i\left(\frac{\delta_{ij}}r-\frac{x_ix_j}{D r^3}\right)
\]

δ 在 i=j 时为1，否则为0。第一项是分子变化，第二项来自分母随 x 变化。
这说明 RMSNorm 不是简单“乘一个固定常数”，也解释了为什么只检查输出、不检查输入梯度会漏掉 detach 分母等错误。

在全零输入处，输出虽然为零，输入的局部导数仍可为 \(\gamma_i/\sqrt{\epsilon}\)。
所以“输出为零”也不等于“这里没有学习信号”，归一化更不意味着梯度幅度总是很小。

这里同一位置内的特征相互影响，但不同 token 的统计仍独立。
“逐位置”不等于“逐元素”：ReLU 单元素响应，RMSNorm 则使用一个位置的所有特征计算分母。

## 精度

平方可能扩大数值范围，均值和开方也会受精度影响。
当前手写实现将内部统计转成 float32，再将结果转回输入 dtype，是为了提高低精度计算中的稳健性。
这不代表任意大小的输入都不会溢出，也不代表所有框架实现采用完全相同的中间精度。

ε 的作用不只是“全零测试时不报错”。它还改变接近零时的缩放和导数，因此它属于计算合同。
两个模型若 ε 不同，输出略有差异不一定是实现错误。

当前练习总是转 float32，所以即使传入 float64，内部也不保持双精度。
不能直接用这份实现做要求双精度精确性的梯度数值检查，然后把误差都解释成公式错误。

## 检验

下面把数值、权重、输入梯度与参数梯度放在同一对照中。
两条路径使用独立叶子对象，避免梯度相互污染：

```python
import torch
from torch import nn

x = torch.tensor([[[3., 4.], [30., 40.]]], requires_grad=True)
gamma = torch.tensor([2., 0.5], requires_grad=True)
reference_x = x.detach().clone().requires_grad_(True)
reference = nn.RMSNorm(2, eps=1e-5)
with torch.no_grad():
    reference.weight.copy_(gamma)
manual = gamma * x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
expected = reference(reference_x)
torch.testing.assert_close(manual, expected)
torch.testing.assert_close(
    manual.detach()[0, 0],
    torch.tensor([1.6971, 0.5657]),
    atol=1e-4,
    rtol=1e-4,
)
manual.sum().backward()
expected.sum().backward()
torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-6, rtol=1e-4)
torch.testing.assert_close(gamma.grad, reference.weight.grad)
assert torch.isfinite(reference(torch.zeros(1, 1, 2))).all()
```

使用非全1权重是为了暴露漏乘 γ；使用三维输入是为了暴露统计轴或广播错误；
比较输入梯度验证向上游的学习通路，比较 γ 梯度验证本层参数的学习通路。
全零测试保护 ε 行为；低精度测试另行检查输出 dtype 和有限值。
只有 shape 正确或结果有限，还不足以证明公式正确。

这些是现有[实践测试](../../../training_engineering/case_studies/minimind/implementation/tests/test_rmsnorm.py)背后的理由。
单组数据与框架一致不代表覆盖所有极端数值条件。

## 结构位置

MiniMind 的 FFN 子层采用：

\[
y=x+\operatorname{FFN}(\operatorname{RMSNorm}(x))
\]

归一化的是分支输入，直接相加的主路径仍然是原来的 x。
这样分支能读取尺度受控的表示，同时残差路径仍携带原表示。
这不等于“归一化会在整个模型中删除所有幅度信息”，也不保证残差流永远不会增长。

归一化、激活、残差的合作将在[第 07 篇](07_RESIDUAL_CONNECTIONS_AND_TRANSFORMER_BLOCKS.md)完整连接。
先进入[前馈网络](06_FEEDFORWARD_AND_SWIGLU.md)，看分支拿到这个输入后具体加工什么。

## 复习

**归一化后两行接近，说明什么？** 对 [3,4] 与 [30,40]，说明正整体尺度差被消除得近似一致，
不是说明内容差异被统一。

**为什么还需要可训练 γ？** 统一当前向量整体尺度后，模型仍可学习各特征应如何缩放。

**它与激活一样都是非线性，为什么不能视作同一职责？** RMSNorm 以输入统计量调节尺度，
激活在 FFN 内形成逐元素响应；它们数学上都可非线性，但处理对象和组织目的不同。
