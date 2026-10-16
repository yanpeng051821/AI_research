# 残差连接：让分支更新已有表示

> 残差将原表示与分支计算结果相加。理解它，需要同时看前向的信息传递、反向的梯度路径，以及分支如何与归一化和激活配合。

[系列导航](../../README.md)

## 定位

我们已经能用 FFN 加工一个位置的特征。接下来需要决定：这些加工结果怎样交给下一层？残差结构让分支产生一份更新量，再加回已有表示。下面先从一个子层出发，最后通过测试检查这条直接路径。

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

## 衔接

现在可以解释单个残差子层如何工作。接着阅读[Decoder 整体结构](../../models/decoder_only_transformer/01_DECODER_ONLY_TRANSFORMER.md)，看 Attention 和 FFN 如何各自使用残差，逐层更新整条序列。
