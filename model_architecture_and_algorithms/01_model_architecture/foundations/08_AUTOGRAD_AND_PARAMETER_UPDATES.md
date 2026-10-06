# 自动求导与参数更新

> 前向计算建立 loss 对参数的依赖，反向传播按链式法则计算梯度，优化器根据梯度更新参数。
> “参与计算”“保存梯度”“属于可更新参数”是三个不同问题。

[系列导航](README.md) · 前篇：[残差与整体结构](07_RESIDUAL_CONNECTIONS_AND_TRANSFORMER_BLOCKS.md)

## 学习闭环

把语言模型暂时缩小成一个可手算的网络：

\[
\hat y=wx+b,\qquad L=(\hat y-t)^2
\]

x=2，目标 t=5，初始 w=1、b=0。前向得到 ŷ=2、L=9。
目标不是让程序“知道答案是5后直接把输出改成5”，而是利用这次误差调整共享参数，
使之后的输入也经过同一函数产生预测。

对参数求导：

\[
\frac{\partial L}{\partial w}=2(\hat y-t)x=-12,\qquad
\frac{\partial L}{\partial b}=2(\hat y-t)=-6
\]

梯度是当前点附近的局部变化率。正梯度表示单独微增参数会使 loss 一阶近似增大，
负梯度则相反；它不是全局因果重要性排名，也不保证沿反方向走任意大一步都更好。

SGD 取学习率0.1：

\[
w'=w-0.1(-12)=2.2,\qquad b'=b-0.1(-6)=0.6
\]

新输出为5，新 loss 为0。这是这个数值例子的结果，不是学习率0.1普遍能一步收敛。

```python
import torch
from torch import nn

w = nn.Parameter(torch.tensor(1.))
b = nn.Parameter(torch.tensor(0.))
optimizer = torch.optim.SGD([w, b], lr=0.1)
x, target = torch.tensor(2.), torch.tensor(5.)
loss = (w * x + b - target).square()
before = w.detach().clone()
loss.backward()
torch.testing.assert_close(w.grad, torch.tensor(-12.))
torch.testing.assert_close(b.grad, torch.tensor(-6.))
torch.testing.assert_close(w, before)
optimizer.step()
with torch.no_grad():
    torch.testing.assert_close(w, torch.tensor(2.2))
    torch.testing.assert_close(b, torch.tensor(0.6))
    torch.testing.assert_close(w * x + b, target)
```

backward 后参数仍没变，step 后才变化。
真实语言模型规模更大、目标常为交叉熵、优化器可能是 AdamW，但这三个职责的划分仍然存在。
AdamW 还使用历史状态，不能把所有优化器都理解为“当前梯度乘学习率”。

## 链式法则

若计算为 x -> z -> h -> L，局部导数相乘：

\[
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial h}
 \frac{\partial h}{\partial z}
 \frac{\partial z}{\partial x}
\]

这里用标量书写；向量形式是 Jacobian 的相应乘积。
autograd 不需要先创建一个巨大的完整 Jacobian 再相乘，反向通常传播向量-Jacobian 积。
对于标量 loss，起点相当于 \(\partial L/\partial L=1\)。

如果一个值流向多个分支，各条路径的贡献相加。
这解释了残差的直接项、SwiGLU 的双分支、重复 token 的 Embedding 梯度，以及跨位置共享 Linear 权重的梯度累计。
它们不是框架中的四条独立魔法规则，而是同一个多路径链式法则。

梯度必须根据前向时的值计算。框架会为一些运算保存所需中间量。
因此反向前随意原地修改参数或激活可能破坏计算，不能把 no_grad 当成绕过一致性检查的万能办法。
[PyTorch 自动求导机制](https://docs.pytorch.org/docs/2.14/notes/autograd.html)

## 输入与参数

### 原始输入

整数 token ID 只是查表索引，不训练编号本身。
普通浮点观测数据也可以视作常量，只对模型参数求梯度。
上面的标量实验中，x 没有 requires_grad，w、b 仍能得到梯度。

### 中间输入

RMSNorm 或 FFN 的“输入”往往是前一个模块的输出。
它虽然不是 optimizer 要更新的独立参数，loss 仍需要对它求导，
才能把学习信号继续传到 Embedding、前层投影等参数。

所以“输入不是参数”不能推出“任何叫 x 的张量都不需要梯度”。
输入是相对于模块边界的名称，参数是可学习状态，计算图身份又是另一层概念。

### 测试输入

组件测试会直接构造浮点 x，并设 requires_grad=True。
目的是把它当作前层输出的替身，检查本模块能否正确把梯度传给上游。
并不是计划把这条测试样本交给 optimizer 训练。
这解释了为何 RMSNorm、SwiGLU 测试既检查 weight.grad，也检查 x.grad。

## 计算图身份

### 叶子与非叶子

对于 requires_grad=True 的张量，若它不是由一个被 autograd 追踪的运算产生，
而是当前图的求导起点，通常是叶子，grad_fn 为 None。
由被追踪运算产生的结果通常是非叶子，具有对应 grad_fn。
requires_grad=False 的张量按 PyTorch 约定也被视为叶子，因此“is_leaf=True”不代表“一定会有梯度”。
[is_leaf 的准确约定](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.is_leaf.html)

下面的 x 是叶子，y 是非叶子：

```python
import torch

x = torch.tensor([-1., 2.], requires_grad=True)
y = 2 * x
assert x.is_leaf and x.grad_fn is None
assert not y.is_leaf and y.grad_fn is not None
y.retain_grad()
y.sum().backward()
torch.testing.assert_close(x.grad, torch.tensor([2., 2.]))
torch.testing.assert_close(y.grad, torch.tensor([1., 1.]))
```

在真实网络里，如果 x=embedding(ids)，x 又会变成非叶子。
所以“它是模块输入”或“它在整条模型中间”都不足以判定，需要看当前图中它怎么产生。

### 梯度保存

需要梯度的叶子参与反向时，默认把计算出的梯度累加进 .grad。
非叶子同样参与反向，但默认一般不保留自己的 .grad。
retain_grad 请求保留该观察结果，不会改变其叶子身份，也不会改变计算公式。
[retain_grad 接口](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.retain_grad.html)

因此看到非叶子的 .grad 为 None，不能认为梯度没有经过它。
默认少保存这些结果是为了避免不必要的内存占用，不是省略链式法则。

### None 与零

零表示已经有一个梯度张量，其中的贡献恰好为零；None 表示当前没有保存梯度张量。
None 可能来自尚未 backward、未参与这次 loss、未要求梯度、非叶子未 retain，
也可能是 zero_grad(set_to_none=True) 刚清除了旧结果。

ReLU 关闭路径、分支贡献抵消、sum 目标下某项系数抵消，都可能产生正确的零梯度。
要结合图和操作判断，不能把所有 None 或零都归为“梯度断了”。

## 图与数值

### no_grad

在 no_grad 作用域内执行的常规运算不记录反向图，即使输入参数本身仍 requires_grad=True。
常见用途是参数初始化、准备测试参考值、推理或手动更新时不追踪更新操作。

```python
import torch
from torch import nn

layer = nn.Linear(2, 1, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[2., 0.5]]))
    observed = layer(torch.tensor([[1., 2.]]))
assert layer.weight.requires_grad
assert not observed.requires_grad
trained_path = layer(torch.tensor([[1., 2.]]))
assert trained_path.requires_grad
```

这解释了“为什么复制权重放在 no_grad”：我们是在设置测试条件，不是在定义要被求导的模型运算。
退出作用域后，正常前向仍能建立图。
更严格地说，某些显式创建 requires_grad 参数的工厂行为不受 no_grad 的普通规则影响；
不要把它理解成自动修改所有对象的标志。
[no_grad 文档](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html)

model.eval() 不等于 no_grad。eval 切换 dropout、BatchNorm 等模块模式，
默认仍可构建计算图；反之，在 no_grad 中也不自动关闭训练模式的 dropout。

### detach 与 clone

detach 返回与原张量共享数据存储、但不继承当前求导历史的张量。
它不修改原张量的 requires_grad，不会把原模型参数“取消梯度链接”。

clone 复制存储；若源张量在图中，单独 clone 通常仍保留可导关系。
detach().clone() 因而常用于保存独立数值快照：
先不继承图，再复制存储，避免后续参数更新改变快照内容。

```python
import torch
from torch import nn

parameter = nn.Parameter(torch.tensor([1., 2.]))
snapshot = parameter.detach().clone()
reference = parameter.detach().clone().requires_grad_(True)
(parameter.square().sum()).backward()
(reference.sum()).backward()
torch.testing.assert_close(parameter.grad, torch.tensor([2., 4.]))
torch.testing.assert_close(reference.grad, torch.tensor([1., 1.]))
with torch.no_grad():
    parameter.add_(10)
torch.testing.assert_close(snapshot, torch.tensor([1., 2.]))
assert parameter.requires_grad
```

reference 是新的独立叶子，用于参考计算，不会与原参数共用 .grad。
单独 detach 则仍共享存储，修改 detached 结果的数值可能影响原对象，这与梯度关系是两回事。

## 累加与更新

PyTorch 默认累加叶子梯度，而不是每次 backward 自动覆盖。
如果两个独立前向得到 L1、L2，依次反向后得到 \(\nabla L_1+\nabla L_2\)。
这既支持多个 micro-batch 的梯度累积，也意味着普通训练需要在合适位置清理旧梯度。

```python
import torch

w = torch.tensor(2., requires_grad=True)
(3 * w).backward()
torch.testing.assert_close(w.grad, torch.tensor(3.))
(4 * w).backward()
torch.testing.assert_close(w.grad, torch.tensor(7.))
w.grad = None
(5 * w).backward()
torch.testing.assert_close(w.grad, torch.tensor(5.))
```

每次这里都重新构建了一个前向图。对同一张已经释放中间量的图反复 backward 是另一问题，
不应为普通梯度累积盲目打开 retain_graph=True，否则可能额外保留大量内存。

完整一次更新通常为：

```text
清理旧梯度
    -> 一个或多个 micro-batch 的 forward
    -> 按目标权重计算 loss 并 backward
    -> 可选梯度裁剪
    -> optimizer.step
    -> scheduler.step（若当前计划按 optimizer step 调度）
```

梯度累积时，清理不能发生在每个 micro-batch 之间；目标平均也必须与整次更新的有效 token 数一致。
这里仅连接计算原理，完整的累积合同、检查点与恢复属于训练工程。

nn.Parameter 注册在 Module 上后会被 model.parameters() 收集，
optimizer 持有哪些对象决定它试图更新哪些参数。requires_grad=True 不会自动把任意 x 加入 optimizer。
同样，参数固定也不一定代表中间运算可以删掉：为前面可训练层传梯度时，仍可能需要经过固定层。

## 语言模型闭环

在第 01 篇，交叉熵对每个 logits 的梯度为 \(p_j-\mathbf{1}[j=y]\)。
用三个等分概率的类别演示，目标为第2类，即编号1：

```python
import math
import torch
import torch.nn.functional as F

logits = torch.zeros(1, 3, requires_grad=True)
loss = F.cross_entropy(logits, torch.tensor([1]))
loss.backward()
torch.testing.assert_close(loss, torch.tensor(math.log(3)))
torch.testing.assert_close(
    logits.grad, torch.tensor([[1 / 3, -2 / 3, 1 / 3]]),
)
```

提高目标 logit 能降低当前 loss，其他类别则收到相反方向的局部信号。
LM head 将这个信号传给隐藏表示；FFN 的双分支、RMSNorm 的分母依赖、attention 的上下文依赖、
残差的直接路径继续按各自导数传播；最终梯度汇入各组共享参数。

loss 并没有逐层规定“这一维应表达语法”。各模块的参数在整体目标约束下共同调整，
中间表示逐渐成为有助于预测的计算状态。
这就是从单个可微组件到整体表示学习的连接，而不是每个模块单独学习一个人工指定的小任务。

## 检验

组件测试中，固定相同权重是控制变量，独立输入与参数是防止梯度混加；
前向比较验证数值，反向比较验证依赖结构。
为参数赋非全1、非全0值，是为了让遗漏乘法或分支更容易暴露。

数值梯度检查还能通过微小扰动估计导数：

\[
\frac{\partial L}{\partial w}\approx\frac{L(w+\delta)-L(w-\delta)}{2\delta}
\]

它是额外核对方法，不是训练时的主要求导方式。
扰动过小会受舍入影响，过大不再局部；不可导点和内部精度转换也需特殊解释。
当前 RMSNorm 内部强制 float32，不能不加分析地按双精度标准检查它。

最重要的测试解释不是“绿了”，而是：
这项测试排除了哪种错误，仍有哪些路径和数值条件没有覆盖。
组件正确只是基础，能否学习、泛化以及达到任务目标还需要更高层证据。

## 总结

现在可以用一条完整链路复述整个系列：

离散编号经 Embedding 获得初始表示；
attention 让允许的位置交流信息，FFN 对已有表示做非线性加工；
归一化控制分支计算的尺度，残差以增量形式更新表示并提供直接路径；
最终 head 输出分数，loss 定义学习信号，反向传播沿实际依赖求导，优化器改变参数。

这些组件不是各自独立“完成理解”，而是在同一个参数化函数中共同服务于目标。
回到[整体结构篇](07_RESIDUAL_CONNECTIONS_AND_TRANSFORMER_BLOCKS.md)，应当能同时讲清前向的信息流和反向的学习流。
