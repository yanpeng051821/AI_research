# 线性、非线性与激活函数

> 线性层学习怎样组合特征；激活函数让组合后的响应随输入区域改变。
> 这既影响网络能表示什么关系，也影响 loss 怎样向前面的参数传递梯度。

[系列导航](README.md) · 前篇：[矩阵与线性层](02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md) · 后篇：[Embedding](04_EMBEDDINGS_AND_REPRESENTATIONS.md)

## 限制

一维线性函数 \(f(x)=2x\) 中，输入增加1，输出总增加2。
严格的线性变换满足：

\[
f(\alpha u+\beta v)=\alpha f(u)+\beta f(v)
\]

多维时就是矩阵变换 \(Wx\)。加上偏置 \(Wx+b\) 后严格称为仿射变换，
但神经网络习惯仍把它叫“线性层”。它不一定经过原点，却仍满足：

\[
f(x+\Delta x)-f(x)=W\Delta x
\]

也就是相同输入增量产生相同输出增量，与起点无关。
网络要处理的关系往往不是这样：一个特征的影响可能取决于其他特征以及当前取值范围。

### 线性堆叠

两层 \(h=2x+1\)、\(y=3h-2\) 合并后是 \(y=6x+1\)。
矩阵情形同理：

\[
y=W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)
\]

层数增加，但整个函数仍是仿射变换。不同参数分解会影响秩、优化和成本，
却不会仅靠堆叠就获得超出仿射关系的表达能力。
因此非线性不是为了“多加一步”，而是为了改变整个函数能够表达的关系。

## 激活

激活函数通常对中间特征逐元素操作，不改变 shape。
“通常没有参数”不等于“只是装饰”：它的输出值和局部导数都会进入后续计算。
以下先用 ReLU 看清机制，再理解当前 MiniMind 使用的 SiLU。

### ReLU

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

将它插在线性层之间：

\[
z=2x+1,\qquad h=\operatorname{ReLU}(z),\qquad y=3h-2
\]

| x | z | h | y |
| --- | --- | --- | --- |
| -1 | -1 | 0 | -2 |
| 0 | 1 | 1 | 1 |
| 1 | 3 | 3 | 7 |

输入两次都增加1，输出却分别增加3与6，无法用同一个 wx+b 表达。
更具体地写：

\[
y=\begin{cases}
-2,&x\leq-1/2\\
6x+1,&x>-1/2
\end{cases}
\]

左半边保持不变，右半边随输入增长。每一段可以是直线，但整体仍是非线性的。
非线性不要求图像处处弯曲，也不要求处处可导。

“根据输入决定信号是否传递”是理解 ReLU 的一个入口。
但它处理的是中间数值 z，不是在理解整个 token 后决定其重要性。
前面的 w、b 决定 z 在什么条件下为正，所以没有参数的 ReLU 仍能被可训练的上下游共同利用。
例如 \(z=wx+b\) 且 w 非零时，分界位置是 \(-b/w\)。

两个单元还可以组合出绝对值：

\[
|x|=\operatorname{ReLU}(x)+\operatorname{ReLU}(-x)
\]

前一个在正区间响应，后一个在负区间响应，下一层把两者合并。
这是“中间特征可以被后续组合”的具体证据。
它证明这组参数能表示绝对值，不证明训练一定会找到它。

### Sigmoid

\[
\sigma(z)=\frac{1}{1+\exp(-z)}
\]

Sigmoid 将实数压到0与1之间，所以常用于连续门控，或二分类模型的概率映射。
处在隐藏层时，它的输出不自动具有经过校准的概率含义，必须看其训练与使用方式。

导数为：

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

z 接近0时响应较敏感，z 很正或很负时接近饱和，导数变小。
这连接了图像与训练：如果一条很深的路径反复乘上小导数，上游的学习信号可能衰减。
并不是“Sigmoid 不好”，而是需要在适用位置理解其数值性质。

### SiLU

SiLU 把输入与自身的 sigmoid 响应相乘：

\[
\operatorname{SiLU}(z)=z\sigma(z)
\]

例如 z=2 时约为1.7616，z=-1 时约为-0.2689。
它不是“负数全丢掉”，也不是“输出必须在0和1之间”。
导数由乘积法则得到：

\[
\operatorname{SiLU}'(z)=\sigma(z)+z\sigma(z)(1-\sigma(z))
\]

较大的正数区近似线性，较负区的输出和导数接近0，中间平滑变化。
“平滑”并不保证所有输入导数都为正，也不保证不会出现较小梯度。
MiniMind 的 SwiGLU 在门控分支使用它；SiLU 是激活函数，SwiGLU 是包含投影与相乘的前馈结构，
二者不是同一个层次。[PyTorch SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html)

### 其他成员

Leaky ReLU 在负数区保留小斜率；GELU 使用 \(z\Phi(z)\)，其中 Φ 为标准正态累积分布函数，
实际实现也可使用近似。它们都属于激活，不应与 ReLU 分成互不相关的知识点。
本轮不需要背诵所有变体，先理解共同职责及其前向、导数差异。
[GELU 接口与近似约定](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html)

## 梯度

### 链式传播

把局部响应接回训练：

\[
z=w_1x+b_1,\quad h=\operatorname{ReLU}(z),\quad
\hat y=w_2h+b_2,\quad L=(\hat y-t)^2
\]

t 是目标。对第一层权重：

\[
\frac{\partial L}{\partial w_1}
=2(\hat y-t)\,w_2\,\operatorname{ReLU}'(z)\,x
\]

它同时包含误差、下游权重、激活导数和输入。参数对 loss 的影响来自整条路径，
不是看参数本身有多大。

ReLU 正数区导数1，负数区导数0；在0处数学上不可导，PyTorch 采用零梯度。
所以某次反向中一个单元的上游权重可能收不到这条路径的梯度。
这不等于整个网络停止学习，也不等于该参数永远没用。
[不可微点的框架约定](https://docs.pytorch.org/docs/2.14/notes/autograd.html#gradients-for-non-differentiable-functions)

### 计算图身份

“模块输入”描述的是结构位置，“叶子张量”描述的是当前 autograd 图中的身份。
在一个小实验中直接创建的 x 可以是叶子；真实模型中传入激活的 x 往往来自 Linear，因此是非叶子。

需要梯度的叶子在参与反向时默认累加 .grad；非叶子也参与反向，但一般不保存自身 .grad。
所以“中间结果没有 .grad”不等于“这里没有梯度经过”。

```python
import torch

x = torch.tensor([-1., 2.], requires_grad=True)
z = 2 * x + 1
z.retain_grad()
y = torch.relu(z)
y.sum().backward()
assert x.is_leaf and not z.is_leaf
torch.testing.assert_close(z.grad, torch.tensor([0., 1.]))
torch.testing.assert_close(x.grad, torch.tensor([0., 2.]))
```

本例 z=[-1,5]，ReLU 的局部导数为[0,1]，再乘前面的2得到 x 的梯度。
retain_grad 只要求保存观察结果，不会把 z 变为叶子，也不会改变这条计算。
detach、no_grad、梯度累加与参数更新的完整区分集中在[第 08 篇](08_AUTOGRAD_AND_PARAMETER_UPDATES.md)。

## 特征加工

可以把一个带激活的 MLP 理解为三步：

```text
Linear：组合输入，形成若干响应量
激活：让响应按当前取值非线性变化
Linear：重新组合这些响应，形成下一层表示
```

例如 ReLU 的各单元可以在不同区域起作用，下一层将它们拼成更复杂的整体关系。
这个机制不要求一个单元对应一个人类概念，也不表示“激活越大，语义越重要”。
可表达某种关系、能从有限数据学到它、能在新输入上泛化，是不同问题。

在 [B,T,D] 上逐元素激活，不会从位置 t 读取位置 t-1。
它能加工已经包含上下文的向量，但不会独立完成跨位置交流。
Attention 则通过不同位置之间的匹配与汇总引入上下文。

“激活引入非线性”也不意味着模型中只有激活才非线性。
RMSNorm 的分母依赖输入，attention 权重依赖输入的 softmax，二者也具有非线性。
不过它们分别承担尺度控制与信息路由，不等同于 FFN 内的逐元素特征响应。
数学性质相同，不代表模块职责相同。

## 检验

下面把线性堆叠与带激活路径放在相同输入上，避免仅凭画出的曲线记忆：

```python
import torch

x = torch.tensor([-1., 0., 1.])
z = 2 * x + 1
linear = 3 * z - 2
nonlinear = 3 * torch.relu(z) - 2
torch.testing.assert_close(linear, 6 * x + 1)
torch.testing.assert_close(nonlinear, torch.tensor([-2., 1., 7.]))
torch.testing.assert_close(nonlinear[1:] - nonlinear[:-1], torch.tensor([3., 6.]))
torch.testing.assert_close(torch.relu(x) + torch.relu(-x), x.abs())
torch.testing.assert_close(torch.nn.functional.silu(x), x * torch.sigmoid(x))
```

不同验证的意义不同：第一项验证代数合并，第二与第三项展示整体关系无法用同一斜率解释；
绝对值展示组合的表达能力；最后一项检查 SiLU 公式。
它们不证明哪个激活在真实任务上最好，也不证明深层网络必然容易训练。

## 复习

**没有可训练参数为什么仍有作用？** 前向改变表示，反向改变局部导数；上下游的可训练参数会适应这种计算。

**负数被置零是否意味着这个 token 被删除？** 不是，ReLU 作用于某个特征分量。
其他通道、残差路径及其他位置仍然存在。

**为什么不是在线性层后加什么都可以？** 层间操作的数值与导数决定整个函数和学习过程。
激活不是泛指“多一步计算”，而是有具体响应机制的非线性映射。

下一篇进入[Embedding](04_EMBEDDINGS_AND_REPRESENTATIONS.md)：在这些连续特征加工之前，离散编号怎样获得可训练表示。
