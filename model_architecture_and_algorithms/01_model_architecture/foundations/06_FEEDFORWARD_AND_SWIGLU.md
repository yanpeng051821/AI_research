# 前馈网络与 SwiGLU

> Transformer 的前馈子层对每个位置的表示进行非线性加工。
> SwiGLU 用两组特征投影形成输入相关的乘法调制，再将结果组合回模型宽度。

[系列导航](README.md) · 前篇：[归一化](05_NORMALIZATION_AND_TRAINING_SCALE.md) · 后篇：[残差与整体结构](07_RESIDUAL_CONNECTIONS_AND_TRANSFORMER_BLOCKS.md)

## 职责

假设一个位置的隐藏向量已经通过 attention 汇入了前文信息。
下一步不仅需要“再从别处读取”，还需要对当前已有信息做更复杂的组合：
哪些特征共同出现，怎样转换为适合后续层和最终预测的表示。

FFN 承担这种逐位置的特征加工。**它不直接在不同位置之间读取信息，
但输入可以已经包含上下文，因此它加工的并不只是最初那个孤立 token。**

这里的“前馈”指沿网络连接向前计算，不是说它没有反向传播。
“逐位置”指同一个子层在每个位置独立执行，不是每个位置有独立参数。
原始 Transformer 的 position-wise FFN 就采用这一组织方式。
[原论文第 3.3 节](https://arxiv.org/html/1706.03762v7#S3.SS3)

## 普通前馈

先看两层的形式，用列向量写：

\[
h=\phi(W_{\mathrm{up}}x+b_{\mathrm{up}}),\qquad
y=W_{\mathrm{down}}h+b_{\mathrm{down}}
\]

输入宽度 D，中间宽度 H，输出回到 D。
第一层形成多个特征组合，激活使这些组合有非线性响应，第二层汇总响应。
如果去掉中间非线性，两层整体仍是一个仿射变换，见[第 03 篇](03_LINEARITY_NONLINEARITY_AND_ACTIVATIONS.md)。

H 通常比 D 大，给模型更多中间计算通道。
这并不是获得更多外部信息，而是用更多可学习组合和非线性响应处理原有信息。
如同两个 ReLU 能组合出绝对值，多个中间响应可表达单个线性映射不能表达的关系。
更多通道意味着更多计算与参数，不等于每个通道自动对应一个明确概念。

输出回到 D，一方面供下一层继续使用统一宽度，另一方面便于与输入残差相加。
因此升维与降维不是为了“先放大再缩小数值”：改变的是表示通道数。

## 门控前馈

当前 MiniMind 的 dense FFN 使用 SwiGLU 形式，省略偏置：

\[
g=W_gx,\qquad u=W_ux,\qquad
a=\operatorname{SiLU}(g)\odot u,\qquad y=W_da
\]

⊙ 是逐元素乘法。三个投影的 shape 分别为：

| 参数 | shape | 作用 |
| --- | --- | --- |
| W_g | [H,D] | 产生供 SiLU 调节的响应 |
| W_u | [H,D] | 产生与这些响应相乘的特征 |
| W_d | [D,H] | 将门控后的 H 维结果重新组合为 D 维 |

两个分支都从同一个 x 出发，参数不同，学习到的组合也可以不同。
up 分支不是“未经处理的原信息”，它已经经过投影；gate 分支也不是人工设置的一张开关表。

第 k 个中间通道为：

\[
a_k=\operatorname{SiLU}(w_{g,k}^Tx)\,(w_{u,k}^Tx)
\]

一个输入相关的响应调节另一个输入相关的响应，这就是这里的门控含义。
SiLU 值可为负，也可大于1，所以 gate 不等于0到1的概率，不是只能“保留或删除”的开关。
负值可以改变符号，大幅值可以放大，对应关系由整体目标训练出来。
[GLU 变体论文](https://arxiv.org/abs/2002.05202)

## 数值链路

使用当前实践测试的同一组权重，D=2、H=3：

\[
x=[1,2]^T,\quad
W_g=\begin{bmatrix}1&0\\0&1\\1&-1\end{bmatrix},\quad
W_u=\begin{bmatrix}2&0\\0&1\\1&1\end{bmatrix},\quad
W_d=\begin{bmatrix}1&0&1\\0&1&-1\end{bmatrix}
\]

先分别投影：

\[
g=[1,2,-1]^T,\qquad u=[2,2,3]^T
\]

gate 应用 SiLU，得到约：

\[
\operatorname{SiLU}(g)=[0.731059,1.761594,-0.268941]^T
\]

逐通道相乘：

\[
a=[1.462117,3.523188,-0.806824]^T
\]

最后 W_d 的第一行相加第1、3项，第二行用第2项减第3项：

\[
y=[0.655293,4.330013]^T
\]

这条链路能区分多个常见错误：把 SiLU 写成 sigmoid；先把 g、u 相乘再激活；
把逐元素乘法当矩阵乘法；最后漏掉 down 投影。它们有时仍能保持 shape，却不再是同一个函数。

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[[1., 2.]]])
wg = torch.tensor([[1., 0.], [0., 1.], [1., -1.]])
wu = torch.tensor([[2., 0.], [0., 1.], [1., 1.]])
wd = torch.tensor([[1., 0., 1.], [0., 1., -1.]])
g = x @ wg.T
u = x @ wu.T
activated = g * torch.sigmoid(g)
middle = activated * u
output = middle @ wd.T
torch.testing.assert_close(g, torch.tensor([[[1., 2., -1.]]]))
torch.testing.assert_close(activated, F.silu(g))
torch.testing.assert_close(
    output, torch.tensor([[[0.655293, 4.330013]]]), atol=1e-6, rtol=1e-5,
)
assert middle.shape == (1, 1, 3)
assert output.shape == (1, 1, 2)
```

分步骤手算不是必须脱离矩阵写一长串标量代码。矩阵参考同样有意义；
关键是能指出每一步的输入与输出，而不是复制待测函数后给变量换名字。

## 梯度分流

门控不只是前向相乘，反向也有两个分支。
设 loss 对 a 的梯度为 q，则：

\[
\frac{\partial L}{\partial u}=q\odot\operatorname{SiLU}(g)
\]
\[
\frac{\partial L}{\partial g}=q\odot u\odot\operatorname{SiLU}'(g)
\]

输入 x 同时被两个投影使用，因此：

\[
\frac{\partial L}{\partial x}
=W_u^T\frac{\partial L}{\partial u}
+W_g^T\frac{\partial L}{\partial g}
\]

学习信号经 down 投影到中间通道，再分到 gate、up，最后汇回 x。
因此输入梯度与三组参数梯度都值得验证。只看到 output.grad 或只检查 down 权重不足以覆盖整条链。

梯度也不要求每个元素都非零。在上面的 W_d 中，如果 loss=y.sum()，
第三个中间通道在两项输出中的系数是 +1 和 -1，其贡献会抵消。
这说明零梯度可能来自正确的代数关系，不能直接判定“这一层坏了”。

## 位置独立性

对 [B,T,D]，每个投影都只作用于最后一维，SiLU 与相乘也不跨位置。
固定参数和随机性时，只改变 X[0,1]，不应改变同一 FFN 中 X[0,0]、X[0,2] 或其他样本对应的输出。

这不是声称整个语言模型没有位置交流。若在 FFN 前经过 attention，修改某个 token
可能改变后续位置的输入表示，随后那些位置的 FFN 输出也会变化。
必须分清“单个子层直接依赖什么”与“整条网络间接依赖什么”。

也不能反向要求“被改位置的输出一定改变”。非线性、权重零空间或数值条件可能使某次改动没有反映到输出。
测试应保护其余位置不受直接影响的结构性质，而不是把一般直觉当成必然数学事实。

## 参数与实现

三个无偏置投影共有 \(DH+DH+HD=3DH\) 个参数；
普通两层无偏置 FFN 为2DH。它们若使用同一个 H，参数量和计算量并不相同。
这帮助理解结构成本，但不要求当前进行激活或宽度对比实验。

本地官方配置通过 hidden_act 选择激活，当前默认 silu；
手写 SwiGLU 固定使用 SiLU，是对当前学习目标的简化，不等于支持官方全部 FFN 变体。
官方还有 MoE 路径，本篇只讨论 dense 子层，不将 MoE 的专家路由与这里的逐通道门控混为一谈。

当前实现的对应关系见
[手写 feedforward.py](../../../training_engineering/case_studies/minimind/implementation/src/minimind_lab/models/feedforward.py)；
官方见[模型文件](../../../../minimind/model/model_minimind.py)的 FeedForward。

## 检验

现有[三个测试](../../../training_engineering/case_studies/minimind/implementation/tests/test_feedforward.py)各有不同职责：

1. 固定权重的前向对照，验证投影、SiLU、相乘、down 的连接与数值。
2. 独立参考路径的梯度对照，验证输入与三组参数的反向通路。
3. 单位置扰动，验证 FFN 不直接混合位置与样本。

前向参考使用框架 SiLU，梯度参考展开成 g*sigmoid(g)，可以从不同表达方式核对关系。
这降低重复犯同一错误的风险，但仍不是对全部数值条件的证明。
小输入可解释，随机输入可扩大覆盖，完整训练才回答是否学到目标能力，三者不可互相替代。

## 复习

**FFN 在哪里获得上下文？** 它读取前一子层交付的向量，向量可能已由 attention 汇入上下文；
它自己不在不同位置间查找信息。

**升维为什么有意义？** 为非线性组合提供更多中间通道，不是凭空增加独立输入信息。

**gate 是判断语义重要性的专用神经元吗？** 不应这样预设。
它是输入相关的数值调制分支，其有用行为由整体任务训练形成。

前馈输出下一步不是孤立替代全部表示，而是加入残差路径。
这正是[下一篇](07_RESIDUAL_CONNECTIONS_AND_TRANSFORMER_BLOCKS.md)要连接的结构关系。
