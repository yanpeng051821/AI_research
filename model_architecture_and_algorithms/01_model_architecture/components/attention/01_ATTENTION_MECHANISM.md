# 注意力机制：原理、计算与模型连接

[专题导航](README.md) · [模型全局图](../../models/decoder_only_transformer/01_DECODER_ONLY_TRANSFORMER.md)

在语言模型中，一个位置要预测后续内容，就需要结合它所在的上下文。Attention 为这种联系提供了具体的计算方式：根据当前表示计算位置之间的匹配权重，再按权重汇总其他位置的特征。

理解这套机制，需要把三个层次连起来：单次加权汇总怎样计算，模型怎样从输入中产生 Q、K、V，以及训练怎样改变这些计算。本文以单头缩放点积注意力为主线，从一个可手算的例子出发，逐步扩展到序列、因果约束和完整模块，最后用梯度与数值对照检验理解。

## 1. Attention 的作用

### 1.1 跨位置的信息交流

考虑模型已经收到“我喜欢”，需要预测下一个 token。为了便于说明，暂时把这些词当作位置；真实 tokenizer 的切分可能不同。

Embedding 给每个 token ID 一个向量。RMSNorm 调整一个向量的尺度，FFN 对这个向量做非线性加工。如果网络只有这些逐位置操作，那么“喜欢”位置只能加工自己的输入，不能直接读取“我”位置。即使叠很多层，这条跨位置的计算依赖也不会凭空出现。

Attention 要补上的是这条路径。它让“喜欢”位置能够读取前面位置的表示，再形成新的特征向量。这份向量交给后续残差与 FFN，最后才经 LM head 变成词表分数。因此，Attention 的输出是内部表示，并不是直接生成一个词，也不是由它单独完成全部语言理解。

先把它放进当前学习的 pre-norm Decoder：

```text
已有表示 X [B, T, D]
  ├─────────────────────────────────────────┐
  └─ RMSNorm → Attention → 更新量 [B, T, D] ─┤ 相加
                                            ↓
                                     新表示 [B, T, D]
                                            ↓
                                    后续 FFN 与其他层
```



### 1.2 内容相关的聚合

一种最简单的办法是把所有位置取平均。这样确实建立了交流，但每次都给所有位置同样的权重。另一种办法是始终读取固定位置。它能表达某种固定结构，却无法随输入内容改变读取方式。

我们希望聚合比例取决于当前输入。不同位置可以从同一组内容中汇总出不同的结果，同一个位置遇到不同上下文时，也应当能改变权重。

这与早期机器翻译中的一个问题相连：一些 encoder-decoder 模型把源序列压缩为固定长度表示，再交给解码器。 Bahdanau 等人的方法让解码器在各输出步结合多个编码器状态，构造随步骤改变的上下文向量。它采用加性打分；本文随后推导的缩放点积是另一种具体实现。这一工作展示了按当前解码状态组织上下文的思路，参见[原论文](https://arxiv.org/abs/1409.0473)。

缩放点积注意力把这个过程分为匹配与汇总：query（Q）和 key（K）用于计算匹配分数，value（V）提供被汇总的内容。下面先固定这三组向量，观察一次计算；它们在模型中的来源随后再展开。

## 2. 单次注意力计算

### 2.1 匹配分数与聚合权重

我们沿用实践脚本中的例子。现在有一个 query，以及三个对应的 key/value：

```text
q  = [1, 0]

k₁ = [1, 0]       v₁ = [10,  0]
k₂ = [0, 1]       v₂ = [ 0, 20]
k₃ = [1, 1]       v₃ = [30, 30]
```

每一对 key/value 属于同一个候选位置。query 与 key 比较后得到该位置的权重，再把权重用于这个位置的 value。可以把 key 理解为匹配依据、value 理解为要传递的特征，但它们都只是可计算的向量，没有预先写好的自然语言“关键词”或“答案”。

先计算点积：

\[
q\cdot k_1=1,\qquad q\cdot k_2=0,\qquad q\cdot k_3=1.
\]

第一个与第三个位置同分，说明在这组具体数值下，query 对二者的匹配程度相同。点积不是普遍意义上的语义相似度，也不是归一化的余弦相似度；向量方向和长度都会影响它。模型需要通过训练，让这个打分对目标任务有用。

每个 key 有两个特征，记为 \(d_k=2\)。缩放后：

\[
s=\frac{[1,0,1]}{\sqrt{2}}\approx[0.7071,0,0.7071].
\]

除以根号维度用于控制点积分数的尺度，具体原因在下一节讨论。随后沿三个候选位置计算 softmax：

\[
a_j=\frac{e^{s_j}}{\sum_{\ell=1}^{3}e^{s_\ell}},
\qquad
a\approx[0.4011,0.1978,0.4011].
\]

这些权重非负，总和为 1，表示各位置在本次汇总中所占的比例。每个 query 都有自己的一行权重；这里的概率形式描述的是聚合比例，而非答案正确率。

### 2.2 Value 的加权汇总

现在按权重组合三个 value：

\[
o=a_1v_1+a_2v_2+a_3v_3.
\]

把两个输出特征分开看：

\[
o_1=0.4011\times10+0.1978\times0+0.4011\times30\approx16.0445,
\]

\[
o_2=0.4011\times0+0.1978\times20+0.4011\times30\approx15.9889.
\]

每个位置的权重会乘在该位置的整个 value 向量上；同一头在这次读取中，没有为 value 的每个特征重新算一套位置权重。结果也没有直接选中某一行 V，而是融合了多行的内容。

下面用矩阵计算与逐行展开互相核对。手算结果保留了四位小数，因此比较时给舍入留出误差：

```python
import torch

q = torch.tensor([[1., 0.]])
k = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
v = torch.tensor([[10., 0.], [0., 20.], [30., 30.]])

scores = (q @ k.T) / q.shape[-1] ** 0.5
weights = torch.softmax(scores, dim=-1)
output = weights @ v
expanded = sum(weights[0, j] * v[j] for j in range(3))

torch.testing.assert_close(output[0], expanded)
torch.testing.assert_close(
    output, torch.tensor([[16.0445, 15.9889]]), atol=1e-4, rtol=0
)
torch.testing.assert_close(weights.sum(-1), torch.ones(1))
print(scores, weights, output, sep="\n")
```

这个例子解释了最后一步为什么是 `weights @ v`：权重的行与 V 的列相乘。前面的 `q @ k.T` 需要转置 K，以便让每个 query 分别与每个 key 的整条特征向量比较；最后的 V 已经按“位置 × 内容特征”排列，不再转置。

### 2.3 权重与内容的独立变化

如果仅把 \(v_2\) 从 [0,20] 改为 [0,100]，Q、K 完全不变，那么 scores 和 weights 也不变。输出第二个特征的增量为：

\[
\Delta o_2=a_2(100-20)\approx15.824.
\]

输出第一个特征不变。这正好区分了“决定传递多少”和“提供传递内容”的职责。

这个观察有一个前提：我们直接改变的是 V。稍后 Q/K/V 都由 X 投影而来时，若改变原始 X，则对应位置的 Q、K、V 通常会一起变化，不能沿用“权重不变”的结论。

## 3. 缩放与 Softmax

上述计算中，缩放控制分数的数值尺度，Softmax 把分数转换为各位置共享的一组聚合比例。这两步共同决定了信息汇总的方式，也影响反向传播时的学习信号。

### 3.1 点积的尺度

点积会把 \(d_k\) 个乘积加在一起。在一个用于分析尺度的简化假设下， q、k 的分量相互独立、均值为 0、方差为 1，那么每个 \(q_rk_r\) 的方差为 1，点积的方差为 \(d_k\)。因此其标准差随 \(\sqrt{d_k}\) 增长：

\[
\operatorname{Var}\!\left(\sum_{r=1}^{d_k}q_rk_r\right)=d_k,
\qquad
\operatorname{Var}\!\left(\frac{q\cdot k}{\sqrt{d_k}}\right)=1.
\]

维度变大时，若分数差异随之变得很大，softmax 会更容易集中在极少数位置。这可能让其局部梯度很小，使匹配关系难以调整。缩放用来控制这种尺度效应。

独立、单位方差是推导假设，不是训练中永远成立的事实；缩放也不保证不会出现尖锐权重或梯度问题。分母用的是参与点积的特征宽度，不是 token 数、batch 大小或词表大小。这也与 RMSNorm 不同：RMSNorm 根据每个输入向量的实际数值计算尺度，缩放点积这里使用固定的维度因子。定义和尺度动机见 [Transformer 原论文第 3.2 节](https://arxiv.org/html/1706.03762v7#S3.SS2)。

### 3.2 权重的归一化

softmax 把一组相对分数转换为平滑的聚合比例。每个分数出现在同一个分母里，所以提高一个位置的分数，通常也会降低其他位置分得的比例。

对任意常数 c，都有：

\[
\operatorname{softmax}(s+c)=\operatorname{softmax}(s).
\]

因为分子和分母中的 \(e^c\) 会约掉。因此匹配权重取决于同一行分数之间的差异，并不取决于这行分数整体加了多大常数。数值实现可以先减去行最大值，以降低指数溢出的风险。

softmax 还使权重随分数连续变化。与直接用 argmax 选位置相比，这给匹配参数提供了可用于普通反向传播的路径。但它不是注意力唯一可能的归一化方式，也不保证模型一定学到我们期望的关系。

在关闭 dropout、各 value 有限且至少存在一个有效候选时，单个头的 AV 是这些 value 的凸组合。经过输出投影、残差和后续网络后，整个模块输出不再受这一个凸组合范围的限制。这个局部性质不能扩展为“整个 Transformer 只能做平均”。

## 4. Q、K、V 的生成

### 4.1 隐藏表示与投影维度

目前 Q/K/V 是我们手写的数值。在模型里，它们由当前层接收的隐藏表示 X 计算得到。忽略 batch，设 X 有 T 行、每行 D 个特征：

\[
X\in\mathbb{R}^{T\times D},
\quad
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
\]

本文采用行向量公式，三个参数矩阵分别为：

\[
W_Q,W_K\in\mathbb{R}^{D\times d_k},
\qquad
W_V\in\mathbb{R}^{D\times d_v}.
\]

PyTorch `nn.Linear(D, d_k, bias=False)` 保存的 weight 形状是 [d_k,D]，所以代码等价于 `X @ layer.weight.T`。这是存储约定的区别，计算关系相同。

### 4.2 投影参数的分工

同一个位置既可能读取其他位置，也可能被其他位置读取。三个投影使这些角色可以采用不同的特征组合： Q 形成这个位置用于发起匹配的表示，K 形成供别人匹配的表示，V 形成要被传递的表示。

它们来自同一个 X，不要求数值相同，因为参数矩阵不同。训练也没有固定指定某列是“人物”、某列是“动作”。可学习投影提供了调整特征组合的自由，任务 loss 决定怎样调整它们才有用。

对于 self-attention 的点积部分：

\[
q_i\cdot k_j=(x_iW_Q)(x_jW_K)^\top
=x_iW_QW_K^\top x_j^\top.
\]

这说明打分本身也在学习：投影参数决定原始特征如何参与两个位置之间的匹配。即便原始点积 \(x_i\cdot x_j\) 不大，经过投影后的匹配也可能很强。由于 Q/K 投影可以不同，分数不必对称；再加上逐行 softmax 和因果约束，两个位置互相读取的比例更不必相同。

### 4.3 模型参数与注意力权重

容易混淆的“权重”实际上有两种：

| 对象 | 如何得到 | 是否由 optimizer 直接更新 |
| --- | --- | --- |
| 投影参数 W_Q、W_K、W_V、W_O | 初始化后作为模型参数保存 | 是 |
| 注意力权重 A | 每次前向由当前 Q/K 和 mask 计算 | 否，通过投影参数变化而间接改变 |

不同 token、不同序列共用同一层的投影参数，但它们计算出的 Q/K/V 和 A 可以不同。不同 block 通常各有自己的参数，因此不能把所有层的 Q/K/V 理解成同一组向量反复使用。

第一层的分支输入来自 Embedding 及前置归一化；更深层接收已经经过前面 block 更新的表示。后层读取的“内容”已经包含之前汇入的上下文，不是始终回到原始词向量查表。

## 5. 序列与批量计算

### 5.1 多个 Query 的并行计算

有了投影之后，每个位置都能发起一次读取。把所有 query 排成矩阵，就能一次计算所有位置的输出。

对于单头 self-attention：

| 张量 | 形状 | 每个轴表示什么 |
| --- | --- | --- |
| X | [T,D] | 位置、模型特征 |
| Q、K | [T,d_k] | 位置、匹配特征 |
| V | [T,d_v] | 位置、内容特征 |
| S = QKᵀ / √d_k | [T,T] | 读取位置、被读取位置 |
| A = softmax(S) | [T,T] | 每个读取位置对应的一行聚合比例 |
| O = AV | [T,d_v] | 读取位置、汇总后的内容特征 |

记 \(A_{ij}\) 为第 i 个 query 从第 j 个位置读取的权重：

\[
o_{ir}=\sum_{j=1}^{T}A_{ij}V_{jr}.
\]

i 决定哪一个位置接收信息，j 遍历信息来源，r 是输出特征。这比只记住“矩阵相乘”更直接地说明每条数据依赖。

若把开始的 q 扩成三行，权重就从 [1,3] 变成 [3,3]，输出从 [1,2] 变成 [3,2]。每个 query 都有自己的权重行，它们不会混成一个共享的 query。

### 5.2 自注意力与交叉注意力

self-attention 的 Q/K/V 来自同一条输入序列；cross-attention 中，Q 可以来自目标序列，K/V 来自另一条来源序列。后者可以有不同长度 \(T_q,T_k\)：

\[
Q:[T_q,d_k],\quad K:[T_k,d_k],\quad V:[T_k,d_v],
\quad A:[T_q,T_k],\quad O:[T_q,d_v].
\]

必要的对应关系是：Q/K 的匹配宽度一致，K/V 的来源位置一一对应。“self”描述来源关系，“causal”描述哪些位置允许被看见，是不同的分类维度。我们的 Decoder 使用 causal self-attention。

### 5.3 批量计算与轴的含义

加入 batch 后，单头张量成为 [B,T,d]。矩阵乘法在每个 batch 内独立进行；不同训练样本之间不会因为放进同一个 batch 就互相读取。高维转置使用 `transpose(-2,-1)` 明确交换最后两个轴，不把二维例子的 `.T` 直接套到所有高维张量上。

## 6. 因果约束与补齐

前面的完整权重矩阵默认允许读取所有位置。对于自回归训练，这会产生一个直接问题：整条训练序列已经在内存里，若前面的位置读到后面的答案，就违背了预测时可用的信息条件。

### 6.1 因果掩码

用零起始位置，位置 i 的表示可以使用位置 0 到 i；它的 logits 用来预测位置 i+1。因此在长度为 3 的序列里，可见性是：

```text
                    被读取位置 j
                    0  1  2
读取位置 i = 0      1  0  0
读取位置 i = 1      1  1  0
读取位置 i = 2      1  1  1
```

我们在 softmax 之前把禁止位置的分数设为负无穷：

\[
A=\operatorname{softmax}(S+M),
\quad
M_{ij}=
\begin{cases}
0,&j\leq i,\\
-\infty,&j>i.
\end{cases}
\]

有限的正常分数参与竞争，禁止位置的指数为 0。不能把禁止位置的分数简单设为 0，因为 \(e^0=1\)，它仍会获得权重。也不能先算完整 softmax 再直接把未来权重清零，那样剩余权重没有按有效候选重新归一化。

这里的 mask 允许当前位置读自己；禁止的是用来预测下一个 token 时偷看下一个 token。 labels 的 causal shift 与 Attention 的可见性约束分别承担预测对齐、信息限制两个职责。

### 6.2 补齐位置与监督范围

batch 中的短样本往往补了 pad。padding mask 通常至少要把 pad 的 key/value 位置排除，防止真实 query 从补齐位置汇总内容。补齐的 query 行可能仍然计算出输出，随后通过有效位置约定、 loss mask 或显式处理把它排除在需要使用的结果之外。不能仅凭“它是 pad”就假设前向结果自动为零。

| 机制 | 控制什么 | 不能替代什么 |
| --- | --- | --- |
| causal mask | 是否允许读取未来位置 | 不能决定哪些目标参与 loss |
| padding mask | 是否读取补齐位置 | 不能自动定义整个训练监督范围 |
| labels 中的 ignore_index | 哪些预测位置参与 loss | 不能阻止前向读取未来信息 |

实现时还要约定：用于 softmax 的有效 query 行至少有一个可读取位置。若整行都是负无穷，普通 softmax 可能得到 NaN；不同融合接口可能另有处理行为，必须看具体合同。下面的观察使用所有位置都有效的短序列，避免把这个边界混进第一次数值实验。

### 6.3 因果性的行为验证

用一个不对称的小例子，把未来位置的 K 和 V 都改掉，再观察早期输出。如果 mask 真正发挥作用，前两个位置的输出不应变化：

```python
import torch

q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
k = q.clone()
v = torch.tensor([[10., 0.], [0., 20.], [30., 30.]])
allowed = torch.ones(3, 3, dtype=torch.bool).tril()

scores = q @ k.T / 2 ** 0.5
weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
output = weights @ v

k_changed, v_changed = k.clone(), v.clone()
k_changed[2] = torch.tensor([100., -100.])
v_changed[2] = torch.tensor([-500., 700.])
changed_scores = q @ k_changed.T / 2 ** 0.5
changed_weights = changed_scores.masked_fill(
    ~allowed, float("-inf")
).softmax(-1)
changed_output = changed_weights @ v_changed

assert torch.count_nonzero(weights[~allowed]).item() == 0
torch.testing.assert_close(weights.sum(-1), torch.ones(3))
torch.testing.assert_close(output[:2], changed_output[:2])
torch.testing.assert_close(output[0], v[0])
```

最后一个断言也有意义：第 0 个位置只能读取自己，因此在这个不含 dropout 的单头聚合里，它的权重必为 [1,0,0]，输出就是 \(v_0\)。加入输出投影和残差后，完整模块的输出当然不必等于原始 X。

这个实验验证的是局部聚合的因果性。完整模型仍需另外验证“改变未来 token，不影响过去位置 logits”，因为错误也可能出现在 mask 以外的模块连接或数据处理里。

## 7. 输出投影与模型连接

### 7.1 内部聚合与模块输出

至此，单头注意力已经能在允许的位置之间完成信息汇总，得到 [T,d_v] 的结果。模型的残差主路径则保持 [T,D]，因此还需要把内部汇总结果映射回模型的隐藏维度。

多个头各自得到内容向量后，会拼接起来，再经过输出投影：

\[
O_{\mathrm{module}}
=\operatorname{Concat}(O_1,\ldots,O_{n_q})W_O.
\]

输入输出关系因此是：

```text
X [B,T,D]
  → Q/K/V 投影
  → 各头内部匹配与聚合
  → 拼接 [B,T,n_q × d_v]
  → 输出投影 [B,T,D]
  → 加回残差主路径
```

在当前 MiniMind 对照版本中，每个头的 Q/K/V 宽度都取 d_head；默认 \(n_q\times d_{\mathrm{head}}=D\)。于是拼接后已经是 D 维， o_proj 就是 D 到 D 的线性层。它仍然会混合各头产生的特征，同宽度不等于恒等映射。如果内部拼接宽度不是 D，输出投影则同时承担维度转换。

### 7.2 同维度投影的作用

例如一个无偏置的二维线性层，完全可以把 [1,2] 变成 [3,-1]，尽管前后都是二维：

```python
import torch
from torch import nn

projection = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    projection.weight.copy_(torch.tensor([[1., 1.], [1., -1.]]))
result = projection(torch.tensor([[1., 2.]]))
torch.testing.assert_close(result, torch.tensor([[3., -1.]]))
```

### 7.3 MiniMind 中的连接

多头的拆分与换轴将在后续文章中展开。这里先交代它在计算链中的位置：PyTorch 矩阵乘法使用最后两维作为矩阵，所以通常把 [B,T,n_q,d_head] 换成 [B,n_q,T,d_head]，让每个头内部的 [T,d_head] 参与点积，得到 [B,n_q,T,T]。换轴整理的是运算对象，不创建新的可学习参数，也不在这个步骤里汇总内容。

本次源码观察见 [MiniMind 固定提交](https://github.com/jingyaogong/minimind/blob/3f1a7cc25b19a861cd1bd6ed313be526b9ecdaf8/model/model_minimind.py)。其中的 Q/K 归一化、RoPE、GQA、dropout 与缓存是在基础计算之上的进一步设计，后续专题将分别讨论它们的作用。

## 8. Attention 的学习过程

前向计算把当前参数和输入转换为上下文表示。训练则从最终预测的误差出发，调整这些参数，使后续的匹配和汇总更有利于预测目标 token。

语言模型通常没有一张人工标注的“正确注意力矩阵”。训练先通过完整网络得到词表 logits，用目标 token 计算 loss，然后把梯度沿实际依赖传回来：

```text
目标 token 与 logits → loss
                         ↓ backward
                 LM head 与后续层
                         ↓
                  Attention 输出 O
                 ↙               ↘
             聚合比例 A         内容 V
                 ↓               ↓
             匹配分数 S         W_V
              ↙     ↘
             Q       K
             ↓       ↓
            W_Q     W_K
```

Q/K/V 都由 X 产生，因此梯度还会沿三个投影分支汇回 X，再继续传给更早的层。 optimizer 最后根据参数梯度更新 W；A、Q、K、V 则在下一次前向重新计算。

### 8.1 内容与聚合权重的梯度

先只看 \(O=AV\)。设下游已经给出 \(G=\partial L/\partial O\)，则：

\[
\frac{\partial L}{\partial V}=A^\top G,\qquad
\frac{\partial L}{\partial A}=GV^\top.
\]

第一条表示一个 value 被多个 query 读取时，会收到来自这些输出位置的梯度之和。第二条表示调整某个聚合比例会怎样影响 loss，要结合这个 value 的内容和下游梯度一起判断。

因此，“注意力权重大”并不直接等于“这个位置一定获得更大的总梯度”，还要看其他 query、梯度方向、相互抵消与后续路径。

### 8.2 匹配分数与投影参数的梯度

对一行 \(a=\operatorname{softmax}(s)\)，设上游对权重的梯度为 u。链式法则给出：

\[
\frac{\partial L}{\partial s_j}
=a_j\left(u_j-\sum_\ell a_\ell u_\ell\right).
\]

这说明一个分数的更新与同一行其他候选相关。softmax 的作用不只是在前向把数变成比例，它也决定了学习信号如何在互相竞争的位置之间传递。

再经过 \(S=QK^\top/\sqrt{d_k}\)，单头矩阵形式为：

\[
\frac{\partial L}{\partial Q}
=\frac{(\partial L/\partial S)K}{\sqrt{d_k}},
\qquad
\frac{\partial L}{\partial K}
=\frac{(\partial L/\partial S)^\top Q}{\sqrt{d_k}}.
\]

最后 \(Q=XW_Q\) 的参数梯度为 \(X^\top(\partial L/\partial Q)\)，其他投影同理。不需要先背住这些式子；重点是看见“输出是否有用”的信号，如何同时影响传递内容和匹配方式。

若某行只有一个可见位置，softmax 恒为 1，这一行通过权重选择产生的 Q/K 梯度为 0 是正常的。第一个位置的 K 仍可能被后续 query 使用，共享的投影参数也会从其他位置获得梯度，所以这不等于整个 Q/K 投影都无法学习。

### 8.3 梯度路径的检查

下面使用随机小输入与随机投影检查梯度路径。这里的平方均值只是观察用的目标，不是语言模型训练的交叉熵，也不代表训练出了语言能力：

```python
import torch
from torch import nn

torch.manual_seed(7)
x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
q_proj = nn.Linear(4, 2, bias=False).double()
k_proj = nn.Linear(4, 2, bias=False).double()
v_proj = nn.Linear(4, 2, bias=False).double()

q, k, v = q_proj(x), k_proj(x), v_proj(x)
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
scores = q @ k.transpose(-2, -1) / 2 ** 0.5
weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
output = weights @ v
output.square().mean().backward()

for name, tensor in [
    ("input", x),
    ("W_Q", q_proj.weight),
    ("W_K", k_proj.weight),
    ("W_V", v_proj.weight),
]:
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    print(name, "grad norm:", tensor.grad.norm().item())
```

梯度存在只证明反向路径连通，不能独立证明计算正确。还要与另一实现比较数值和梯度，并使用因果性等行为约束防止“形状对了、计算对象却错了”。

## 9. 计算验证

前面的手算、因果实验和梯度检查分别提供了不同层次的证据。将它们合起来，才能区分公式理解、实现正确性和模型实际效果：

| 验证 | 目的 | 不能单独证明什么 |
| --- | --- | --- |
| 小数值展开与矩阵结果一致 | 索引、乘法与聚合关系正确 | 高维 batch、多头与所有边界正确 |
| 权重沿候选轴求和为 1，禁止位置为 0 | 归一化轴与 mask 的位置正确；检查在 dropout 前 | 模型确实学到了有用语义 |
| 修改未来 K/V 不改变过去输出 | 局部可见性约束生效 | 整个模型所有连接都满足因果性 |
| 与参考算子前向、梯度一致 | 数值与反向实现没有明显偏差 | 真实训练一定收敛或任务表现改善 |

### 9.1 参考算子的数值对照

下面是一个可独立运行的参考对照，固定 CPU、float64、无 dropout、无缓存，使用方形 causal attention。手算路径明确构造 mask，参考路径由 PyTorch SDPA 处理因果约束。生产环境不同 kernel、精度和运算顺序可能需要不同容差。

```python
import torch
from torch.nn import functional as F

torch.manual_seed(11)
originals = [torch.randn(1, 1, 3, 2, dtype=torch.float64) for _ in range(3)]
q, k, v = [x.clone().requires_grad_() for x in originals]
rq, rk, rv = [x.clone().requires_grad_() for x in originals]
allowed = torch.ones(3, 3, dtype=torch.bool).tril()

scores = q @ k.transpose(-2, -1) / 2 ** 0.5
actual = scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ v
reference = F.scaled_dot_product_attention(
    rq, rk, rv, dropout_p=0.0, is_causal=True
)
torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)

upstream = torch.randn_like(actual)
actual_grads = torch.autograd.grad((actual * upstream).sum(), (q, k, v))
reference_grads = torch.autograd.grad(
    (reference * upstream).sum(), (rq, rk, rv)
)
for actual_grad, reference_grad in zip(actual_grads, reference_grads):
    torch.testing.assert_close(
        actual_grad, reference_grad, rtol=1e-10, atol=1e-10
    )
print("forward and Q/K/V gradients match")
```

这里用随机上游梯度，让不同输出元素产生不同学习信号，避免过于对称的目标掩盖问题。框架定义见 [PyTorch 2.14 SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。尤其要检查各接口对布尔 mask 的语义：不能把一个 API 中“True 表示允许”的约定直接套到所有 API。

## 10. 总结与后续

Attention 的完整计算关系可以概括为：输入形成多种投影，Q/K 决定允许位置之间的聚合比例， V 提供被汇总内容，输出投影把结果交回模型主路径，下游 loss 则沿这些依赖塑造参数。

解释模型行为时，需要结合完整的计算链。点积匹配的意义由训练塑造；某个头会随输入呈现不同模式，不能预设“一个头只能学一种语法关系”。注意力权重也不是模型判断原因的完整解释，因为内容向量、其他头、残差和后续非线性都会影响预测。相关讨论见 [Attention is not Explanation](https://arxiv.org/abs/1902.10186)。

这条机制缩短了允许位置之间的信息交互路径，但不保证长距离信息一定被可靠使用。生成时仍要逐 token 推进；训练序列已知时可以并行计算多个位置，并不意味着自回归生成也能一次得到所有未来 token。

本篇尚未引入位置表示。没有位置机制、没有因果等顺序相关约束的 self-attention 对输入置换是等变的：输入行换序，输出也相应换序，而不是输出矩阵完全不变。因果可见性会引入另一种结构约束，它与显式位置机制如何配合，留到位置篇推导。

多头注意力进一步让模型并行形成多组匹配与内容汇总，位置机制则使计算能够利用顺序和距离信息。接下来的文章沿这两个方向展开，阅读顺序见[专题导航](README.md)。
