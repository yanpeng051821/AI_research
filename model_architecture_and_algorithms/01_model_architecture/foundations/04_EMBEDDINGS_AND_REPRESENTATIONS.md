# Embedding 与表示学习

> Embedding 把离散编号转换为可训练向量，是 token 进入连续计算的接口。
> 它学到的是初始表示；上下文表示由后面的信息交流与特征加工共同形成。

[系列导航](README.md) · 前篇：[激活](03_LINEARITY_NONLINEARITY_AND_ACTIVATIONS.md) · 后篇：[归一化](05_NORMALIZATION_AND_TRAINING_SCALE.md)

## 接口

tokenizer 给每个 token 一个整数编号，但编号的大小没有自然语义。
如果直接把“编号10”和“编号30”当普通数值输入，线性层就会把后者当成前者的三倍；
这种数值关系是词表排列带来的，不是语言本身要求的。

我们需要给每种离散符号一组可学习的连续特征。
Embedding 因而维护一个参数矩阵：

\[
E\in\mathbb{R}^{V\times D}
\]

V 是词表大小，D 是每个 token 的表示宽度。ID=i 时，输出第 i 行：

\[
h=E_{i,:}
\]

一条长为 T 的序列得到 [T,D]；一批 [B,T] 编号得到 [B,T,D]。
这里没有把不同位置的表示混合，也没有读取相邻 token。
[PyTorch Embedding 的查表定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

## 查表与线性

查表可以写成 one-hot 与矩阵的乘法。
令 e_i 是长度 V 的向量，仅第 i 项为1，其余为0，则：

\[
h=e_i^TE
\]

展开后，只有 E 的第 i 行被保留。这解释了 Embedding 与线性运算的联系：
它不是神秘的“文字理解器”，而是对离散选择的一种高效参数化。

但实际不必构造巨大的 one-hot。直接索引避免创建大部分元素为零的张量。
两者在这个普通查表操作上数值等价，不代表所有 Embedding 的特殊配置都等同于一个无条件 Linear。

```python
import torch
import torch.nn.functional as F

table = torch.tensor([[1., 0.], [0., 2.], [3., 4.]])
ids = torch.tensor([[2, 0, 2]])
lookup = table[ids]
one_hot = F.one_hot(ids, num_classes=3).to(table.dtype)
torch.testing.assert_close(lookup, one_hot @ table)
assert lookup.shape == (1, 3, 2)
torch.testing.assert_close(lookup[0, 0], lookup[0, 2])
```

两个位置的 ID 相同，便使用同一行参数。这不是为每次出现复制一套可训练参数：
前向可以产生多个输出位置，反向的贡献却汇入同一行 E。

## 学习

为什么“只查表”仍能训练？因为表里的浮点数是参数，最终 loss 依赖查出的数值。
如果改变 E 的某一行会改变预测，反向传播就能计算该行如何影响 loss。

设位置 t 使用 ID i_t，查表输出为 h_t，上游传回梯度 g_t。
普通查表对表中第 i 行的梯度为：

\[
\frac{\partial L}{\partial E_{i,:}}
=\sum_{t:i_t=i}\frac{\partial L}{\partial h_t}
\]

相同 ID 的不同出现次数共用参数，所以梯度相加。
但它不是永远等于出现频次：每次的 g_t 可能不同，也可能正负抵消。

此前的观察实验令 loss 为全部输出元素之和，此时每个 g_t 都是全1，
才得到“查两次的行梯度为2、一次为1”的结果：

```python
import torch
from torch import nn

embedding = nn.Embedding(6, 3)
ids = torch.tensor([[1, 2, 1], [3, 0, 2]])
embedding(ids).sum().backward()
expected = torch.tensor([
    [1., 1., 1.],
    [2., 2., 2.],
    [2., 2., 2.],
    [1., 1., 1.],
    [0., 0., 0.],
    [0., 0., 0.],
])
torch.testing.assert_close(embedding.weight.grad, expected)
assert not ids.requires_grad
```

原始整数 ID 不需要也不能像浮点参数那样对其求普通连续梯度。
需要更新的是被选中的向量，而不是把 ID 从2更新成2.03。
因此“模型输入不求梯度”与“Embedding 参数能学习”完全相容。

## 表示层次

### 初始表示

同一张表、同一 ID，直接查表得到相同向量。
刚初始化时它通常是随机的；经过训练后，它成为有助于后续预测的参数。
不能仅凭随机向量的几个数，解释它表示了什么。

多个 token 的使用环境相近，训练可能推动它们在某些方向上形成相近的表示，
但相似性不是查表操作自带的保证，也不意味着每个维度都能命名。
一个 token 的含义可以分散在多个特征中，同一特征也可能参与多个语义现象。

### 上下文表示

考虑两次出现的“苹果”：一次在“我吃了苹果”，一次在“苹果发布了产品”。
为便于说明，假设它们对应同一个 token ID。初始查表向量相同，
但前文不同，因果 attention 汇入的信息不同，后续 FFN 加工的输入便可能不同。

因此：

```text
相同 token ID
    -> 相同的初始查表向量
    -> 不同上下文和位置信息参与后续计算
    -> 可能不同的隐藏表示
    -> 不同的后续 token 分布
```

不应说“Embedding 已经知道这里是水果还是公司”，也不应反过来说“查表固定，所以模型不能区分含义”。
区分来自整条网络，而不是单个模块包办。

### 输出表示

最后的隐藏状态 [B,T,D] 要转成 [B,T,V] 的词表分数：

\[
z_{t,j}=w_j^Th_t+b_j
\]

LM head 的每一行相当于对候选 token 的一个打分方向。
它与输入 Embedding 的职责不同：前者从隐藏表示得到预测分数，后者把已知编号变成输入向量。

有些模型让二者共享同一个 [V,D] 参数矩阵，称为权重绑定。
本地参考 MiniMind 配置默认启用这种绑定；应以实际配置为准。
共享参数节省存储，也让同一矩阵参与输入和输出两种计算，但它不是两个互逆操作。

这还有梯度上的重要后果：**“未查到的行没有梯度”只适用于独立的输入查表路径。**
如果与 LM head 共享参数，某个 token 即使没有作为输入出现，其行也可能因词表分类的输出路径收到梯度。
optimizer 的权重衰减或历史状态又是参数更新的另一层因素，不能简单根据本批查表次数断言参数是否变化。

## 位置与补齐

词嵌入本身不区分“第一个位置”和“第五个位置”。
位置关系由模型另行处理：有的架构直接加入位置向量，有的在 attention 的 Q、K 上应用 RoPE。
本地 MiniMind 使用后一种方式，所以不要在概念图里默认它一定把位置 embedding 加到词 embedding 上。

padding 也不是仅靠一个编号就自动消失。
nn.Embedding 的 padding_idx 可让指定行不从查表路径接收梯度，
但 attention mask 和 loss mask 仍分别决定补齐位置是否参与上下文和监督。
当前手写 TokenEmbedding 是直接索引，没有实现 nn.Embedding 全部可选行为；
理解和测试应限于已经约定的功能范围。

## 检验

查表前向测试使用非重复的行值，检查 ID 是否取到正确行，同时验证 [B,T,D] 形状。
重复 ID 用于验证共享参数，未出现 ID 用于验证查表路径的零贡献。
这些设计分别保护不同性质，不是为了增加测试数量。

参数必须与参考实现统一；否则两套随机表的输出本来就不同。
复制参数放在 no_grad 中，是为了把它当成测试准备，不让“配置参考值”进入求导过程。
反向比较使用独立参数对象，防止两条路径把梯度累加到同一个对象上。

前向正确不证明梯度正确；梯度正确也不证明表示已经学出有用语义。
后者需要训练数据、目标与评测共同提供证据。
已有实践见[Embedding 测试](../../../training_engineering/case_studies/minimind/implementation/tests/test_embedding.py)。

## 复习

**Embedding 是常量字典吗？** 索引规则固定，但表里的向量通常是可训练参数。

**同一个 token 每次都是相同向量吗？** 单看同一输入表的查表结果是；加入位置、上下文和后续计算后不一定是。

**为什么频次高时梯度不一定大？** 梯度是各次使用的学习信号之和，不是只数次数。
sum 实验把每次信号都设成1，才得到频次等于梯度的特殊结果。

有了初始表示，后面的层开始反复加工它。下一篇解释这种加工为什么需要关注[数值尺度](05_NORMALIZATION_AND_TRAINING_SCALE.md)。
