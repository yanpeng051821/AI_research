# 自回归语言建模

> 自回归规定如何把序列概率分解成连续预测；神经网络负责计算这些预测。
> 从这里出发，可以把任务、架构、训练目标与参数更新放进同一套框架。

[系列导航](README.md) · 下一篇：[向量、矩阵与线性层](02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md)

## 任务

假设我们希望模型接着“我喜欢”继续说话。为每一句可能的完整回答建立一个独立类别并不可行：
序列长度会变化，组合数量又极大。但若把问题拆成“给定前文，预测下一个 token”，
每次预测就面对固定大小的词表，连续使用这些预测又能组成可变长度的回答。

因此语言生成与分类并非完全无关。**next-token 预测可以看成依赖上下文的 V 类分类，
完整生成则是这些条件分布的连续使用。** 单步类别固定，不等于完整回答固定。

以下是教学分词，不代表实际 tokenizer 一定如此切分：

```text
<BOS> 我 喜欢 数学 <EOS>
```

模型看到 `<BOS> 我 喜欢` 后，可能给“数学”0.60、“编程”0.20、“音乐”0.10，
其余 token 合计0.10。这是分布，不是已经选好的答案。取最大值还是采样，属于解码策略。
EOS 也要获得概率，因为“什么时候结束”同样是生成行为。

## 系统

将一次监督学习写为：

\[
\hat y=f_\theta(x),\qquad L=\ell(\hat y,y)
\]

x 是输入，θ 是可训练参数，f 的组织方式是架构，ŷ 是输出，y 是目标标签。
loss 衡量当前输出在训练标准下的表现；反向传播计算它对参数的局部变化率，
优化器才据此更新参数。

以情感分类为例：任务是判断情感，标签可能是正面或负面，模型输出两个分数，
交叉熵是可用的 loss，准确率是评测指标。CNN、RNN 或 Transformer 都能构建这样的模型，
但组织信息的方式、成本和适用条件不同。

换成 next-token 预测时，输出通常变成每个位置的 V 个分数，标签变成后续 token，仍可用交叉熵。
**同一种 loss 不对应唯一架构，同一种架构也不对应唯一任务。**
它们需要在输入表示、可见信息、输出含义和监督方式上匹配，而不是名称上相同。

“训练目标”也不等同于“标签”。标签是样本提供的监督内容；
数学目标可能是最小化数据分布上的平均 NLL；现实目标可能是提高工具调用可靠性。
三者相关，但训练目标改善不自动代表现实目标达成。

## 表示

tokenizer 将文本转换成离散编号，例如：

```text
<BOS> -> 1， 我 -> 10， 喜欢 -> 20， 数学 -> 30， <EOS> -> 2
```

输入成为 `[1,10,20,30,2]`。编号只是索引，30 不比10“语义多三倍”。
Embedding 将它们映射为可训练向量，后续网络处理这些向量。
tokenizer 影响建模单位和序列长度，但其分词规则不是模型内部的上下文理解。

一个常见 decoder-only 模型的前向路径为：

```text
input_ids [B,T]
    -> Embedding [B,T,D]
    -> 多层因果上下文计算与特征加工 [B,T,D]
    -> 最终归一化 [B,T,D]
    -> LM head [B,T,V]
    -> logits
```

logits 是实数分数，还不是概率。对最后的词表维应用 softmax：

\[
p_{t,j}=\frac{\exp(z_{t,j})}{\sum_{k=1}^{V}\exp(z_{t,k})}
\]

得到位置 t 对下一个 token 的分布。一个分数的作用取决于它与其他分数的相对关系；
给全部 logits 加同一个常数不会改变分布。
内部组件将在后续拆开，此时先确定输入和输出的意义。

## 分解

### 概率链

对于序列 \(x_1,\ldots,x_T\)，概率链式法则给出：

\[
P(x_{1:T})=\prod_{t=1}^{T}P(x_t\mid x_{<t})
\]

链式法则是概率恒等式；建模工作是用带参数的网络 \(P_\theta\) 近似这些条件分布。
“自回归”不是 Transformer 的别名。RNN 也能用前文预测后文，
Transformer 也可以使用双向信息与不同目标。

假设固定 BOS 后，“我、喜欢、数学、EOS”分别得到0.5、0.4、0.8、0.5：

\[
P_\theta(\text{我,喜欢,数学,EOS}\mid\text{BOS})=0.5\times0.4\times0.8\times0.5=0.08
\]

长序列概率连乘容易变得极小，所以通常使用对数：

\[
-\log P_\theta(x_{1:T})=\sum_t-\log P_\theta(x_t\mid x_{<t})
\]

右边是各目标 token 的负对数似然之和。以上四项之和约2.5257，平均约0.6314。
它是在真实前缀下的预测误差，不是“整段生成正确率”。

### 交叉熵

若正确目标是编号 y，目标分布 q 为 one-hot，交叉熵化为：

\[
\ell=-\sum_jq_j\log p_j=-\log p_y
\]

此时交叉熵与目标 token 的 NLL 是同一个量，不是两套先后计算的损失。
正确目标概率0.8对应约0.2231，概率0.1对应约2.3026；概率越低，惩罚越大。

对 logits 的导数是 \(p_j-\mathbf{1}[j=y]\)，不仅影响正确类别，也影响其他类别分数。
这个结果可以从同一公式看出来：

\[
\ell=-z_y+\log\sum_k\exp(z_k)
\]

第一项对正确类别求导为-1，对其余类别为0；
第二项对 z_j 求导为 \(\exp(z_j)/\sum_k\exp(z_k)=p_j\)，两者相加便得到上述结果。
因此“提高正确答案的概率”并不是只更新一个孤立分数，而是调整整个竞争分布。

这些梯度再沿 LM head 和骨干传播。
框架交叉熵通常直接接收 logits，并稳定地计算 log-softmax；不要先 softmax 再把概率当 logits 传入。
[PyTorch 交叉熵接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

### 聚合

若 m 表示是否监督，常见的 token 平均为：

\[
L=\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}{\sum_{b,t}m_{b,t}}
\]

有效 token 的系数相同，不意味着梯度数值相同，后者还取决于预测和整条计算路径。
先对每条样本平均，再对样本平均，会赋予不同的 token 权重，这是目标定义差异。
全 mask 时分母为零，应按合同报错或显式处理，而不是把 NaN 当成正常结果。

相同 tokenizer、数据、mask 和聚合方式下，困惑度为 \(\exp(L)\)。
它是概率预测指标，不是推理或生成质量的通用分数，也不适合直接跨 tokenizer 比较。

## 对齐

位置 t 已经读到 \(x_t\)，该位置的 logits 应预测 \(x_{t+1}\)：

| 位置 | 输入 | logits 对应目标 |
| --- | --- | --- |
| 0 | BOS | 我 |
| 1 | 我 | 喜欢 |
| 2 | 喜欢 | 数学 |
| 3 | 数学 | EOS |
| 4 | EOS | 本条样本没有下一个目标 |

若 labels 最初与 input_ids 同位置存储，取 logits 前 T-1 项与 labels 后 T-1 项对齐。
第一个 label 被去掉，因为没有它的前置预测位置；最后一项 logits 被去掉，因为没有对应目标。
**EOS label 没有被去掉，它由前一位置预测。**

以下实验用40个词表项容纳教学编号，不代表实际词表大小：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
ids = torch.tensor([[1, 10, 20, 30, 2]])
labels = ids.clone()
labels[:, :3] = -100
logits = torch.randn(1, 5, 40, requires_grad=True)
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
assert shift_labels.tolist() == [[-100, -100, 30, 2]]
losses = F.cross_entropy(
    shift_logits.reshape(-1, 40),
    shift_labels.reshape(-1),
    reduction="none",
    ignore_index=-100,
)
valid = shift_labels.reshape(-1) != -100
loss = losses[valid].mean()
loss.backward()
assert valid.sum().item() == 2
torch.testing.assert_close(logits.grad[:, :2], torch.zeros(1, 2, 40))
torch.testing.assert_close(logits.grad[:, -1], torch.zeros(1, 40))
assert logits.grad[:, 2:4].abs().sum() > 0
```

本例证明输出分数哪些位置直接收到 loss 梯度，**不证明 prompt 表示没有梯度**：
回答位置可以通过 attention 使用 prompt，回答 loss 仍沿依赖传回 prompt 的中间表示。

Dataset 也可以直接返回错位输入 `[BOS,我,喜欢,数学]` 和目标 `[我,喜欢,数学,EOS]`。
两种接口都合理，但只能对齐一次。先查清 Dataset、model、loss 谁负责 shift，不能机械重复。

## 可见性

### 因果约束

完整句子已存在，为什么训练不能偷看答案？因为每层跨位置计算都要限制可见范围。
因果 attention 中，位置 t 看自己及之前，不能看之后：

```text
          key 位置
          0 1 2 3
query 0   1 0 0 0
query 1   1 1 0 0
query 2   1 1 1 0
query 3   1 1 1 1
```

“看自己”不泄漏 next-token 答案，因为位置 t 的目标在 t+1。
如果错对成同位置目标，模型就可能学会复述输入，而非预测后续。
矩阵运算同时处理所有行，不代表每行都能使用所有列。

### 三种 mask

因果 mask 限制未来信息；padding attention mask 限制补齐位置作为有效内容参与注意力；
loss mask 控制哪些预测进入目标函数。三者作用于不同环节。

SFT 的 user 文字通常是有效上下文，attention 仍应读取；但在 assistant-only supervision 中，
它不作为直接监督目标，对应 labels 设为 -100。这个 -100 是 loss 忽略标记，不是 Embedding 的词表编号。

动态 padding 到当前 batch 最长长度，是组织张量的办法，与生成允许新增多少 token 不是同一参数。
“padding 不进入 loss”也不能自动保证 attention 不读取 padding，需分别检查。

## 训练与生成

teacher forcing 使用真实前缀。预测“数学”时，前缀是原文“我喜欢”，
不是模型在前两个位置自己生成的内容，所以一条序列上的条件预测可以并行计算。

常规自回归生成则先选出一个 token，追加后再计算下一次分布。
前一次选错，后续输入也随之变化。NLL 改善不保证自由生成改善，因为两者经历的前缀不同。

生成通常读取最后一个有效位置的 logits。KV cache 缓存历史 attention 的键和值，
避免重复计算历史部分；它不提前知道未来，也不把普通逐 token 生成变成同时确定整段回答。
具体缓存、采样和停止规则属于推理实现。

推理通常不需要记录梯度图。model.eval() 与关闭梯度又是两回事：
前者影响 dropout 等模块模式，后者控制 autograd，详见[第 08 篇](08_AUTOGRAD_AND_PARAMETER_UPDATES.md)。

## 阶段与架构

预训练常用大规模文本的 next-token 目标；SFT 使用示范或对话，并选择监督区域。
二者可以沿用同一骨干、词表头和交叉熵，同时改变数据分布、格式和 mask。
“进入 SFT”通常不要求重新设计 Transformer。

偏好训练可继续用同一语言模型产生序列 log-prob，再用 chosen/rejected 构造另一目标。
强化学习的奖励可能不可微，需要构造合适的梯度估计或代理目标，而不是直接对任意评分器反向传播。
改变学习信号不意味着必须更换架构，也不意味着所有训练问题都是逐 token 分类。

更换任务时，需要检查输入、输出和可见信息：分类可能增加分类头；语音需要声学输入表示；
图像可用 patch 表示或卷积；多模态还需处理表示对齐。
“神经网络是参数化函数”提供了共同视角，但不能抹平不同任务的结构与成本约束。

## 检验

| 检验 | 防止的错误 | 不能直接证明 |
| --- | --- | --- |
| 小序列逐位置对齐 | 同位置复述、重复 shift、漏掉 EOS | 具有语言能力 |
| 修改未来 token，核对此前 logits | 因果泄漏；测试需固定随机性 | 所有 mask 场景都正确 |
| 核对忽略位置和 loss | loss mask 错误 | prompt 不参与上下文 |
| 核对 loss_sum 和有效 token 数 | 平均分母不一致 | 权重适合所有任务 |
| 对照 NLL 与自由生成 | 把拟合误当成任务效果 | 已确定改善原因 |

Dataset、model 和 loss 可能在正确计算同一个“错误对齐的问题”。
所以手算目标位置、验证信息访问和参考数值对照需要互补。

## 复习

**为什么架构不是目标？** 架构规定函数族和信息通路，目标决定训练偏好哪种行为。
同一 decoder 可以在不同数据上做预训练与 SFT。

**为什么并行训练却逐步生成？** 训练已有真实序列，可以在因果约束下并行计算；
普通生成中后续输入尚未产生，存在依赖顺序。

**NLL 下降说明什么？** 固定数据、tokenizer、mask 和聚合方式后，
被监督目标的平均负对数概率降低。不能直接推出事实正确性、推理成功率或终止行为改善。

接下来进入[第 02 篇](02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md)，把“处理向量”落实到矩阵计算。
