# 自回归语言建模

> 自回归语言模型把一段文本的联合概率，分解为一系列“给定已有前文，预测下一个 token”的条件概率。训练时，完整序列已经存在，模型可以在因果约束下并行计算各位置的预测；生成时，未来 token 尚不存在，只能预测一个、追加一个，再继续预测。

## 结论

1. 自回归语言建模是一种概率建模方式，不是一种特定的神经网络结构。
2. 一个 token 序列的联合概率，可以通过概率链式法则拆成一系列 next-token 条件概率。
3. 模型在序列的每个位置输出整个词表上的 logits，这个位置的 logits 用来预测下一个 token。
4. 训练时拥有完整的真实序列，因而可以同时计算多个位置；causal mask 保证每个位置只能使用自己及之前的信息。
5. 生成时未来 token 尚不存在，因此必须根据当前前缀预测一个 token，将其追加到上下文后再继续。
6. 预训练和 SFT 都可以使用自回归目标，主要区别在于数据分布、消息结构以及哪些 token 被监督。
7. Decoder-only Transformer 是当前实现自回归语言模型的主流架构，但自回归建模本身并不依赖 Transformer。

## 问题

语言模型面对的不是一个固定类别集合，而是一个由 token 组成的序列。假设一句话经过 tokenizer 后得到：

```text
<BOS> 我 喜欢 数学 <EOS>
```

其中：

- `<BOS>` 表示序列开始；
- `<EOS>` 表示序列结束；
- “我”“喜欢”“数学”是示例 token；
- 实际模型是否显式添加 `<BOS>`，以及使用哪个结束 token，由 tokenizer 和 chat template 决定。

模型没有直接学习“看到这句话就把整句话背出来”，而是把它拆成连续的预测任务：

```text
给定 <BOS>                  -> 预测“我”
给定 <BOS> 我               -> 预测“喜欢”
给定 <BOS> 我 喜欢          -> 预测“数学”
给定 <BOS> 我 喜欢 数学     -> 预测 <EOS>
```

每一步的输出也不是一个已经确定的 token，而是整个词表上的分数或概率分布。例如，在看到 `<BOS> 我 喜欢` 后，模型可能给出：

```text
数学     0.60
编程     0.20
音乐     0.10
其他     0.10
```

“数学”只是这个分布中概率最高的候选。最终选择哪个 token，还取决于 greedy、sampling 等解码策略。解码方法属于推理阶段，本文只关注这个概率分布为什么能够构成完整的语言模型。

## 表示

神经网络不能直接处理自然语言字符串。文本首先经过 tokenizer，转换为 token，再映射为整数 token ID：

```text
文本
-> tokenizer
-> token 序列
-> token ID 序列
-> 模型输入
```

例如，假设词表中有以下映射：

```text
<BOS> -> 1
我    -> 10
喜欢  -> 20
数学  -> 30
<EOS> -> 2
```

那么示例序列可以表示为：

```python
input_ids = [1, 10, 20, 30, 2]
```

这里需要区分三件事：

1. **文本**是人类看到的字符串。
2. **token**是 tokenizer 定义的建模单位，不一定等于一个汉字或一个完整单词。
3. **token ID**是词表中对应 token 的整数编号，是模型真正接收的离散输入。

tokenizer 会影响序列长度、词表大小和模型需要预测的基本单位，但 BPE、Unigram 等 tokenizer 算法不属于本文范围。对于自回归建模来说，关键前提只是：<u>文本已经被表示为一个有顺序的离散 token 序列</u>。

## 分解

### 联合概率

对于 token 序列：

\[
x_1,x_2,\ldots,x_T
\]

模型希望描述整段序列出现的概率：

\[
P(x_1,x_2,\ldots,x_T)
\]

概率链式法则可以把这个联合概率分解为：

\[
P(x_1,x_2,\ldots,x_T)
=
\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1})
\]

也可以简写为：

\[
P(x_{1:T})
=
\prod_{t=1}^{T}P(x_t\mid x_{<t})
\]

其中：

- \(x_t\) 是当前需要预测的 token；
- \(x_{<t}\) 是当前 token 之前的全部 token；
- \(P(x_t\mid x_{<t})\) 是给定前文后，当前 token 出现的条件概率。

因此，示例序列的概率可以写成：

\[
\begin{aligned}
P(&\text{我, 喜欢, 数学, EOS}\mid\text{BOS})
= {} &P(\text{我}\mid\text{BOS}) \\
&\times P(\text{喜欢}\mid\text{BOS, 我}) \\
&\times P(\text{数学}\mid\text{BOS, 我, 喜欢}) \\
&\times P(\text{EOS}\mid\text{BOS, 我, 喜欢, 数学})
\end{aligned}
\]

### 数值示例

假设模型分配的条件概率为：

```text
P(我 | BOS)                       = 0.50
P(喜欢 | BOS, 我)                 = 0.40
P(数学 | BOS, 我, 喜欢)           = 0.80
P(EOS | BOS, 我, 喜欢, 数学)      = 0.50
```

那么整个序列的条件概率为：

```text
0.50 * 0.40 * 0.80 * 0.50 = 0.08
```

只要模型能够估计每一步的 next-token 条件概率，就能够计算整个序列的概率。训练语言模型因此可以转化为：不断提高真实下一个 token 的条件概率。

实际计算通常不会直接连乘概率，因为许多小于 1 的数相乘容易变成非常小的数。对数可以把乘法转换为加法：

\[
\log P(x_{1:T})
=
\sum_{t=1}^{T}\log P(x_t\mid x_{<t})
\]

训练时使用的负对数似然和交叉熵，正是从这里继续得到的。它们将在 `02_training_objectives/01_CROSS_ENTROPY_NLL_AND_PERPLEXITY.md` 中单独展开。

### 建模含义

自回归分解有三个重要含义：

1. **复杂序列被拆成局部预测。** 模型不必直接为所有可能的完整句子分别建立类别，而是反复解决 next-token prediction。
2. **每一步都依赖已有前缀。** 同一个 token 在不同上下文中的条件概率可以完全不同。
3. **局部错误会影响后续生成。** 生成阶段一旦选出不合适的 token，后面的预测会以这个 token 为新前缀继续进行。

链式分解来自概率论，而不是 Transformer 发明的。RNN、LSTM 和 Decoder-only Transformer 都可以用来参数化这些条件概率。

## 对齐

### 位置语义

假设完整输入是：

```python
input_ids = [1, 10, 20, 30, 2]
```

模型接收 batch 后，通常输出：

```text
logits.shape = [B, T, V]
```

其中：

- `B` 是 batch size；
- `T` 是当前 batch 的序列长度；
- `V` 是词表大小；
- `logits[b, t, :]` 是第 `b` 条样本在位置 `t` 上，对整个词表给出的未归一化分数。

在 causal language modeling 中，每个位置的 logits 用于预测它后面的 token：

| logits 位置 | 当前可见前缀 | 目标 token |
|---|---|---|
| 0 | `<BOS>` | `我` |
| 1 | `<BOS> 我` | `喜欢` |
| 2 | `<BOS> 我 喜欢` | `数学` |
| 3 | `<BOS> 我 喜欢 数学` | `<EOS>` |
| 4 | 完整序列 | 当前样本中没有后继目标 |

可以把这个关系画成：

```text
输入位置：   <BOS>      我       喜欢      数学      <EOS>
                |        |         |         |
预测目标：      我       喜欢      数学      <EOS>
```

因此，位置 `t` 的 logits 不是用来重新识别位置 `t` 已经输入的 token，而是用来预测位置 `t+1` 的 token。

### Causal Shift

如果 `logits` 和 `labels` 在序列维度上长度相同，就需要错开一位计算：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

对示例来说：

```text
参与计算的 logits 位置：0, 1, 2, 3
参与计算的 label 位置： 1, 2, 3, 4
```

它们形成以下配对：

```text
logits[0] -> label[1] = 我
logits[1] -> label[2] = 喜欢
logits[2] -> label[3] = 数学
logits[3] -> label[4] = EOS
```

最后一个位置的 logits 被丢弃，是因为当前序列没有提供它应该预测的下一个 token。第一个 label 不参与比较，是因为当前输入中没有更早的位置负责预测它。

不同训练框架可能在不同层次完成 shift：

- 自定义 loss 函数可以显式执行 shift；
- `AutoModelForCausalLM` 一类模型在收到 `labels` 时，也可能在内部完成 shift。

两种方式不能在同一条路径中重复执行，否则标签会错开两次。判断是否需要手动 shift，必须查看当前模型和 Trainer 的实际合同。

## 训练

### Teacher Forcing

训练数据已经包含完整的正确序列。因此，在预测“数学”时，模型使用的前缀是数据中的真实内容：

```text
<BOS> 我 喜欢
```

即使模型在前一个位置原本更倾向于生成“讨厌”，训练过程仍然使用真实 token“喜欢”作为下一位置的上下文。这种做法称为 teacher forcing。

Teacher forcing 带来两个直接好处：

1. 每个位置都能获得稳定、正确的真实前缀；
2. 一条完整序列中的多个预测位置可以在一次 forward 中计算。

它也带来训练和生成之间的差异：

- 训练时，前缀来自真实数据；
- 生成时，后续前缀包含模型自己刚刚生成的 token。

因此，生成阶段早期出现的错误可能改变后续条件分布，并继续影响后面的 token。仅仅降低 teacher-forced loss，并不能保证所有自由生成行为都会同步改善。

### Causal Mask

训练时，整个 token 序列虽然一次进入模型，但每个位置不能看到未来 token。以五个位置为例，允许访问的关系可以表示为：

```text
查询位置 \ 被访问位置   0    1    2    3    4
0                      yes
1                      yes  yes
2                      yes  yes  yes
3                      yes  yes  yes  yes
4                      yes  yes  yes  yes  yes
```

位置 2 可以使用位置 0、1、2 的信息，但不能使用位置 3、4。对应到示例中，模型在位置“喜欢”产生用于预测“数学”的 logits 时，不能提前读取“数学”和 `<EOS>`。

这类下三角可见性约束通常由 causal mask 实现。它解决的是模型内部信息流问题：

> 当前位置在计算 hidden state 和 logits 时，可以读取哪些位置？

causal mask 不等于 SFT 的 completion mask。completion mask 解决的是另一个问题：

> 哪些目标 token 应该参与 loss 并产生训练梯度？

前者限制注意力可见范围，后者限制监督范围。两者可能同时存在，但职责不同。

### 并行计算

自回归概率分解看起来是顺序的，但训练不必逐 token 调用模型。原因是：

1. 完整的真实序列已经存在；
2. 每个位置需要的真实前缀已经准备好；
3. causal mask 阻止每个位置读取未来信息；
4. 因而不同位置的 hidden state 和 logits 可以放在同一次张量计算中完成。

所以，训练中的“并行”不是取消了自回归条件，而是同时计算许多个受到不同前缀约束的条件概率：

```text
P(我 | BOS)
P(喜欢 | BOS, 我)
P(数学 | BOS, 我, 喜欢)
P(EOS | BOS, 我, 喜欢, 数学)
```

它们在数学上仍然是不同的条件概率，只是在工程上被组织成一次并行 forward。

## 生成

### 逐步解码

生成时只有 prompt，没有真实的未来 token。假设当前 prompt 是：

```text
<BOS> 我 喜欢
```

概念上的生成流程是：

```text
第 1 步
输入：<BOS> 我 喜欢
读取最后一个有效位置的 logits
选择：数学

第 2 步
输入：<BOS> 我 喜欢 数学
读取最后一个有效位置的 logits
选择：<EOS>

第 3 步
检测到 <EOS>
停止生成
```

可以概括为：

```text
已有前缀
-> forward
-> next-token logits
-> 选择一个 token
-> 追加到前缀
-> 重复
-> 遇到停止条件
```

因为第 2 步的输入取决于第 1 步实际选择了什么，所以时间维度上的生成必须逐 token 进行。多个样本仍然可以组成 batch 并行生成，但每条序列内部的后一个 token 依赖前一个生成结果。

成熟框架通常会用 KV Cache 避免在每一步重新计算全部历史 token，但这只改变计算效率，不改变自回归依赖关系。KV Cache 的结构和显存成本属于推理文档。

### 训练对照

| 维度 | 训练 | 生成 |
|---|---|---|
| 完整目标序列 | 已知 | 未知 |
| 前缀来源 | 真实数据 | prompt 与已生成 token |
| 时间维度计算 | 可并行多个位置 | 逐 token 推进 |
| 主要输出 | 各位置 logits 与 loss | 下一个 token |
| 参数更新 | 有 | 无 |
| 停止边界 | 样本长度 | EOS、长度上限或其他停止条件 |

训练和生成使用的是同一个条件概率模型，但调用方式不同。训练评估的是“给定真实前缀时能否预测真实下一个 token”，生成评估的是“模型沿着自己产生的前缀能否持续得到合适的完整输出”。

## 关系

### 自回归与 Transformer

自回归描述概率如何分解：

```text
完整序列概率
-> 一系列给定前文的 next-token 概率
```

Transformer 描述条件概率如何由神经网络计算：

```text
token IDs
-> embeddings
-> Transformer blocks
-> hidden states
-> LM head
-> logits
```

两者不在同一个抽象层次。RNN 和 LSTM 也能实现自回归语言模型；Decoder-only Transformer 只是当前更常见的实现。Transformer 内部的数据流将在 `02_DECODER_ONLY_TRANSFORMER.md` 中展开。

### 自回归与 Causal LM

Causal Language Modeling 通常指以下组合：

1. 根据左侧前缀预测后续 token；
2. 使用 causal mask 阻止未来信息泄漏；
3. 使用 next-token prediction 构造训练目标。

在当前大语言模型语境中，“自回归语言模型”和“causal LM”经常指向相近的模型，但前者更强调概率分解，后者更强调具体训练任务和因果可见性。

### 自回归与 Decoder-only

Decoder-only Transformer 通常为每个位置产生一个只能依赖过去与当前位置的 hidden state，再通过 LM head 输出 next-token logits。它非常适合实现 causal LM，但“Decoder-only”是架构分类，“自回归”仍然是建模与生成方式。

### 自回归与 SFT

预训练和 SFT 都可以继续使用 next-token prediction：

| 阶段 | 典型数据 | 常见监督范围 | 直接目标 |
|---|---|---|---|
| 预训练 | 大规模自然文本 | 大多数非 padding token | 学习文本分布与通用表征 |
| SFT | prompt 与示范回答 | 常见为 assistant completion token | 学习目标回答、格式与行为 |
| 推理 | 用户 prompt | 不计算训练 loss | 根据当前前缀生成输出 |

SFT 通常没有把模型改造成另一种概率模型。它仍然让模型预测下一个 token，只是：

- 使用了更有目标的数据分布；
- 可能只监督 assistant 回答；
- 通过训练让目标回答中的 token 在对应上下文下获得更高概率。

completion-only loss 如何构造 labels、使用 `-100` 以及统计有效 token，将在 `02_training_objectives/02_COMPLETION_ONLY_LOSS_AND_MASKING.md` 中单独说明。

### 自回归与 Masked LM

自回归 causal LM 根据前文预测后续 token。Masked Language Modeling 则通常遮住输入中的某些 token，再利用遮住位置两侧的信息恢复它们。

```text
自回归：我 喜欢 -> 预测“数学”
Masked LM：我 [MASK] 数学 -> 利用左右文预测“喜欢”
```

二者的数据构造、可见范围和生成方式不同。本文研究的是第一轮 SFT 所使用的自回归 causal LM。

## 映射

第一轮 Qwen3-0.6B OpenR1-Math SFT 中，自回归建模并不只存在于概念层，而是可以在代码和测试中直接定位。

### Loss 对齐

实验仓库中的：

```text
small_model_post_training/independent_implementation/
└── src/post_training_core/sft.py
```

在 `masked_sft_loss_sum_and_count()` 中显式执行：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

这对应本文的 next-token 对齐：位置 `t` 的 logits 与位置 `t+1` 的标签比较。函数随后使用 `ignore_index` 排除不需要监督的标签。后一个步骤属于 completion-only SFT，而 shift 本身来自 causal language modeling。

### 行为测试

对应测试位于：

```text
small_model_post_training/independent_implementation/
└── tests/test_sft_loss.py
```

其中：

- `test_applies_causal_shift()` 验证 logits 与下一位置标签正确配对；
- `test_ignores_masked_targets()` 验证监督范围之外的位置不影响 loss；
- `test_backward_only_updates_supervised_positions()` 验证只有有效监督位置产生梯度；
- `test_matches_pytorch_reference()` 验证自定义实现与 PyTorch 参考计算一致。

这些测试把三个层次分开验证：

```text
自回归位置对齐
-> SFT 监督位置筛选
-> loss 与梯度行为
```

### 生成调用

checkpoint 生成健康检查位于：

```text
small_model_post_training/independent_implementation/
└── scripts/verify_checkpoint_generation.py
```

脚本先通过 chat template 构造 prompt，再调用：

```python
output_ids = model.generate(
    input_ids=input_ids,
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    eos_token_id=eot_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
```

`model.generate()` 在框架内部封装了逐 token 的自回归解码。脚本中的参数分别固定：

- `do_sample=False`：使用确定性的非采样解码；
- `max_new_tokens`：限制最多生成多少新 token；
- `eos_token_id`：定义正常终止 token；
- `pad_token_id`：定义 batch 补齐 token。

生成后，脚本只截取 prompt 之后的新 token，并保存是否生成 EOS、生成 token 数量和停止原因。这使“模型会不会生成、能不能正常结束”成为可审计证据，而不是只在终端中观察一次输出。

## 边界

### 自回归不等于检索

模型不是从训练集中搜索一条完全相同的句子，再复制下一个词。它根据参数和当前上下文计算整个词表的条件分布。模型可能记忆训练内容，但自回归计算本身仍然是条件概率预测。

### 自回归不等于确定性

同一组 logits 可以通过不同解码方法得到不同 token。模型定义条件分布，解码器决定如何从分布中选择结果。

### 完整输入不等于未来泄漏

训练时可以把完整序列放入同一个张量。是否泄漏取决于每个位置的计算能否访问未来位置，而不是未来 token 是否物理存在于输入张量中。causal mask 正是用来限制这种访问。

### Causal Mask 不等于 Loss Mask

- causal mask 控制模型内部的信息可见范围；
- loss mask 控制哪些目标 token 参与训练损失。

一个位置可以作为上下文被后续 token 读取，但它自己的预测目标仍然被 loss mask 忽略。例如 SFT 中的用户 prompt 通常参与构造上下文，却不一定作为监督目标。

### Teacher-forced Loss 不等于生成质量

验证集 NLL 下降说明模型在真实前缀下更会预测目标 token，但不能单独证明：

- 自由生成一定更连贯；
- 数学答案一定更准确；
- 推理过程一定更合理；
- EOS 终止一定更稳定；
- 通用能力没有回退。

这些结论需要生成评测、任务指标和逐样本分析共同支持。

### 上下文不等于长期记忆

模型可以条件化于当前上下文窗口中的 token，但这不意味着它天然拥有跨会话的永久记忆。上下文长度、参数记忆和外部检索是不同机制。

### 原始序列概率不宜跨长度直接比较

序列概率是多个小于等于 1 的条件概率之积。序列越长，原始乘积通常越小。因此，对不同长度序列做比较时，常常需要使用平均 token NLL、长度归一化分数或任务特定指标，而不是直接比较概率乘积。

## 检查

1. 对序列 `[BOS, A, B, EOS]`，模型需要完成哪三个 next-token 预测？
2. 为什么位置 `A` 的 logits 应该与标签 `B` 比较？
3. 为什么当前序列最后一个位置的 logits 通常不参与 loss？
4. 训练时完整序列已经进入模型，为什么这不必然构成标签泄漏？
5. causal mask 与 completion loss mask 分别控制什么？
6. 为什么训练可以并行计算多个位置，而生成必须逐 token 推进？
7. teacher forcing 中的前缀来自哪里？它和生成时的前缀有什么不同？
8. 自回归语言建模、Causal LM 和 Decoder-only Transformer 分别属于什么层次？
9. 预训练和 completion-only SFT 都可以预测下一个 token，它们的主要差别是什么？
10. 为什么 `<EOS>` 也是一个需要监督或评测的 token？
11. 验证集 next-token NLL 下降，能够直接证明哪些事情，又不能直接证明哪些事情？
12. 在第一轮 SFT 代码中，哪两行实现了 causal shift？哪个测试证明它没有错位？

能够不依赖本文回答这些问题，并把概率公式、token 对齐和代码实现对应起来，才算真正掌握了自回归语言建模的基础。
