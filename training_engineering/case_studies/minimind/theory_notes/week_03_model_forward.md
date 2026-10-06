# 第 3 周：模型主干复盘

## 1. 本周学习目标

第 3 周开始进入 `model/model_minimind.py`。

你已经学过 LLM forward 主线，所以这一周不从零讲 Transformer，而是把已有理解落实到 MiniMind 源码：

- 模型由哪些类组成。
- `input_ids` 如何变成 `hidden_states`。
- `hidden_states` 如何变成 `logits`。
- loss 在哪里计算。
- 每一步 tensor shape 如何变化。

本周结束后，你应该能独立写出：

```text
input_ids -> embedding -> blocks -> norm -> lm_head -> logits -> shift loss
```

并能指出它们分别在源码哪个类里。

## 2. MiniMind 模型类结构

源码位置：

```text
/Users/yanpeng/Documents/PythonProjects/datawhale/minimind/model/model_minimind.py
```

核心类如下：

| 类名 | 作用 |
|---|---|
| `MiniMindConfig` | 保存模型超参数 |
| `RMSNorm` | 归一化层 |
| `Attention` | 自注意力层 |
| `FeedForward` | FFN / MLP 层 |
| `MOEFeedForward` | MoE 版本 FFN |
| `MiniMindBlock` | 一个 Transformer block |
| `MiniMindModel` | embedding + 多层 block + final norm |
| `MiniMindForCausalLM` | 包装语言模型头、loss、generate |

最重要的层级关系：

```text
MiniMindForCausalLM
  ├── MiniMindModel
  │     ├── embed_tokens
  │     ├── MiniMindBlock x num_hidden_layers
  │     │     ├── RMSNorm
  │     │     ├── Attention
  │     │     ├── RMSNorm
  │     │     └── FeedForward / MOEFeedForward
  │     └── final RMSNorm
  └── lm_head
```

## 3. Config 先看哪些参数

`MiniMindConfig` 里第 3 周重点看这些：

```python
hidden_size = 768
num_hidden_layers = 8
vocab_size = 6400
num_attention_heads = 8
num_key_value_heads = 4
head_dim = hidden_size // num_attention_heads
intermediate_size = ...
max_position_embeddings = 32768
tie_word_embeddings = True
```

它们对应的直觉：

| 参数 | 含义 |
|---|---|
| `vocab_size` | 词表大小，决定 logits 最后一维 |
| `hidden_size` | token embedding / hidden state 的宽度 |
| `num_hidden_layers` | Transformer block 层数 |
| `num_attention_heads` | query heads 数 |
| `num_key_value_heads` | key/value heads 数，MiniMind 使用 GQA |
| `head_dim` | 每个 attention head 的维度 |
| `intermediate_size` | FFN 中间层宽度 |
| `tie_word_embeddings` | 是否共享 embedding 和 lm_head 权重 |

## 4. 整体 forward 主线

### 4.1 MiniMindForCausalLM.forward

最外层 forward：

```python
hidden_states, past_key_values, aux_loss = self.model(
    input_ids,
    attention_mask,
    past_key_values,
    use_cache,
    **kwargs
)
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

也就是：

```text
input_ids
-> MiniMindModel
-> hidden_states
-> lm_head
-> logits
```

如果传入 `labels`，再计算 loss：

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

### 4.2 MiniMindModel.forward

主体模型 forward：

```python
batch_size, seq_length = input_ids.shape
hidden_states = self.dropout(self.embed_tokens(input_ids))
position_embeddings = (...)
for layer, past_key_value in zip(self.layers, past_key_values):
    hidden_states, present = layer(...)
hidden_states = self.norm(hidden_states)
return hidden_states, presents, aux_loss
```

也就是：

```text
input_ids
-> embedding
-> position embeddings / RoPE
-> 多层 MiniMindBlock
-> final norm
-> hidden_states
```

## 5. Forward Shape 表

设：

```text
B = batch_size
T = seq_length
C = hidden_size
V = vocab_size
L = num_hidden_layers
```

MiniMind 默认可以先记：

```text
C = 768
V = 6400
L = 8
```

shape 变化：

| 步骤 | shape | 说明 |
|---|---|---|
| `input_ids` | `[B, T]` | token id |
| `embed_tokens(input_ids)` | `[B, T, C]` | 查 embedding table |
| `dropout` | `[B, T, C]` | shape 不变 |
| `MiniMindBlock x L` | `[B, T, C]` | 每层 shape 不变 |
| `final norm` | `[B, T, C]` | shape 不变 |
| `lm_head` | `[B, T, V]` | 映射到词表空间 |
| `logits[..., :-1, :]` | `[B, T-1, V]` | 预测位置 |
| `labels[..., 1:]` | `[B, T-1]` | 目标 token |
| `cross_entropy` 展平输入 | `[B*(T-1), V]` | 多个 token 分类任务 |
| `cross_entropy` 展平目标 | `[B*(T-1)]` | 每个位置一个目标 id |

一句话版本：

```text
MiniMind 的 forward 主干就是把 [B,T] 的 token id 变成 [B,T,V] 的词表 logits。
```

## 6. MiniMindBlock 内部结构

源码：

```python
residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    position_embeddings,
    past_key_value,
    use_cache,
    attention_mask
)
hidden_states += residual
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
return hidden_states, present_key_value
```

可以翻译成：

```text
输入 hidden_states
-> RMSNorm
-> Attention
-> 残差相加
-> RMSNorm
-> FFN / MoE FFN
-> 残差相加
-> 输出 hidden_states
```

这是典型 pre-norm Transformer block：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

shape 始终保持：

```text
[B,T,C] -> [B,T,C]
```

## 7. lm_head 和 embedding 权重共享

MiniMind 里：

```python
self.model.embed_tokens.weight = self.lm_head.weight
```

前提是：

```python
tie_word_embeddings = True
```

含义是：

```text
输入时 token id 通过 embedding table 查向量。
输出时 hidden state 通过 lm_head 投影回词表。
二者可以共享同一张权重表。
```

直觉上：

```text
同一套 token 语义表，既负责把 token 读进来，也负责把 hidden state 翻译回 token 概率。
```

## 8. 本周 Checklist

- [ ] 能说出 `MiniMindForCausalLM` 和 `MiniMindModel` 的区别。
- [ ] 能解释 `embed_tokens` 的输入输出 shape。
- [ ] 能解释 `MiniMindBlock` 的 pre-norm + residual 结构。
- [ ] 能写出 `[B,T] -> [B,T,C] -> [B,T,V]`。
- [ ] 能解释 `lm_head` 的作用。
- [ ] 能解释 loss 的 shift 位置。
- [ ] 能解释 `tie_word_embeddings` 的基本含义。

## 9. 当前进度

状态：进行中

已完成：

- [x] 第 1-2 周输入链路。
- [x] 已有 LLM forward 主线基础。
- [x] 定位 MiniMind 模型主类。

待完成：

- [ ] 逐行复述 `MiniMindModel.forward`。
- [ ] 逐行复述 `MiniMindForCausalLM.forward`。
- [ ] 写出完整 forward shape 表。
- [ ] 进入第 4 周前，单独深挖 Attention、RoPE、KV Cache。

