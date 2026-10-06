# 第 2 周：Tokenizer 与数据样本

## 1. 本周学习目标

第 2 周从“模型怎么训练”往前走一步，重点看数据如何变成模型能吃的样子。

本周不急着研究 Transformer 细节，目标是搞清楚：

- tokenizer 为什么存在。
- 文本如何变成 `input_ids`。
- `PretrainDataset` 如何构造 `labels`。
- `SFTDataset` 为什么只训练 assistant 的回答。
- `-100` 为什么能屏蔽某些位置的 loss。

本周结束后，你应该能用自己的话讲清楚：

```text
同样是 input_ids 和 labels，pretrain 让模型学习续写所有文本，SFT 只让模型学习 assistant 应该如何回答。
```

## 2. 从 CV 数据集迁移到 LLM 数据集

你熟悉的 CV 图像分类大致是：

```text
图片文件
-> transforms
-> image tensor
-> label
-> model
-> cross entropy
```

LLM 的 pretrain 数据集大致是：

```text
文本
-> tokenizer
-> input_ids
-> labels
-> model
-> next token cross entropy
```

最大的变化是：

```text
CV 的 label 通常是一个类别 id。
LLM 的 labels 通常是一整串 token id。
```

## 3. PretrainDataset 主线

源码位置：

```text
/Users/yanpeng/Documents/PythonProjects/datawhale/minimind/dataset/lm_dataset.py
```

核心代码：

```python
sample = self.samples[index]
tokens = self.tokenizer(
    str(sample['text']),
    add_special_tokens=False,
    max_length=self.max_length - 2,
    truncation=True
).input_ids
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
input_ids = torch.tensor(input_ids, dtype=torch.long)
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
return input_ids, labels
```

### 3.1 每一行在做什么

```text
sample = self.samples[index]
```

从 jsonl 数据中取出一条文本样本。

```text
tokenizer(...).input_ids
```

把字符串切分并转换成 token id。

```text
max_length=self.max_length - 2
```

给 `bos` 和 `eos` 留两个位置。

```text
truncation=True
```

如果文本太长，就截断。

```text
tokens = [bos] + tokens + [eos]
```

告诉模型一句话从哪里开始、到哪里结束。

```text
pad 到 max_length
```

保证 batch 里的所有样本长度一致。

```text
labels = input_ids.clone()
```

pretrain 阶段默认每个非 padding token 都参与 next token prediction。

```text
labels[input_ids == pad_token_id] = -100
```

padding 只是为了凑长度，不应该参与 loss。

## 4. Pretrain 的训练目标

假设原文本是：

```text
我喜欢学习AI
```

经过 tokenizer 后可以抽象成：

```text
[BOS, 我, 喜欢, 学习, AI, EOS, PAD, PAD]
```

`input_ids` 是：

```text
[BOS, 我, 喜欢, 学习, AI, EOS, PAD, PAD]
```

`labels` 是：

```text
[BOS, 我, 喜欢, 学习, AI, EOS, -100, -100]
```

模型 forward 里再 shift：

```text
logits[:-1] 对齐 labels[1:]
```

真正训练的是：

```text
BOS -> 我
我 -> 喜欢
喜欢 -> 学习
学习 -> AI
AI -> EOS
EOS -> -100，不计入 loss
PAD -> -100，不计入 loss
```

## 5. SFTDataset 主线

Pretrain 的样本是普通文本，SFT 的样本是对话。

SFT 的目标不是“学会续写所有文字”，而是：

```text
看到 system/user/history 后，只学习 assistant 应该输出什么。
```

所以 SFT 需要做一件 pretrain 不需要做的事情：

```text
把 user/system 部分的 labels 设为 -100，只保留 assistant 部分参与 loss。
```

### 5.1 为什么不是整段对话都参与 loss

一条 SFT 对话通常包含多个角色：

```text
system: 你是一个有帮助的助手
user: 你好
assistant: 你好，有什么我可以帮你？
```

这些内容都会作为 `input_ids` 输入给模型，但不代表它们都应该参与 loss。

关键区别是：

```text
输入给模型看：作为上下文和条件。
参与 loss：要求模型模仿和学习生成。
```

如果 user 部分也参与 loss，模型会同时学习两件互相混杂的事情：

```text
学习如何像 user 一样提问。
学习如何像 assistant 一样回答。
```

这会让 SFT 的目标变得奇怪。我们真正想训练的是：

```text
给定 system/user/history，assistant 应该如何回答。
```

所以：

```text
system/user/history = 上下文、条件、题目
assistant = 目标答案、训练对象
```

### 5.2 generate_labels 的核心思路

`SFTDataset.generate_labels` 的核心逻辑可以概括为：

```python
labels = [-100] * len(input_ids)

for 每一段 assistant 回答:
    找到 assistant_start
    找到 assistant_end
    for j in assistant_start 到 assistant_end:
        labels[j] = input_ids[j]
```

为什么要先全部设成 `-100`？

因为默认情况下，SFT 中所有 token 都只是上下文，不应该参与 loss。只有被确认属于 assistant 回答的 token，才恢复成真实 token id。

一句话版本：

```text
先默认都不学，再只把 assistant 的答案拿出来学。
```

## 6. SFT 和 Pretrain 的核心区别

| 阶段 | 输入内容 | labels 怎么做 | 学到什么 |
|---|---|---|---|
| Pretrain | 普通文本 | 非 PAD 基本都参与 loss | 语言规律、知识、续写能力 |
| SFT | 对话数据 | 只让 assistant 部分参与 loss | 按指令回答、遵守对话格式 |

一句话版本：

```text
Pretrain 是让模型学会“语言怎么接下去”。
SFT 是让模型学会“用户这样问时，助手应该这样答”。
```

## 7. Chat Template 的作用

SFT 数据通常不是一段普通字符串，而是结构化对话：

```python
[
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么我可以帮你？"}
]
```

模型不能直接理解 Python 字典，所以需要 `chat_template` 把它转换成一整段标准文本。

MiniMind 的 tokenizer 使用类似这样的角色边界：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好，有什么我可以帮你？<|im_end|>
```

### 7.1 为什么不能简单拼接 content

如果只拼 content：

```text
你好
你好，有什么我可以帮你？
```

模型无法稳定知道：

```text
哪一段是 user。
哪一段是 assistant。
哪一轮对话已经结束。
什么时候该由 assistant 继续生成。
```

所以 chat template 的价值是：

```text
明确角色。
明确轮次边界。
标准化训练格式。
为 assistant-only loss 提供可识别的起止标记。
为 thinking/tool call 等协议留出结构。
```

### 7.2 MiniMind 里的特殊协议

MiniMind 的 tokenizer 配置中包含：

```text
<|im_start|>
<|im_end|>
<think>
</think>
<tool_call>
</tool_call>
<tool_response>
</tool_response>
```

这意味着 chat template 不只是“把文本排整齐”，它还在告诉模型：

```text
这是用户输入。
这是助手回答。
这是思考区。
这是工具调用。
这是工具返回。
```

后面学 Agentic RL 和 Tool Use 时，这些特殊标记会非常重要。

### 7.3 从 conversations 到 input_ids 的完整流程

原始 SFT 样本一般长这样：

```python
conversations = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么我可以帮你？"}
]
```

第一步，`create_chat_prompt` 会调用：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    tools=tools
)
```

这一步还没有分词，只是把结构化对话渲染成一整段字符串：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好，有什么我可以帮你？<|im_end|>
```

第二步，`__getitem__` 再调用：

```python
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
```

这一步才真正分词，把整段 prompt 转成 token id。

所以 role 不是没有被分词，而是先被写进 prompt：

```text
<|im_start|>user\n
<|im_start|>assistant\n
```

然后和 content 一起进入 tokenizer。

一句话版本：

```text
conversations 先通过 chat_template 变成带角色标记的字符串，再通过 tokenizer 变成 input_ids。
```

## 8. 本周 Checklist

- [x] 能解释 tokenizer encode/decode。
- [x] 能解释 BPE/ByteLevel 为什么不是简单按词切分。
- [x] 能逐行解释 `PretrainDataset.__getitem__`。
- [x] 能解释 `bos`、`eos`、`pad` 的作用。
- [x] 能解释为什么 `labels = input_ids.clone()`。
- [x] 能解释为什么 padding 的 label 要设为 `-100`。
- [x] 能解释 pretrain 和 SFT 的 label 差异。
- [x] 能读懂 `SFTDataset.generate_labels` 的主逻辑。
- [x] 能解释为什么需要 chat template 和 role 标记。

## 9. 当前进度

状态：进行中

已完成：

- [x] 第 1 周项目地图。
- [x] 最小训练闭环：`input_ids -> embedding -> Transformer -> logits -> loss`。
- [x] 基础 shape：`[B,T] -> [B,T,C] -> [B,T,V]`。
- [x] 通过 toy 例子写出 pretrain 的 `input_ids` 和 `labels`。
- [x] 理解 SFT 中 user/system 是上下文，assistant 才是 loss 学习目标。
- [x] 理解 `generate_labels` 先全置 `-100`，再恢复 assistant token 的原因。
- [x] 理解 chat template 用于角色边界、轮次边界、标准化格式和工具协议。

待完成：

- [ ] 对比 pretrain loss 和 SFT assistant-only loss。
- [ ] 用一段真实 tokenizer 输出观察 `<|im_start|>`、`<|im_end|>` 和 labels mask。
