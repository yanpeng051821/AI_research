# 第 7 周：推理与采样

## 1. 本周目标

第 7 周从训练切到推理，核心问题是：

```text
给模型一段 prompt 后，模型如何一个 token 一个 token 生成回答？
```

训练阶段关注：

```text
input_ids + labels
-> forward
-> logits
-> loss
-> backward
-> optimizer.step
```

推理阶段关注：

```text
prompt
-> tokenizer
-> model.generate
-> 自回归生成 next_token
-> decode / stream
```

本周需要能解释：

- `eval_llm.py` 如何加载模型和 tokenizer。
- pretrain 权重和 SFT 权重的 prompt 构造区别。
- `generate` 为什么要逐 token 循环生成。
- KV cache 为什么只保存 K/V，不保存 Q。
- `attention_mask`、causal mask、labels mask 的区别。
- `temperature`、`top_k`、`top_p`、`repetition_penalty` 如何影响输出。
- streamer 如何实现边生成边输出。
- 如何用固定 prompt 建立推理评估 baseline。

对应源码：

```text
eval_llm.py
model/model_minimind.py
scripts/chat_api.py
scripts/serve_openai_api.py
```

## 2. 推理入口：eval_llm.py

`eval_llm.py` 是 MiniMind 的基础推理入口，主线可以压缩成：

```text
解析参数
-> 加载 tokenizer
-> 加载模型结构和权重
-> 准备 conversation
-> 构造 prompt
-> tokenizer 转成 input_ids / attention_mask
-> model.generate
-> decode 输出
-> 把 assistant 回复加入 conversation
```

核心代码：

```python
model, tokenizer = init_model(args)
conversation.append({"role": "user", "content": prompt})

inputs = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=bool(args.open_thinking)
)

inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    streamer=streamer,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p,
    temperature=args.temperature,
    repetition_penalty=1
)
```

推理阶段不会创建 optimizer，也不会计算 loss，更不会反向传播。

```python
return model.half().eval().to(args.device), tokenizer
```

这里的含义：

- `half()`：使用 FP16 推理，节省显存并提升速度。
- `eval()`：进入推理模式，关闭 dropout 等训练行为。
- `to(device)`：把模型放到 CPU 或 GPU。

## 3. 模型加载方式

MiniMind 推理支持两种加载方式。

### 3.1 原生 PyTorch 权重

默认：

```text
--load_from model
--save_dir out
--weight full_sft
```

流程是：

```text
从 model/ 加载 tokenizer
-> 用 MiniMindConfig 创建模型结构
-> 从 out/full_sft_768.pth 加载参数矩阵
-> model.half().eval().to(device)
```

关键理解：

```text
.pth 普通权重通常只是 state_dict。
模型结构来自当前代码里的 MiniMindConfig 和 MiniMindForCausalLM。
```

### 3.2 Transformers 格式模型

如果 `load_from` 指向一个完整模型目录，则走：

```python
AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
```

这种目录通常包含：

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
generation_config.json
```

它不是 resume checkpoint。

```text
Transformers 格式模型：用于推理、部署、分享。
resume checkpoint：用于断点续训，包含 optimizer、scaler、epoch、step。
```

## 4. pretrain 和 SFT 的 prompt 区别

Pretrain 权重和 SFT 权重的 prompt 构造不同。

### 4.1 Pretrain 权重

```python
inputs = tokenizer.bos_token + prompt
```

pretrain 学的是普通文本续写，所以输入只需要：

```text
<bos>用户文本
```

模型继续预测后面的 token。

### 4.2 SFT 权重

SFT 模型训练过 chat template，所以推理时要构造对话格式：

```python
conversation.append({"role": "user", "content": prompt})

inputs = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=bool(args.open_thinking)
)
```

抽象后类似：

```text
<|im_start|>user
用户问题
<|im_end|>
<|im_start|>assistant
```

`add_generation_prompt=True` 会在最后补上 assistant 开头。它等于告诉模型：

```text
现在轮到 assistant 继续生成。
```

训练时通常不需要 `add_generation_prompt=True`，因为训练样本里已经有 assistant 答案。

## 5. `<think></think>` 是什么

MiniMind 的 `<think></think>` 不是 Tree of Thoughts，也不是模型结构。

它只是 chat template 里的文本协议：

```text
<think>
中间思考内容
</think>

最终回答
```

对模型来说，`<think>` 和 `</think>` 本质上也是 token。

`open_thinking=0` 时，模板预填空 think，模型更倾向于直接回答。

`open_thinking=1` 时，模板只打开 `<think>`，模型可以先生成显式思考，再生成最终回答。

如果模型没有稳定学会 thinking 格式，就可能出现 `<think>` 泄漏或格式混乱。

## 6. 训练和推理的核心区别

训练阶段有完整输入和完整 labels，因此可以一次性并行计算所有位置的 logits。

```text
input_ids: [B, T]
logits:    [B, T, V]
labels:    [B, T]
```

每个位置的 logits 都用于预测下一个 token。

推理阶段没有未来 token，也没有 labels。模型只能根据已有上下文预测一个 next_token，然后把 next_token 拼回输入，继续预测下一个。

```text
prompt
-> 预测 token_1
-> prompt + token_1
-> 预测 token_2
-> prompt + token_1 + token_2
-> ...
```

所以训练是：

```text
一次 forward，并行预测所有下一个 token。
```

推理是：

```text
多次 forward，自回归逐 token 生成。
```

## 7. generate 的 8 步流程

`MiniMindForCausalLM.generate` 的核心可以记成 8 步：

```text
1. 接收 input_ids
2. 初始化 past_key_values 和 finished
3. 进入 max_new_tokens 循环
4. 根据 past_len 只取未计算的新 token
5. forward 得到最后位置 logits
6. temperature / repetition_penalty / top_k / top_p 处理 logits
7. sample 或 argmax 得到 next_token
8. 拼接 next_token，更新 KV cache，检查 eos
```

关键代码：

```python
for _ in range(max_new_tokens):
    past_len = past_key_values[0][0].shape[1] if past_key_values else 0
    outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache)
    logits = outputs.logits[:, -1, :] / temperature
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    past_key_values = outputs.past_key_values if use_cache else None
```

每轮只生成一个 token。

`max_new_tokens=120` 表示最多生成 120 个新 token，不包括原始 prompt。

## 8. 为什么只取最后一个位置的 logits

模型 forward 后输出：

```text
logits: [B, T, V]
```

其中每个位置都表示：

```text
当前位置之后，下一个 token 的分数分布。
```

推理时只关心当前完整上下文后面应该接什么，所以取：

```python
logits = outputs.logits[:, -1, :]
```

如果当前输入是：

```text
为什么 天空 是 蓝色 的
```

那 `logits[:, -1, :]` 表示：

```text
"的" 后面最可能接什么 token。
```

## 9. logits、softmax 和采样

模型输出的是 logits：

```text
词表中每个 token 的原始分数。
```

logits 不是概率：

```text
可以为负。
不要求相加等于 1。
分数越高，代表模型越倾向选择这个 token。
```

采样前会通过 softmax 转成概率：

```python
probs = torch.softmax(logits, dim=-1)
```

最后：

```python
next_token = torch.multinomial(probs, num_samples=1)
```

表示按概率随机抽一个 token。

如果 `do_sample=False`：

```python
next_token = torch.argmax(logits, dim=-1)
```

就是每次都选分数最高的 token。

## 10. KV cache

如果没有 KV cache，每生成一个新 token，都需要重新计算整段历史的 K/V。

```text
第 1 步：计算 prompt 的 K/V
第 2 步：重新计算 prompt + token_1 的 K/V
第 3 步：重新计算 prompt + token_1 + token_2 的 K/V
```

这会越来越慢。

有 KV cache 后，历史 token 的 K/V 会被保存下来。下一轮只需要计算新 token 的 Q/K/V，然后让新 token 的 Q 去查询历史 K/V。

第一次生成：

```text
input_ids: [1, prompt_len]
past_key_values: None
输出 cache: 每层保存 K/V
```

第二次生成：

```text
input_ids 总长度: [1, prompt_len + 1]
past_len = prompt_len
input_ids[:, past_len:] = [1, 1]
```

也就是说，第二轮只把新 token 输入模型。

但是注意：

```text
虽然输入模型的只是新 token，它仍然可以通过历史 K/V 看到完整上下文。
```

## 11. 为什么 cache 只保存 K/V

Attention 中：

```text
Q：当前 token 要查询什么。
K：历史 token 能如何被匹配。
V：历史 token 真正提供的信息。
```

生成下一个 token 时，当前 token 的 Q 用完就不再需要。

但历史 token 的 K/V 会被未来每个新 token 反复查询。

所以：

```text
Q 是一次性查询。
K/V 是可复用历史。
```

因此 cache 保存 K/V，不保存历史 Q。

## 12. attention_mask、causal mask、labels mask

这三个 mask 名字相似，但作用完全不同。

### causal mask

causal mask 防止模型看未来。

训练时虽然一次性输入完整序列，但第 t 个 token 只能看见它自己和它之前的 token。

```text
A 只能看 A
B 只能看 A, B
C 只能看 A, B, C
```

不能让 B 偷看未来的 C。

### attention_mask

attention_mask 主要用于屏蔽 PAD。

```text
input_ids:
[A, B, C, PAD, PAD]

attention_mask:
[1, 1, 1, 0, 0]
```

代码中会把 PAD 对应的 attention score 加上巨大负数：

```python
scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
```

如果 mask 是 1：

```text
1 - 1 = 0
score 不变
```

如果 mask 是 0：

```text
1 - 0 = 1
score += -1e9
softmax 后概率约等于 0
```

它发生在每个 block 的 self-attention 中：

```text
QK 算完 scores 之后
softmax 之前
```

### labels mask

labels 中的 `-100` 控制 loss 是否计算。

```python
loss = F.cross_entropy(
    x.view(-1, x.size(-1)),
    y.view(-1),
    ignore_index=-100
)
```

`labels=-100` 的位置不参与 loss。

### 三者区别

```text
causal mask：不能看未来。
attention_mask：不能看 PAD。
labels=-100：不计算这个位置的 loss。
```

SFT 中 user token 通常是：

```text
attention_mask = 1
labels = -100
```

含义：

```text
模型可以看 user，因为它是回答上下文。
但不训练模型生成 user。
```

PAD token 通常是：

```text
attention_mask = 0
labels = -100
```

含义：

```text
模型不应该看 PAD。
也不训练模型生成 PAD。
```

assistant 回复 token 通常是：

```text
attention_mask = 1
labels = token_id
```

含义：

```text
模型可以看它前面的上下文。
也训练模型生成 assistant 内容。
```

## 13. generate 中 attention_mask 为什么每轮拼 1

推理时每生成一个新 token，这个 token 都是真实 token，不是 PAD。

所以 attention_mask 也要同步增长。

```text
input_ids:      [10, 20, 30]
attention_mask: [1,  1,  1]

生成 40 后：

input_ids:      [10, 20, 30, 40]
attention_mask: [1,  1,  1,  1]
```

原因是 attention score 的形状是：

```text
scores: [B, heads, query_len, key_len]
```

attention_mask 会广播成：

```text
[B, 1, 1, key_len]
```

开启 KV cache 后，虽然每轮只输入新 token：

```text
query_len = 1
```

但 key/value 长度仍然是完整上下文长度：

```text
key_len = 历史长度 + 当前 token
```

所以 attention_mask 必须和完整 key_len 对齐。

## 14. 采样参数

模型最后位置输出 logits：

```text
logits: [B, V]
```

logits 不是概率，而是每个 token 的原始分数。

### temperature

```python
logits = logits / temperature
```

`temperature < 1`：

```text
放大 logits 差距，输出更稳定、更保守。
```

`temperature > 1`：

```text
缩小 logits 差距，输出更随机、更发散。
```

### top_k

top-k 固定保留分数最高的 k 个候选 token。

```text
top_k=30 表示只在当前最高分的 30 个 token 中采样。
```

### top_p

top-p 保留累计概率达到 p 的候选集合。

```text
top_p=0.8 表示只保留累计概率约 0.8 的高概率 token。
```

top-k 是固定候选数量。

top-p 是固定累计概率质量。

### repetition_penalty

repetition penalty 会降低已经出现过的 token 再次出现的概率。

```python
seen = torch.unique(input_ids[i])
score = logits[i, seen]
logits[i, seen] = torch.where(
    score > 0,
    score / repetition_penalty,
    score * repetition_penalty
)
```

如果已出现 token 的 score 是正数：

```text
score / penalty
```

如果 score 是负数：

```text
score * penalty
```

目的都是让已出现 token 更难再次被采样。

## 15. streamer 和 API 推理

`TextStreamer` 的作用是边生成边 decode，用户不用等整段生成完才看到结果。

`eval_llm.py` 中：

```python
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
```

`generate` 中每生成一个 token，就会：

```python
if streamer:
    streamer.put(next_token.cpu())
```

所以你在终端看到的是逐 token 输出。

`scripts/chat_api.py` 和 `scripts/serve_openai_api.py` 可以看作把同一套推理能力包装成服务：

```text
chat_api.py：客户端，请求模型服务。
serve_openai_api.py：服务端，接收请求，构造 prompt，调用 generate，以 OpenAI API 风格返回。
```

这对后续做教育 Agent 平台很重要，因为 Agent 最终不是只在终端里跑，而是要通过 API 接入产品。

## 16. 本周新增结构化评估脚本

为了让推理实验可复现，本周新增了结构化评估脚本：

```text
eval_llm_report.py
```

它不替换原来的 `eval_llm.py`，而是专门用于固定 prompt 评估。

脚本会输出：

```text
eval_reports/*.jsonl
eval_reports/*.md
```

固定评估维度：

- 身份稳定性。
- 实时信息边界。
- LLM/SFT/RL 专业知识。
- Python 代码能力。
- 教学解释能力。
- 宠物对比推理。
- 重复与长度控制。

本次使用的三组采样参数：

```text
safe:
temperature=0.3
top_p=0.8
top_k=30
repetition_penalty=1.15

balanced:
temperature=0.8
top_p=0.9
top_k=50
repetition_penalty=1.1

creative:
temperature=1.1
top_p=0.95
top_k=80
repetition_penalty=1.05
```

## 17. baseline 评估结果

最新报告：

```text
eval_reports/full_sft_768_20260630_183627.md
```

评估设置：

```text
weight: full_sft
hidden_size: 768
num_hidden_layers: 8
max_new_tokens: 120
open_thinking: False
```

共 24 条记录：

```text
8 个 prompt x 3 组 preset
```

速度：

```text
safe 平均约 45.13 tokens/s
balanced 平均约 47.55 tokens/s
creative 平均约 47.80 tokens/s
```

### 参数结论

`safe` 最稳，适合作为当前模型的默认严肃问答配置。

```text
temperature=0.3
top_p=0.8
top_k=30
repetition_penalty=1.15
max_new_tokens=120
```

`balanced` 有时更短更干净，例如机器学习 120 字以内的问题。

`creative` 幻觉和跑题明显增多，不适合当前模型做严肃问答。

### 能力结论

当前 `full_sft_768` 已经具备基本对话格式能力，但仍不是可靠问答模型。

表现较好的能力：

- 能在天气类问题中承认无法获取实时信息。
- 在 `safe` 参数下身份回答较稳定。
- 能输出基本结构化文本。

表现较弱的能力：

- 专业知识弱：`LLM 的 SFT 和强化学习区别` 三组都失败。
- 代码能力弱：斐波那契函数三组都生成不完整或错误代码。
- 教学解释弱：天空为什么是蓝色没有稳定讲到瑞利散射。
- 指令约束一般：不少样例打满 `max_new_tokens=120`，不会自然收尾。
- 高随机参数下容易幻觉，例如编造知识来源、把 SFT/RL 跑偏到 Linux/Unix 等。

### 归因

这些问题主要不是训练流程错误，也不是单纯采样参数问题。

更可能来自：

- 小模型容量有限。
- pretrain 和 SFT 数据规模有限。
- 专业知识、代码、教育解释类高质量数据不足。
- SFT 对短回答、边界、格式、停止条件的约束不够强。

采样参数只能压住一部分发散，不能让模型从根本上获得不存在的知识和代码能力。

## 18. 下一步

第 7 周已经完成，可以进入第 7.5 周实验工程。

第 7.5 周要做的事情不是继续盲目训练，而是建立实验流水线：

```text
固定 prompt
固定 baseline
一次只改一个变量
记录参数、权重、输出、速度、问题类型
对比改动是否真的有效
```

然后第 8 周进入 LoRA 时，就可以用同一套评估集验证：

```text
高质量教育/LLM 知识/代码数据是否真的改善了模型。
```
