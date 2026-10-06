# 第 10 周：DPO 偏好优化

## 1. 这一周学什么

DPO 的核心问题是：

```text
同一个问题下，哪个回答更好？
```

它和 SFT 的区别：

```text
SFT：给模型一个标准答案，让模型模仿。
DPO：给模型一对回答，告诉模型 chosen 比 rejected 更好。
```

在 MiniMind 中，对应源码：

```text
dataset/lm_dataset.py -> DPODataset
trainer/train_dpo.py
```

本周要建立这条链路：

```text
chosen/rejected 偏好数据
-> DPODataset
-> policy/ref 两个模型
-> 计算 chosen/rejected logprob
-> DPO loss
-> 只更新 policy
```

## 2. 为什么需要 DPO

SFT 只能告诉模型：

```text
这个答案应该被模仿。
```

但很多时候，我们更容易构造偏好数据：

```text
回答 A 比回答 B 更好。
```

例如教育场景：

```text
问题：这道数学题怎么做？

chosen：
先引导学生分析题意，再提示关键公式，最后给出步骤。

rejected：
直接给答案，或者胡乱解释。
```

DPO 学的是：

```text
让模型更偏向 chosen，远离 rejected。
```

## 3. 和 SFT、LoRA、蒸馏的区别

```text
SFT：
学习标准答案。

LoRA：
冻结 base model，只训练低秩增量参数。

Distillation：
学习 teacher 的概率分布。

DPO：
学习 chosen > rejected 的偏好关系。
```

DPO 不要求显式 reward 分数，只需要偏好对：

```text
chosen 比 rejected 好。
```

## 4. DPO 数据格式

MiniMind 的 `DPODataset` 期待每条数据有：

```text
chosen
rejected
```

每个字段都是一个对话列表。

示例：

```json
{
  "chosen": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是一种让计算机从数据中学习规律的方法。"}
  ],
  "rejected": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "不知道，反正就是AI。"}
  ]
}
```

直觉：

```text
chosen：更好的回答。
rejected：更差的回答。
```

## 5. DPODataset 做了什么

代码位置：

```text
dataset/lm_dataset.py
```

核心流程：

```python
chosen = sample["chosen"]
rejected = sample["rejected"]

chosen_prompt = tokenizer.apply_chat_template(
    chosen,
    tokenize=False,
    add_generation_prompt=False
)

rejected_prompt = tokenizer.apply_chat_template(
    rejected,
    tokenize=False,
    add_generation_prompt=False
)
```

和 SFT 一样，先把结构化对话转成 chat template 文本。

然后 tokenizer：

```python
chosen_encoding = tokenizer(
    chosen_prompt,
    truncation=True,
    max_length=max_length,
    padding="max_length"
)
```

得到：

```text
chosen_input_ids
rejected_input_ids
```

再生成 loss mask：

```python
chosen_loss_mask = generate_loss_mask(chosen_input_ids)
rejected_loss_mask = generate_loss_mask(rejected_input_ids)
```

这个 mask 仍然只打开 assistant 回答部分。

最后手动 shift：

```python
x_chosen = chosen_input_ids[:-1]
y_chosen = chosen_input_ids[1:]
mask_chosen = chosen_loss_mask[1:]
```

rejected 同理。

所以一条 DPO 样本返回：

```python
{
    "x_chosen": x_chosen,
    "y_chosen": y_chosen,
    "mask_chosen": mask_chosen,
    "x_rejected": x_rejected,
    "y_rejected": y_rejected,
    "mask_rejected": mask_rejected
}
```

注意：

```text
SFTDataset 返回 input_ids, labels，让模型内部 shift。
DPODataset 返回 x/y/mask，已经在 Dataset 中手动 shift。
```

## 6. policy model 和 reference model

`train_dpo.py` 中会加载两个模型：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)
```

它们一开始来自同一个权重：

```text
full_sft
```

区别：

```text
policy model：继续训练，会更新参数。
reference model：冻结，不更新参数。
```

reference model 的作用：

```text
作为原始模型行为的锚点，防止 policy 过度漂移。
```

DPO 不是无脑提高 chosen 概率，而是：

```text
让 policy 相比 reference 更偏向 chosen。
```

## 7. DPO batch 如何拼接

训练循环里：

```python
x_chosen = batch["x_chosen"]
x_rejected = batch["x_rejected"]
y_chosen = batch["y_chosen"]
y_rejected = batch["y_rejected"]
mask_chosen = batch["mask_chosen"]
mask_rejected = batch["mask_rejected"]
```

然后拼起来：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

如果原始 batch size 是 `B`：

```text
x_chosen:   [B, T-1]
x_rejected: [B, T-1]
```

拼接后：

```text
x:    [2B, T-1]
y:    [2B, T-1]
mask: [2B, T-1]
```

约定：

```text
前半部分是 chosen。
后半部分是 rejected。
```

后面的 `dpo_loss` 会按这个顺序拆开。

## 8. DPO 完整数据流和 shape 变化

先约定几个符号：

```text
B：DataLoader 的 batch size
T：tokenizer padding/truncation 后的 max_length
L：模型实际训练序列长度，等于 T - 1
V：词表大小 vocab_size
```

MiniMind DPO 默认配置大致是：

```text
B = 4
T = 1024
L = 1023
V = 6400
```

### 8.1 原始 JSON 样本

一条 DPO 样本不是单条 text，而是一对回答：

```json
{
  "chosen": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "更好的回答"}
  ],
  "rejected": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "更差的回答"}
  ]
}
```

这里的关键是：

```text
chosen 和 rejected 是同一个问题下的两条候选回答。
训练目标不是单独模仿 chosen，而是让模型更偏好 chosen 而不是 rejected。
```

### 8.2 chat template 阶段

`DPODataset.__getitem__` 会分别处理 chosen 和 rejected：

```python
chosen_prompt = tokenizer.apply_chat_template(
    chosen,
    tokenize=False,
    add_generation_prompt=False
)

rejected_prompt = tokenizer.apply_chat_template(
    rejected,
    tokenize=False,
    add_generation_prompt=False
)
```

这一步把结构化消息列表变成模型认识的完整文本，例如：

```text
<s>user
...
assistant
更好的回答</s>
```

此时还没有变成 tensor，只是两段字符串：

```text
chosen_prompt:   str
rejected_prompt: str
```

### 8.3 tokenizer 阶段

接着分别分词：

```python
chosen_encoding = tokenizer(
    chosen_prompt,
    truncation=True,
    max_length=max_length,
    padding="max_length"
)
```

得到：

```text
chosen_input_ids:   [T]
rejected_input_ids: [T]
```

如果文本太长，会被截断到 `T`。如果文本太短，会用 pad 补到 `T`。

### 8.4 loss mask 阶段

然后生成 mask：

```python
chosen_loss_mask = generate_loss_mask(chosen_input_ids)
rejected_loss_mask = generate_loss_mask(rejected_input_ids)
```

shape 是：

```text
chosen_loss_mask:   [T]
rejected_loss_mask: [T]
```

mask 的作用是告诉 loss 哪些 token 参与训练：

```text
assistant 回答部分：1
user / system / pad 等部分：0
```

DPO 比较的是回答质量，所以一般只让 assistant 回答部分参与 logprob 求和。

### 8.5 手动 shift 阶段

DPO Dataset 里会手动构造自回归训练的输入和目标：

```python
x_chosen = chosen_input_ids[:-1]
y_chosen = chosen_input_ids[1:]
mask_chosen = chosen_loss_mask[1:]
```

含义是：

```text
x：当前看到的 token 序列
y：下一个要预测的 token 序列
mask：y 中哪些位置参与 loss
```

shape 从 `[T]` 变成 `[L]`：

```text
x_chosen:    [L]
y_chosen:    [L]
mask_chosen: [L]

x_rejected:    [L]
y_rejected:    [L]
mask_rejected: [L]
```

所以 `Dataset` 每次返回的一条样本是六个一维 tensor：

```python
{
    "x_chosen": x_chosen,
    "y_chosen": y_chosen,
    "mask_chosen": mask_chosen,
    "x_rejected": x_rejected,
    "y_rejected": y_rejected,
    "mask_rejected": mask_rejected
}
```

### 8.6 DataLoader 组 batch

`DataLoader` 会把 `B` 条样本堆起来：

```text
x_chosen:      [B, L]
y_chosen:      [B, L]
mask_chosen:   [B, L]

x_rejected:    [B, L]
y_rejected:    [B, L]
mask_rejected: [B, L]
```

注意，这里每个 batch 里仍然保留两组数据：

```text
一组 chosen
一组 rejected
```

### 8.7 chosen/rejected 拼接

训练循环里会把 chosen 和 rejected 在 batch 维度拼起来：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

shape 变成：

```text
x:    [2B, L]
y:    [2B, L]
mask: [2B, L]
```

排列顺序非常重要：

```text
x[0:B]      是 chosen
x[B:2B]     是 rejected
y[0:B]      是 chosen 的目标 token
y[B:2B]     是 rejected 的目标 token
mask[0:B]   是 chosen 的有效回答位置
mask[B:2B]  是 rejected 的有效回答位置
```

之所以要这样拼，是为了让 policy/ref 模型一次 forward 同时算完 chosen 和 rejected，后面再按前半/后半拆开。

### 8.8 reference model forward

reference model 不更新参数，只作为原始策略锚点：

```python
with torch.no_grad():
    ref_outputs = ref_model(x)
    ref_logits = ref_outputs.logits
```

输入：

```text
x: [2B, L]
```

输出：

```text
ref_logits: [2B, L, V]
```

含义是：

```text
对 2B 条序列中每个位置，都给出一个 V 维词表分数。
```

### 8.9 policy model forward

policy model 是真正要训练的模型：

```python
outputs = model(x)
logits = outputs.logits
```

shape 同样是：

```text
logits: [2B, L, V]
```

区别是：

```text
ref_logits：来自冻结的 reference model
logits：来自正在训练的 policy model
```

### 8.10 从 logits 取出目标 token 的 logprob

代码：

```python
ref_log_probs = logits_to_log_probs(ref_logits, y)
policy_log_probs = logits_to_log_probs(logits, y)
```

输入 shape：

```text
ref_logits: [2B, L, V]
logits:     [2B, L, V]
y:          [2B, L]
```

`logits_to_log_probs` 内部先做：

```python
log_probs = F.log_softmax(logits, dim=2)
```

得到：

```text
log_probs: [2B, L, V]
```

然后用 `gather` 按 `y` 取目标 token 的 logprob：

```python
log_probs_per_token = torch.gather(
    log_probs,
    dim=2,
    index=y.unsqueeze(2)
).squeeze(-1)
```

shape 变化：

```text
y:                 [2B, L]
y.unsqueeze(2):    [2B, L, 1]
gather 后:          [2B, L, 1]
squeeze(-1) 后:     [2B, L]
```

最终得到：

```text
ref_log_probs:    [2B, L]
policy_log_probs: [2B, L]
```

这两个 tensor 的含义是：

```text
每条序列、每个位置上，模型给真实下一个 token 的 logprob。
```

### 8.11 mask 并求整句 logprob

进入 `dpo_loss` 后：

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)
policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```

shape 变化：

```text
ref_log_probs:    [2B, L] -> [2B]
policy_log_probs: [2B, L] -> [2B]
mask:             [2B, L]
```

这里的 `[2B]` 表示：

```text
每条回答的整段 assistant logprob 总和。
```

因为 log 概率相加等价于原始概率相乘：

```text
log P(整段回答)
= log P(token1) + log P(token2) + ... + log P(tokenN)
```

### 8.12 拆回 chosen/rejected

因为前面拼接时约定前半是 chosen，后半是 rejected，所以这里直接切片：

```python
batch_size = ref_log_probs.shape[0]

chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
reject_ref_log_probs = ref_log_probs[batch_size // 2:]

chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
reject_policy_log_probs = policy_log_probs[batch_size // 2:]
```

由于这里的 `batch_size` 实际是 `2B`，所以切完后：

```text
chosen_ref_log_probs:     [B]
reject_ref_log_probs:     [B]
chosen_policy_log_probs:  [B]
reject_policy_log_probs:  [B]
```

每个位置仍然是一一对应的：

```text
chosen_policy_log_probs[i]
reject_policy_log_probs[i]
```

来自同一条原始偏好样本中的 chosen/rejected 对。

### 8.13 计算偏好差

policy 自己对 chosen 的偏好强度：

```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
```

reference 自己对 chosen 的偏好强度：

```python
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
```

shape 都是：

```text
pi_logratios:  [B]
ref_logratios: [B]
```

然后比较 policy 相对 reference 的偏好变化：

```python
logits = pi_logratios - ref_logratios
```

shape：

```text
logits: [B]
```

这里的 `logits` 不是词表 logits，而是 DPO loss 里的偏好分数：

```text
logits > 0：policy 比 reference 更偏向 chosen
logits < 0：policy 没有比 reference 更偏向 chosen，甚至更偏向 rejected
```

### 8.14 计算 DPO loss 并反传

最后：

```python
loss = -F.logsigmoid(beta * logits)
loss = loss.mean()
```

shape 变化：

```text
beta * logits:        [B]
-logsigmoid(...):     [B]
mean 后:              scalar
```

然后训练循环执行：

```python
scaler.scale(loss + aux_loss).backward()
scaler.step(optimizer)
scaler.update()
```

只有 policy model 更新参数：

```text
policy model：参与 backward，参数更新
reference model：no_grad + requires_grad_(False)，参数不更新
```

完整 shape 流可以压缩成这张表：

```text
原始一条样本：
chosen/rejected messages

chat template：
chosen_prompt / rejected_prompt: str

tokenizer：
chosen_input_ids / rejected_input_ids: [T]

shift：
x/y/mask: [L]

DataLoader：
x_chosen / x_rejected: [B, L]

cat：
x/y/mask: [2B, L]

model forward：
logits/ref_logits: [2B, L, V]

gather 目标 token：
policy_log_probs/ref_log_probs: [2B, L]

mask + sum：
policy_log_probs/ref_log_probs: [2B]

split chosen/rejected：
chosen/rejected: [B]

logratio：
pi_logratios/ref_logratios: [B]

DPO logits：
logits: [B]

loss：
scalar
```

## 9. logits_to_log_probs 做什么

代码：

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(
        log_probs,
        dim=2,
        index=labels.unsqueeze(2)
    ).squeeze(-1)
    return log_probs_per_token
```

输入：

```text
logits: [batch, seq_len, vocab_size]
labels: [batch, seq_len]
```

第一步：

```python
log_probs = F.log_softmax(logits, dim=2)
```

把每个位置的词表 logits 转成 log 概率。

第二步：

```python
torch.gather(..., index=labels.unsqueeze(2))
```

从整个词表的 log_probs 中，取出真实目标 token 对应的 logprob。

输出：

```text
log_probs_per_token: [batch, seq_len]
```

意思是：

```text
模型对这一段回答中每个目标 token 的 logprob。
```

## 10. dpo_loss 的直觉

代码：

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)
policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```

这一步把 token 级 logprob 加起来，得到整段 assistant 回答的 logprob：

```text
一句回答的 logprob = 每个有效 token 的 logprob 总和。
```

然后拆 chosen/rejected：

```python
chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
reject_ref_log_probs = ref_log_probs[batch_size // 2:]
chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
reject_policy_log_probs = policy_log_probs[batch_size // 2:]
```

policy 对 chosen 的偏好：

```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
```

reference 对 chosen 的偏好：

```python
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
```

DPO 关心：

```python
logits = pi_logratios - ref_logratios
```

也就是：

```text
policy 比 reference 更偏向 chosen 多少。
```

最后：

```python
loss = -F.logsigmoid(beta * logits)
```

如果 policy 已经比 reference 更偏向 chosen：

```text
logits 大，loss 小。
```

如果 policy 没有更偏向 chosen，甚至更偏向 rejected：

```text
logits 小或为负，loss 大。
```

训练会推动 policy：

```text
提高 chosen 相对 rejected 的优势。
```

## 11. beta 的作用

MiniMind 默认：

```python
beta = 0.15
```

beta 控制偏好优化强度：

```text
beta 大：偏好信号更强，policy 更快远离 reference。
beta 小：更新更保守，更不容易破坏原模型。
```

MiniMind 的 DPO 学习率也很小：

```text
4e-8
```

代码注释写着：

```text
建议 <= 5e-8 避免遗忘。
```

这说明 DPO 对模型行为影响很敏感，需要谨慎。

## 12. 为什么 DPO 不需要显式 reward model

传统 RLHF 常见流程：

```text
收集偏好数据
-> 训练 reward model
-> 用 PPO 等 RL 方法优化 policy
```

DPO 直接把偏好对写成一个 supervised-style loss：

```text
chosen > rejected
```

所以它不需要单独训练 reward model。

但 DPO 仍然需要：

```text
reference model
```

reference 不是 reward model。

它只是原始策略的锚点，用来约束 policy 不要偏离太远。

## 13. DPO 完整训练流

```text
DPODataset
-> chosen / rejected 两条回答
-> 分别构造 x/y/mask
-> 拼成 [chosen; rejected] batch
-> reference no_grad forward
-> policy forward
-> 分别计算 chosen/rejected 的整句 logprob
-> 比较 policy 相对 reference 的偏好变化
-> DPO loss
-> backward 只更新 policy
-> 保存 dpo_xxx.pth
```

最终保存的是：

```text
policy model
```

不是 reference model。

## 14. 和教育 Agent 的关系

教育场景里，DPO 适合优化这些偏好：

```text
chosen：启发式提示。
rejected：直接泄题。

chosen：承认不确定，给出求证方法。
rejected：编造事实。

chosen：结构清晰，分步骤解释。
rejected：长篇复读、逻辑混乱。

chosen：先问学生卡在哪里。
rejected：直接输出完整答案。
```

也就是说，SFT 可以让模型学会教学格式，DPO 可以继续让模型偏好更好的教学行为。

## 15. 现阶段检查题

1. DPO 数据为什么需要 chosen/rejected？
2. DPODataset 为什么要分别返回 x/y/mask？
3. policy model 和 reference model 有什么区别？
4. reference model 为什么冻结？
5. `logits_to_log_probs` 为什么要用 `gather`？
6. DPO 为什么比较整句 logprob，而不是只看单个 token？
7. `pi_logratios - ref_logratios` 表示什么？
8. beta 变大和变小分别会发生什么？
9. DPO 为什么不需要显式 reward model？

能讲清楚这些，就可以继续进入第 11 周 GRPO / RL。
