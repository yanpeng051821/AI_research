# 第 11 周：GRPO 与 RL 入门

## 1. 这一周学什么

GRPO 要解决的问题不是“给模型一个标准答案，让它模仿”，而是：

```text
让模型先自己回答，再根据回答质量调整它以后生成这类回答的概率。
```

所以它和前面几种训练方式的差异很明显：

```text
SFT：
数据里已经有 prompt + answer。
训练目标是模仿标准 assistant 回复。

DPO：
数据里已经有 prompt + chosen + rejected。
训练目标是让模型更偏好 chosen，而不是 rejected。

GRPO：
数据里主要有 prompt。
模型要先 rollout 生成多个 response。
再用规则 / reward model 给这些 response 打分。
最后根据分数反过来更新 policy model。
```

MiniMind 中主要看这几个文件：

```text
trainer/train_grpo.py
trainer/rollout_engine.py
trainer/trainer_utils.py -> LMForRewardModel
dataset/lm_dataset.py -> RLAIFDataset
```

本周最重要的是把下面这条链路走顺：

```text
prompt
-> rollout 生成多个 response
-> reward_model / 规则打分
-> 同组 reward 归一化为 advantage
-> policy/ref 计算 token logprob
-> policy gradient loss + KL penalty
-> 只更新 policy
```

一句话先建立直觉：

```text
completion 是模型交的卷子。
reward 是判卷分数。
advantage 是同一道题里这份卷子比其他卷子好多少。
logprob 是模型写出这份卷子的概率痕迹。
ratio 是当前模型相对写卷子时变了多少。
KL 是防止模型为了高分学歪。
loss 是把这些信号合成一个可反向传播的标量。
```

## 2. 从 prompt 到 rollout

SFT 和 DPO 的 answer 都已经在数据里了，所以训练时直接读数据即可。

GRPO 不一样，它需要先让当前 policy model 真的生成一批回答：

```text
prompt -> policy model.generate(...) -> response
```

这个“让模型现场生成回答”的过程就叫：

```text
rollout
```

在 `train_grpo.py` 中，DataLoader 先拿到一批 prompt：

```python
prompts = batch["prompt"]
prompt_inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    padding_side="left",
    add_special_tokens=False,
).to(args.device)
```

这里 tokenizer 的作用只是：

```text
把已有 prompt 文本编码成 prompt_ids。
```

然后 rollout 才会真正生成新 token：

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

所以要分清楚：

```text
tokenizer：
编码已有文本，不创造新内容。

model.generate：
基于 prompt，一个 token 一个 token 生成新的 response。
```

如果 prompt 已经通过 chat template 组织成：

```text
<|im_start|>system
你是一个助手
<|im_end|>
<|im_start|>user
解释什么是机器学习
<|im_end|>
<|im_start|>assistant
```

那么最后的：

```text
<|im_start|>assistant
```

就是 assistant 回复的起始位置。

GRPO 的 generate 本质上是在这个位置继续写：

```text
chat prompt + assistant 起始标记
-> assistant response
```

所以它仍然是“续写”，只是续写的是 assistant 应该回答的位置，而不是普通文章。

## 3. 为什么同一个 prompt 要生成多个 response

MiniMind 默认每个 prompt 会生成多个回答：

```python
num_generations = 6
```

假设 batch size 是 `B`，每个 prompt 生成 `G` 个回答，那么 rollout 后会得到：

```text
B * G
```

条 response。

例如：

```text
prompt_1 -> response_1_1, response_1_2, ..., response_1_6
prompt_2 -> response_2_1, response_2_2, ..., response_2_6
```

为什么要一题多答？

因为 GRPO 的 `G` 是 Group，核心是组内比较：

```text
同一个 prompt 下，哪个回答比同组平均更好？
哪个回答比同组平均更差？
```

这和 DPO 的思路不同：

```text
DPO：
数据集提前告诉你 chosen > rejected。

GRPO：
模型现场生成多个回答，再通过 reward 判断相对好坏。
```

如果一个 prompt 只生成一个回答，就很难知道它“相对同题其他回答”好还是差。

## 4. RolloutResult 里到底有什么

`rollout_engine.rollout(...)` 返回的是一个 `RolloutResult`：

```python
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor
```

它是 GRPO 后面所有计算的中转站。

约定：

```text
B：DataLoader batch size
G：num_generations
P：prompt token 长度
R：response token 长度
V：vocab_size
```

rollout 之后主要得到：

```text
output_ids:        [B * G, P + R]
completion_ids:    [B * G, R]
per_token_logps:   [B * G, R]
completions:       list[str]，长度 B * G
prompt_lens:       [B * G]
completion_mask:   [B * G, R]
```

这些字段分别用于不同分支：

```text
completion_ids：
只包含模型生成的 response token。
后面会 decode 成文本，交给 reward 评分。

completions：
response 文本。
后面进入 calculate_rewards。

output_ids：
prompt + response 的完整 token。
后面重新喂给 policy/ref，计算这些 response token 的 logprob。

per_token_logps：
rollout 生成这些 response 时，旧 policy 对每个 response token 的 logprob。
后面用来计算 ratio。

completion_mask：
标记哪些 completion token 是有效 token，哪些是 padding。
后面用于 loss 统计。
```

这里最容易混的是：

```text
completion_ids 不是 label。
completion_ids 是模型自己刚刚生成出来的回答 token。
```

GRPO 会把模型自己生成的 completion 临时变成训练材料。

## 5. reward 分支：从 completion 文本到 reward

rollout 之后，模型已经交出了 `B * G` 份回答。

接下来要判断每份回答质量如何：

```python
rewards = calculate_rewards(prompts, completions, reward_model)
```

注意这里用的是：

```text
completions: list[str]
```

也就是把 `completion_ids` decode 后得到的文本。

MiniMind 的 reward 不是单一来源，而是：

```text
最终 reward = 规则分 + reward_model 分
```

规则分包括：

```text
1. 回答长度是否合适。
2. 是否包含 </think>。
3. thinking 内容长度是否合适。
4. </think> 是否只出现一次。
5. 回答是否重复。
```

对应代码：

```python
rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

if "</think>" in response:
    thinking_content, answer_content = response.split("</think>", 1)
    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
    rewards[response_idx] += 0.25 if response.count("</think>") == 1 else -0.25

rewards[response_idx] -= rep_penalty(answer)
```

然后再交给 reward model：

```python
score = reward_model.get_score(messages, answer)
reward_model_scores.append(score)
rewards += reward_model_scores
```

`LMForRewardModel.get_score(...)` 会把问题上下文和模型回答组织成：

```python
eval_messages = [
    {"role": "user", "content": message_context},
    {"role": "assistant", "content": response},
]
```

再调用：

```python
score = self.model.get_score(self.tokenizer, eval_messages)
return max(min(score, 3.0), -3.0)
```

也就是把 reward model 的分数裁剪到：

```text
[-3.0, 3.0]
```

最终：

```text
rewards: [B * G]
```

每个 response 对应一个标量 reward。

重要区别：

```text
label：
告诉模型每个位置应该预测哪个 token。

reward：
只告诉模型整段回答整体好不好。
```

reward 不能像交叉熵 label 那样直接逐 token 监督，所以后面还需要把它变成 advantage，再和 logprob 结合。

## 6. 从 reward 到 advantage

同一个 prompt 的多个回答应该放在一起比较。

MiniMind 中的代码是：

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

假设：

```text
B = 2
G = 3
```

rollout 生成顺序通常是：

```text
prompt_1 的回答 1
prompt_1 的回答 2
prompt_1 的回答 3
prompt_2 的回答 1
prompt_2 的回答 2
prompt_2 的回答 3
```

reward model 打分后：

```python
rewards = [2.0, 1.0, 0.0, 10.0, 9.0, 8.0]
```

直接比较绝对 reward 不公平，因为不同 prompt 难度不同。

所以先 reshape 成组：

```python
grouped_rewards = [
    [2.0, 1.0, 0.0],
    [10.0, 9.0, 8.0],
]
```

每一行就是同一个 prompt 的多个回答。

然后在每组内部算：

```text
advantage = (当前回答 reward - 同组平均 reward) / 同组 reward 标准差
```

对于第一组：

```text
reward: [2.0, 1.0, 0.0]
mean:   1.0
std:    约 0.816

advantage:
[(2.0 - 1.0) / 0.816, (1.0 - 1.0) / 0.816, (0.0 - 1.0) / 0.816]
= [1.22, 0, -1.22]
```

含义是：

```text
advantage > 0：
这个 response 比同组平均更好，应该提高它的生成概率。

advantage < 0：
这个 response 比同组平均更差，应该降低它的生成概率。

advantage ≈ 0：
这个 response 接近同组平均，训练信号弱。
```

所以：

```text
reward 是原始分数。
advantage 是组内相对分数。
```

如果同一个 prompt 下的 reward 都差不多：

```text
[1.0, 1.0, 1.0, 1.0]
```

那么：

```text
reward - mean ≈ 0
advantage ≈ 0
```

训练信号就会很弱。

这也是为什么 GRPO 要同一个 prompt 采样多个回答：

```text
只有组内回答之间有差异，relative advantage 才有意义。
```

## 7. logprob 分支：为什么还要重新跑 policy/ref

到这里，reward 分支已经得到了：

```text
advantages: [B * G]
```

但 advantage 只是方向信号：

```text
好回答提高概率。
差回答降低概率。
```

它本身不能直接反向传播到模型参数。

模型真正能通过梯度更新的是：

```text
当前 policy 对这些 completion token 的 logprob。
```

所以 GRPO 还要把完整输出重新喂给当前 policy：

```python
res = model_unwrapped(outputs, attention_mask=full_mask)
```

其中：

```text
outputs = output_ids = prompt + completion
```

得到 logits：

```text
logits: [B * G, P + R, V]
```

然后从 logits 中取出 completion 部分每个真实生成 token 的 logprob：

```python
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1) \
    .gather(2, outputs[:, 1:].unsqueeze(-1)) \
    .squeeze(-1) \
    .gather(1, logp_pos)
```

这里和 DPO 里的 `logits_to_log_probs` 很像，本质仍然是：

```text
logits -> log_softmax -> gather 目标 token 的 logprob
```

最后得到：

```text
per_token_logps: [B * G, R]
```

也就是：

```text
当前 policy 对每个 completion token 的 logprob。
```

与此同时，MiniMind 还会用冻结的 reference model 计算同一批 token 的 logprob：

```text
ref_per_token_logps: [B * G, R]
```

所以同一批 completion token 会有三套概率信息：

```text
old_per_token_logps：
rollout 生成这些 token 时，旧 policy 的 logprob。

per_token_logps：
当前正在训练的 policy 重新计算出来的 logprob。

ref_per_token_logps：
冻结 reference model 计算出来的 logprob。
```

这三套 logprob 分别有不同用途，不能混在一起。

## 8. ratio 和 KL：它们看起来像，但管的不是一件事

GRPO 中两个最容易混的量是：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)

kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
```

它们都在比较 logprob，所以会感觉像一个东西。

但它们比较对象不同，作用也不同。

### 8.1 ratio：当前 policy 和 old policy 比

`ratio` 比的是：

```text
当前 policy vs rollout 时的 old policy
```

因为：

```text
per_token_logps = log 当前 policy 概率
old_per_token_logps = log rollout 时旧 policy 概率
```

所以：

```text
ratio = exp(log 当前概率 - log 旧概率)
      = 当前概率 / 旧概率
```

含义：

```text
ratio > 1：
当前 policy 比 rollout 时更愿意生成这个 token。

ratio < 1：
当前 policy 比 rollout 时更不愿意生成这个 token。
```

ratio 主要控制：

```text
这一次参数更新，相对刚才生成这些 response 时，变化不要太猛。
```

它回答的问题是：

```text
当前模型相比刚才采样这些回答时的自己，改了多少？
```

### 8.2 KL：当前 policy 和 reference model 比

`per_token_kl` 比的是：

```text
当前 policy vs 冻结的 reference model
```

reference model 通常来自训练前的 SFT 权重：

```python
ref_model = ref_model.eval().requires_grad_(False)
```

它不更新，只当基准。

KL 主要控制：

```text
policy 不要为了追 reward，长期偏离原来的语言能力和行为风格太远。
```

它回答的问题是：

```text
当前模型相比初始基准模型，跑偏了多少？
```

为什么需要这个约束？

因为 reward 不完美，模型可能为了拿高分学会钻空子：

```text
为了长度分，故意写很长。
为了 thinking 分，乱写 </think>。
为了迎合 reward_model，说空泛但看起来漂亮的话。
```

这叫：

```text
reward hacking
```

所以：

```text
ratio 像本轮更新的安全带。
KL 像长期训练的稳定器。
```

再用一句更形象的话：

```text
advantage 是油门。
ratio 是限制这脚油门不要踩爆。
KL/ref_model 是防止车开偏离主路。
```

## 9. advantage、ratio、KL 怎么汇合成 loss

到这里我们有两条分支：

```text
reward 分支：
completion_ids -> decode -> completions -> reward -> advantage

logprob 分支：
output_ids -> policy/ref forward -> per_token_logps / ref_per_token_logps
```

真正汇合的地方是 loss。

最简化的 policy gradient 直觉是：

```text
loss = - advantage * logprob
```

如果：

```text
advantage > 0
```

说明这个 response 比同组平均好。

优化器为了降低 loss，会推动这段 response 的 token logprob 变大：

```text
logprob 变大 -> 生成概率变大
```

如果：

```text
advantage < 0
```

说明这个 response 比同组平均差。

优化器为了降低 loss，会推动这段 response 的 token logprob 变小：

```text
logprob 变小 -> 生成概率变小
```

MiniMind 的真实代码会再加入 ratio 和 KL。

默认 `loss_type="cispo"` 时：

```python
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
per_token_loss = -(
    clamped_ratio * advantages.unsqueeze(1) * per_token_logps
    - args.beta * per_token_kl
)
```

可以拆成：

```text
per_token_loss
= - reward_related_term + beta * KL
```

其中：

```text
clamped_ratio：
控制这次更新幅度，不让更新过猛。

advantages.unsqueeze(1)：
把每个 response 的整体 advantage 广播给这段 response 的所有 token。

per_token_logps：
当前 policy 真正可反向传播的对象。

beta * per_token_kl：
惩罚 policy 偏离 reference model。
```

shape 上：

```text
per_token_logps:      [B * G, R]
old_per_token_logps:  [B * G, R]
ratio:                [B * G, R]
advantages:           [B * G]
advantages.unsqueeze: [B * G, 1]
per_token_kl:         [B * G, R]
per_token_loss:       [B * G, R]
```

`advantages.unsqueeze(1)` 会广播：

```text
[B * G, 1] -> [B * G, R]
```

也就是说：

```text
一个回答整体 reward 高，这个回答里的所有有效 token 都被鼓励。
一个回答整体 reward 低，这个回答里的所有有效 token 都被压低。
```

这确实比 token-level label 粗糙，因为 reward model 给的是整段回答的分数，不是每个 token 的分数。

但在 RLHF/RLAIF 里，这种做法很常见：

```text
整段回答的好坏信号，会分配给这段回答的生成路径。
```

如果使用 `loss_type="grpo"`，代码是：

```python
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss1 = ratio * advantages.unsqueeze(1)
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

这条分支更像 PPO 的 clipped objective。

当前阶段不需要死扣推导，先记住主线：

```text
advantage 决定提高还是降低概率。
logprob 是可训练对象。
ratio 限制本轮更新幅度。
KL 限制长期偏离 reference。
```

## 10. completion_mask：为什么最后还要 mask

不同 response 长度不同，短回答后面会 padding。

padding token 不应该参与 loss。

MiniMind 最后用：

```python
policy_loss = (
    (per_token_loss * completion_mask).sum(dim=1)
    / completion_mask.sum(dim=1).clamp(min=1)
).mean()
```

其中：

```text
per_token_loss:  [B * G, R]
completion_mask: [B * G, R]
```

这一步做了两件事：

```text
1. 只统计有效 response token，不统计 padding。
2. 先对每条 response 的 token loss 求平均，再对所有 response 求平均。
```

最后得到：

```text
policy_loss: scalar
```

有了标量 loss，才能：

```python
loss.backward()
optimizer.step()
```

而且 GRPO 中：

```text
只更新 policy model。
reference model 冻结。
reward model 只打分，也不在这里更新。
```

## 11. policy、old policy、reference model、reward model

GRPO 里有四个容易混淆的对象：

```text
policy model：
当前正在训练、会更新参数的模型。

old policy：
不是一个单独模型，而是 rollout 时保存下来的 old_per_token_logps。

reference model：
冻结的基准模型，用来计算 KL penalty，防止 policy 跑太远。

reward model：
裁判模型，只负责给 response 打分，不负责生成，也不在 GRPO 里更新。
```

在 MiniMind 中：

```python
model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)
```

一开始：

```text
policy model 和 reference model 通常来自同一个 full_sft 权重。
```

训练过程中：

```text
policy model：
更新。

reference model：
冻结。

old policy：
用 old_per_token_logps 固化 rollout 时的行为。

reward model：
只打分。
```

## 12. 完整数据流和 shape

完整流程可以按这条线复盘：

```text
RLAIFDataset:
prompt: list[str], 长度 B

tokenizer:
prompt_ids:     [B, P]
attention_mask: [B, P]

rollout:
每个 prompt 生成 G 个 response

output_ids:
[B * G, P + R]

completion_ids:
[B * G, R]

old_per_token_logps:
[B * G, R]

completions:
list[str], 长度 B * G

calculate_rewards:
rewards: [B * G]

grouped_rewards:
[B, G]

advantages:
[B * G]

policy forward:
logits: [B * G, P + R, V]

取 completion token logprob:
per_token_logps: [B * G, R]

reference forward:
ref_per_token_logps: [B * G, R]

ratio:
ratio: [B * G, R]

KL:
per_token_kl: [B * G, R]

per_token_loss:
[B * G, R]

completion_mask 后求平均:
loss: scalar
```

把代码和意义合在一起看：

```text
1. DataLoader 给 prompt。
2. tokenizer 把 prompt 变成 prompt_ids。
3. rollout_engine 调 model.generate。
4. 得到 output_ids = prompt_ids + completion_ids。
5. completion_ids decode 成 completions 文本。
6. calculate_rewards 给每个 completion 打 reward。
7. grouped rewards 把 reward 转成 advantage。
8. 当前 policy 重新计算 completion token 的 per_token_logps。
9. rollout 时保存的 old_per_token_logps 提供旧概率。
10. ratio = 当前概率 / 旧概率。
11. ref_model 计算 ref_per_token_logps。
12. per_token_kl 衡量 policy 偏离 reference 的程度。
13. advantage 决定提高还是降低这些 token 概率。
14. KL penalty 限制 policy 不要为了 reward 跑偏。
15. completion_mask 去掉 padding。
16. 汇总成 scalar loss。
17. backward 只更新 policy。
```

## 13. GRPO 代码级完整数据流和 shape

这一节按真实执行顺序看：

```text
原始 JSON 文本
-> RLAIFDataset
-> DataLoader
-> tokenizer
-> rollout_engine.rollout
-> policy/ref 重新算 logprob
-> reward / advantage
-> GRPO loss
-> backward / optimizer.step
```

先约定符号：

```text
B：DataLoader batch size
G：num_generations，每个 prompt 生成几个回答
P：prompt token 长度
R：completion/response token 长度
S：完整序列长度，S = P + R
V：vocab_size
```

MiniMind 默认大致是：

```text
B = 2
G = 6
max prompt length = 768
max generation length = 1024
```

所以一次训练 step 里，真正参与 RL 计算的 response 数量是：

```text
B * G = 12
```

### 13.1 原始 JSON 样本

GRPO 使用的是：

```text
dataset/lm_dataset.py -> RLAIFDataset
```

原始数据中每条样本大致是：

```json
{
  "conversations": [
    {"role": "user", "content": "解释什么是机器学习"},
    {"role": "assistant", "content": "参考回答..."}
  ]
}
```

但是注意：

```text
GRPO 训练时不会直接拿最后的 assistant 参考回答当 labels。
```

`RLAIFDataset` 里真正返回的是 prompt：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    prompt = self.create_chat_prompt(sample['conversations'])

    return {
        'prompt': prompt,
        'answer': ""
    }
```

这里的 `answer` 是空字符串，当前 GRPO 主流程没有用它训练。

### 13.2 RLAIFDataset：messages -> prompt 文本

`create_chat_prompt` 里：

```python
return self.tokenizer.apply_chat_template(
    conversations[:-1],
    tokenize=False,
    open_thinking=use_thinking,
    add_generation_prompt=True
)
```

这里有三个关键点：

```text
1. conversations[:-1]：
   去掉原始数据里的最后一条 assistant 回答。

2. tokenize=False：
   此时还不分词，只生成字符串。

3. add_generation_prompt=True：
   在结尾加上 assistant 回复的起始标记，让模型从这里开始续写。
```

所以一条样本从结构化 messages 变成：

```text
prompt: str
```

例子：

```text
<|im_start|>user
解释什么是机器学习
<|im_end|>
<|im_start|>assistant
```

此时没有 tensor，也没有 shape。

### 13.3 DataLoader：单条 prompt -> batch prompts

`DataLoader` 把 `B` 条样本合成一个 batch：

```python
prompts = batch['prompt']
```

此时：

```text
prompts: list[str], 长度 B
```

如果：

```text
B = 2
```

那么：

```text
prompts = [
    prompt_1,
    prompt_2
]
```

这里仍然是文本列表，还没有 token id。

### 13.4 tokenizer：batch prompts -> prompt_ids

训练循环中：

```python
prompt_inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    return_token_type_ids=False,
    padding_side="left",
    add_special_tokens=False
).to(args.device)
```

这一步经过的模块是：

```text
tokenizer
```

shape 从文本列表变成 tensor：

```text
prompts: list[str], 长度 B

input_ids:      [B, P]
attention_mask: [B, P]
```

其中：

```text
input_ids：
prompt 的 token id。

attention_mask：
哪些位置是真实 prompt token，哪些位置是 padding。
```

因为这里使用：

```python
padding_side="left"
```

所以短 prompt 会在左侧 padding：

```text
[pad, pad, pad, 真实 prompt token...]
```

随后如果超过最大 prompt 长度，会截断左侧更早内容，只保留最后一段：

```python
prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

shape 仍然是：

```text
[B, P]
```

只是 `P` 被限制到不超过 `args.max_seq_len`。

### 13.5 rollout_engine.rollout：prompt_ids -> 多个 response

训练循环调用：

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

这一步经过的模块是：

```text
trainer/rollout_engine.py -> TorchRolloutEngine.rollout
```

在 `TorchRolloutEngine.rollout` 中，先把每个 prompt 复制 `G` 份：

```python
prompt_ids.repeat_interleave(num_generations, dim=0)
attention_mask.repeat_interleave(num_generations, dim=0)
```

shape 变化：

```text
prompt_ids:     [B, P] -> [B * G, P]
attention_mask: [B, P] -> [B * G, P]
```

如果：

```text
B = 2
G = 6
```

就变成：

```text
[2, P] -> [12, P]
```

排列顺序是：

```text
prompt_1, prompt_1, prompt_1, prompt_1, prompt_1, prompt_1,
prompt_2, prompt_2, prompt_2, prompt_2, prompt_2, prompt_2
```

也就是每个 prompt 连续重复 `G` 次。

### 13.6 model.generate：生成完整 output_ids

rollout 里真正生成 response 的代码：

```python
output_ids = model.generate(
    input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
    attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    num_return_sequences=1,
    pad_token_id=self.tokenizer.pad_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
)
```

这一步经过的模块是：

```text
policy model 的 generate 方法
```

输入：

```text
[B * G, P]
```

输出：

```text
output_ids: [B * G, S]
```

其中：

```text
S = P + R
```

`output_ids` 是完整序列：

```text
prompt tokens + 模型新生成的 response tokens
```

注意：

```text
tokenizer 只是把已有 prompt 文本编码成 token id。
generate 才是模型现场生成新的 response token id。
```

### 13.7 rollout 切出 completion_ids

rollout 里接着做：

```python
prompt_len = prompt_ids.size(1)
completion_ids = output_ids[:, prompt_len:]
```

shape 变化：

```text
output_ids:      [B * G, S]
completion_ids:  [B * G, R]
```

含义：

```text
output_ids：
prompt + response。

completion_ids：
只保留 response。
```

然后：

```python
completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
```

得到：

```text
completions: list[str], 长度 B * G
```

这个文本列表后面会交给 reward 逻辑打分。

### 13.8 rollout 计算 old_per_token_logps

rollout 中还会计算一次生成结果的 logprob：

```python
full_mask = (output_ids != self.tokenizer.pad_token_id).long()
per_token_logps = compute_per_token_logps(
    self.policy_model,
    output_ids,
    completion_ids.size(1),
    attention_mask=full_mask
)
```

这一步经过的模块是：

```text
rollout_engine.py -> compute_per_token_logps
```

它做的事情是：

```text
把完整 output_ids 再喂给 policy model，
取出 completion 部分每个 token 的 logprob。
```

输出 shape：

```text
per_token_logps: [B * G, R]
```

训练循环里会保存为：

```python
old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
```

为什么叫 old？

```text
因为它代表生成这些 response 时的 policy 概率。
后面 policy 参数会更新，所以这份 logprob 要固定下来作为旧策略基准。
```

### 13.9 rollout_result 回到训练循环

训练循环拿到：

```python
outputs = rollout_result.output_ids
completion_ids = rollout_result.completion_ids
completions = rollout_result.completions
old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
prompt_lens = rollout_result.prompt_lens.to(args.device)
```

对应 shape：

```text
outputs:             [B * G, S]
completion_ids:      [B * G, R]
completions:         list[str], 长度 B * G
old_per_token_logps: [B * G, R]
prompt_lens:         [B * G]
```

然后构造完整序列 mask：

```python
full_mask = (outputs != tokenizer.pad_token_id).long()
```

shape：

```text
full_mask: [B * G, S]
```

### 13.10 logp_pos：定位 completion token 在完整序列里的位置

训练循环中：

```python
logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(
    completion_ids.size(1),
    device=args.device
).unsqueeze(0)
```

shape：

```text
prompt_lens:              [B * G]
prompt_lens.unsqueeze(1): [B * G, 1]
torch.arange(R):          [R]
unsqueeze(0):             [1, R]
logp_pos:                 [B * G, R]
```

它的作用是：

```text
告诉后面从完整序列 logprob 里取哪些位置，才对应 completion token。
```

为什么有 `prompt_lens - 1`？

因为自回归模型的 logits 是错位预测的：

```text
第 t 个位置的 logits，用来预测第 t+1 个 token。
```

所以 completion 第一个 token 的 logprob，来自 prompt 最后一个位置的 logits。

### 13.11 calculate_rewards：response 文本 -> reward 分数

训练循环中：

```python
rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)
```

这一步经过的模块是：

```text
train_grpo.py -> calculate_rewards
```

输入：

```text
prompts:     list[str], 长度 B
completions: list[str], 长度 B * G
```

输出：

```text
rewards: [B * G]
```

注意：

```text
reward 是每条生成回答一个标量分数。
不是每个 token 一个 reward。
```

### 13.12 grouped_rewards：按 prompt 分组

代码：

```python
grouped_rewards = rewards.view(-1, args.num_generations)
```

shape：

```text
rewards:         [B * G]
grouped_rewards: [B, G]
```

如果：

```text
B = 2
G = 6
```

那么：

```text
[12] -> [2, 6]
```

每一行是同一个 prompt 的 `G` 个回答：

```text
第 0 行：prompt_1 的 6 个 response reward
第 1 行：prompt_2 的 6 个 response reward
```

### 13.13 advantage：组内 reward -> 相对优势

代码：

```python
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

shape：

```text
grouped_rewards: [B, G]

grouped_rewards.mean(dim=1): [B]
repeat_interleave(G):        [B * G]

grouped_rewards.std(dim=1):  [B]
repeat_interleave(G):        [B * G]

advantages:                  [B * G]
```

含义：

```text
advantages[i] 表示第 i 条 response 比同 prompt 组内平均水平好多少。
```

### 13.14 当前 policy 重新 forward

训练循环中会把 rollout 得到的完整序列重新喂给当前 policy：

```python
res = model_unwrapped(outputs, attention_mask=full_mask)
```

这一步经过的模块是：

```text
policy model forward
```

输入：

```text
outputs:   [B * G, S]
full_mask: [B * G, S]
```

输出：

```text
res.logits: [B * G, S, V]
```

随后代码用：

```python
F.log_softmax(res.logits[:, :-1, :], dim=-1)
```

shape：

```text
res.logits:              [B * G, S, V]
res.logits[:, :-1, :]:   [B * G, S - 1, V]
log_softmax 后:          [B * G, S - 1, V]
```

为什么去掉最后一个位置？

```text
因为最后一个位置没有下一个 token 可以预测。
```

### 13.15 gather 真实下一个 token 的 logprob

代码：

```python
.gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)
```

这里和 DPO 的逻辑一样：

```text
logits 里每个位置有 V 个词表分数。
outputs[:, 1:] 是每个位置真正要预测的下一个 token。
gather 按 token id 取出真实 token 的 logprob。
```

shape：

```text
outputs[:, 1:]:                 [B * G, S - 1]
outputs[:, 1:].unsqueeze(-1):   [B * G, S - 1, 1]

gather 后:                      [B * G, S - 1, 1]
squeeze(-1) 后:                 [B * G, S - 1]
```

此时得到的是完整序列每个位置的目标 token logprob：

```text
prompt 部分 + completion 部分
```

### 13.16 gather completion 部分 logprob

接着：

```python
.gather(1, logp_pos)
```

shape：

```text
完整序列 token logprob: [B * G, S - 1]
logp_pos:              [B * G, R]

gather 后:
per_token_logps:       [B * G, R]
```

这一步的作用是：

```text
只取 response/completion 部分的 token logprob。
```

所以：

```text
per_token_logps
= 当前 policy 对这些生成回答中每个 response token 的 logprob
```

### 13.17 reference model 计算 ref_per_token_logps

代码：

```python
with torch.no_grad():
    ref_per_token_logps = F.log_softmax(
        ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :],
        dim=-1
    ).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

这一步经过的模块是：

```text
reference model forward
```

shape 和当前 policy 一样：

```text
ref_per_token_logps: [B * G, R]
```

区别是：

```text
policy model：会训练更新。
reference model：no_grad + frozen，只提供 KL 对照。
```

### 13.18 completion_mask：只保留有效 response token

代码：

```python
completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
...
completion_mask = (...)
```

输出：

```text
completion_mask: [B * G, R]
```

它的作用是：

```text
只让有效 completion token 参与 loss。
pad token 不参与。
eos 之后的 token 不参与。
```

### 13.19 KL、ratio、per_token_loss

KL penalty：

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
```

shape：

```text
ref_per_token_logps: [B * G, R]
per_token_logps:     [B * G, R]
per_token_kl:        [B * G, R]
```

ratio：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
```

shape：

```text
per_token_logps:     [B * G, R]
old_per_token_logps: [B * G, R]
ratio:               [B * G, R]
```

advantage 扩展到 token 维度：

```python
advantages.unsqueeze(1)
```

shape：

```text
advantages:              [B * G]
advantages.unsqueeze(1): [B * G, 1]
```

它会广播到每个 response token：

```text
[B * G, 1] -> [B * G, R]
```

所以每个 response 的所有 token 使用同一个 response-level advantage。

最后得到：

```text
per_token_loss: [B * G, R]
```

### 13.20 token loss -> response loss -> batch loss

代码：

```python
policy_loss = (
    (per_token_loss * completion_mask).sum(dim=1)
    / completion_mask.sum(dim=1).clamp(min=1)
).mean()
```

shape 变化：

```text
per_token_loss:             [B * G, R]
completion_mask:            [B * G, R]

per_token_loss * mask:      [B * G, R]
sum(dim=1):                 [B * G]
completion_mask.sum(dim=1): [B * G]

每条 response 的平均 loss: [B * G]
mean 后:                  scalar
```

然后：

```python
loss = (policy_loss + aux_loss) / args.accumulation_steps
loss.backward()
```

最终：

```text
loss: scalar
```

只有 policy model 会产生梯度并被 optimizer 更新。

### 13.21 optimizer.step 与 rollout_engine.update_policy

梯度累积到指定步数后：

```python
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

这一步更新：

```text
policy model 参数
```

reference model 不更新。

之后代码还会定期：

```python
rollout_engine.update_policy(model)
```

它的作用是：

```text
让 rollout_engine 后续生成 response 时，使用最新 policy。
```

如果使用 torch rollout，引擎内部直接换成当前 model。

如果使用 sglang rollout，则会把当前 policy 权重保存到共享目录，再通知 sglang 服务更新权重。

### 13.22 一张总表

```text
1. JSON 样本
   conversations: list[dict]

2. RLAIFDataset.create_chat_prompt
   conversations[:-1] -> apply_chat_template(add_generation_prompt=True)
   prompt: str

3. DataLoader
   prompts: list[str], 长度 B

4. tokenizer
   input_ids:      [B, P]
   attention_mask: [B, P]

5. rollout repeat_interleave
   input_ids:      [B * G, P]
   attention_mask: [B * G, P]

6. policy.generate
   output_ids: [B * G, S]

7. 切 completion
   completion_ids: [B * G, R]
   completions: list[str], 长度 B * G

8. compute_per_token_logps
   old_per_token_logps: [B * G, R]

9. calculate_rewards
   rewards: [B * G]

10. rewards.view
    grouped_rewards: [B, G]

11. advantage
    advantages: [B * G]

12. policy forward
    logits: [B * G, S, V]

13. gather 目标 token + gather completion 位置
    per_token_logps: [B * G, R]

14. reference forward
    ref_per_token_logps: [B * G, R]

15. KL / ratio / token loss
    per_token_kl:  [B * G, R]
    ratio:         [B * G, R]
    per_token_loss:[B * G, R]

16. completion_mask 加权平均
    response loss: [B * G]
    final loss: scalar

17. backward / optimizer.step
    更新 policy model
```

## 14. GRPO 和 DPO 对比

```text
DPO：
数据集里已经有 chosen/rejected。
训练目标是让 policy 相比 reference 更偏好 chosen。
不需要模型现场生成回答。

GRPO：
数据集里主要是 prompt。
模型先 rollout 生成多个 response。
reward 给 response 打分。
通过组内 advantage 判断哪些 response 应该提高概率。
需要 KL penalty 限制 policy 偏离 reference。
```

一句话总结：

```text
DPO 学的是静态偏好对。
GRPO 学的是模型自己生成结果后的动态 reward 信号。
```

## 15. 常见疑问

### Q1：`model.generate(...)` 得到的 `output_ids` 是 tokenizer 做的吗？

不是。

tokenizer 做的是：

```text
把已有文本转成 token id。
```

`model.generate(...)` 做的是：

```text
根据 prompt_ids，一个 token 一个 token 地生成新的 response token id。
```

所以：

```text
prompt_ids:     [B * G, P]
completion_ids: [B * G, R]
output_ids:     [B * G, P + R]
```

一句话：

```text
tokenizer 是编码已有文本。
generate 是模型现场创造新 token。
```

### Q2：GRPO 为什么不像 SFT 那样直接有 labels？

因为 GRPO 的训练目标不是模仿固定答案，而是：

```text
让模型自己生成回答，再根据 reward 判断哪些回答值得提高概率。
```

SFT 里：

```text
labels 是标准答案 token。
```

GRPO 里：

```text
completion_ids 是模型自己生成的回答。
reward/advantage 是这个回答的质量信号。
```

所以 GRPO 的监督信号不是 label，而是：

```text
advantage * logprob
```

### Q3：reward_model 是如何打分的？

MiniMind 的 reward model 会看到：

```text
用户问题 / 对话上下文
模型生成的 assistant response
```

然后给出一个标量分数。

MiniMind 还会叠加规则分，包括：

```text
长度是否合适。
thinking 格式是否合理。
重复度是否过高。
reward model 自身评分。
```

最后得到：

```text
rewards: [B * G]
```

### Q4：为什么 reward 要按组归一化成 advantage？

因为不同 prompt 难度不同，reward 的绝对值不能直接横向比较。

例如：

```text
prompt A 的回答 reward 普遍在 0-2。
prompt B 的回答 reward 普遍在 8-10。
```

这不一定说明 prompt B 的回答都比 prompt A 好，可能只是问题类型不同、reward model 打分尺度不同。

GRPO 关心的是：

```text
同一个 prompt 下，哪个回答比同组其他回答更好。
```

所以要做：

```text
advantage = (reward - group_mean) / group_std
```

### Q5：ratio 和 KL 为什么看起来很像？

因为它们都来自 logprob 差值。

但比较对象不同：

```text
ratio：
当前 policy vs rollout 时的 old policy。
控制本轮更新幅度。

KL：
当前 policy vs 冻结 reference model。
控制长期行为不要跑偏。
```

所以不是二选一，而是一起进入 loss：

```text
loss = - reward/advantage 相关项 + beta * KL
```

### Q6：从 advantage 到更新参数，中间到底怎么连上？

关键是：

```text
advantage 不能反向传播。
per_token_logps 可以反向传播。
```

所以 GRPO 用 advantage 给 logprob 加权：

```text
advantage > 0：
提高这些 completion token 的 logprob。

advantage < 0：
降低这些 completion token 的 logprob。
```

最后通过：

```python
loss.backward()
```

梯度会从 `per_token_logps` 回到 policy model 参数。

### Q7：为什么一个 response 的 advantage 会作用到所有 token？

因为 reward model 给的是整段 response 的分数，不是每个 token 的分数。

所以代码里：

```python
advantages.unsqueeze(1)
```

会从：

```text
[B * G]
```

变成：

```text
[B * G, 1]
```

再广播到：

```text
[B * G, R]
```

也就是：

```text
整段回答好，这段回答的有效 token 都被鼓励。
整段回答差，这段回答的有效 token 都被压低。
```

### Q8：GRPO 里真正被更新的是谁？

只更新：

```text
policy model
```

不更新：

```text
reference model
reward model
old policy
```

其中：

```text
old policy 不是模型，只是 rollout 时保存下来的 old_per_token_logps。
```

## 16. 现阶段检查题

1. 什么是 rollout？
2. 为什么 GRPO 需要同一个 prompt 生成多个 response？
3. `RolloutResult` 里的 `output_ids` 和 `completion_ids` 有什么区别？
4. `old_per_token_logps` 为什么要保存？
5. reward 和 label 有什么区别？
6. grouped rewards 为什么要 reshape 成 `[B, num_generations]`？
7. advantage 大于 0 和小于 0 分别代表什么？
8. reward 方差太小时，为什么训练信号会弱？
9. policy model、old policy、reference model、reward model 分别是什么？
10. ratio 表示什么？
11. KL penalty 为什么能限制模型跑偏？
12. ratio 和 KL 的区别是什么？
13. GRPO 和 DPO 最大区别是什么？
14. `model.generate(...)` 和 tokenizer 的区别是什么？
15. GRPO 中 assistant 回复的起始位置来自哪里？
16. MiniMind 的最终 reward 由哪几部分组成？

能讲清楚这些，就可以继续深入 `rollout_engine.py` 和 GRPO loss 的逐行实现。
