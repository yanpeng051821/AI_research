# 第 11 周补充：PPO 完整模块与数据流

## 1. 这篇文档解决什么问题

这篇文档只讲 PPO。

它的目标不是推导 PPO 论文，而是让你能把 LLM 里的 PPO 训练流程完整落地到代码：

```text
prompt
-> actor rollout 生成 response
-> reward model 给 response 打分
-> critic/value model 估计 value
-> reward + value 计算 advantage / return
-> 当前 actor 重新计算 new logprob
-> ratio + clip 构造 policy loss
-> value loss 训练 critic
-> KL penalty 限制 actor 偏离 reference
-> 更新 actor 和 critic
```

PPO 的核心问题是：

```text
模型自己生成回答后，如何稳定地提高好回答概率，降低差回答概率？
```

它比 SFT、DPO、GRPO 都更像传统强化学习，因为它有一个明确的 actor-critic 结构：

```text
actor：
负责生成 token。

critic：
负责估计当前状态未来大概能拿多少 reward。
```

在 MiniMind 中，对应源码主要是：

```text
trainer/train_ppo.py
trainer/rollout_engine.py
trainer/trainer_utils.py -> LMForRewardModel
dataset/lm_dataset.py -> RLAIFDataset
```

## 2. PPO 在 LLM 训练阶段里的位置

先把它和前面的阶段串起来：

```text
Pretrain：
让模型学语言续写。
训练信号是 labels。

SFT：
让模型学 assistant 回复格式和回答方式。
训练信号还是 labels，但通常只训练 assistant 部分。

DPO：
让模型学 chosen 比 rejected 更好。
训练信号是偏好对。

PPO：
让模型先自己生成 response。
再用 reward model / 规则给 response 打分。
然后用强化学习方法更新模型。
```

所以 PPO 和 SFT 最大差异是：

```text
SFT：
答案来自数据集。

PPO：
答案来自模型自己 rollout。
```

PPO 和 DPO 最大差异是：

```text
DPO：
数据集提前给 chosen/rejected。

PPO：
模型现场生成 response，再由 reward model 打分。
```

PPO 和 GRPO 最大差异先放一句话：

```text
PPO 用 critic/value model 估计 baseline。
GRPO 用同 prompt 多 response 的组内均值估计 baseline。
```

这篇文档先把 PPO 自己讲顺，最后再单独对比 GRPO。

## 3. PPO 里的四个模型

LLM PPO 通常同时涉及四个模型或模型角色：

```text
actor / policy model
critic / value model
reference model
reward model
```

它们分工完全不同。

### 3.1 Actor / Policy：负责生成

Actor 就是当前正在训练的语言模型。

它的输入是：

```text
prompt + 已经生成的 token
```

它的输出是：

```text
下一个 token 的概率分布
```

在 MiniMind PPO 中：

```python
actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
```

它会：

```text
1. rollout 阶段生成 response。
2. 训练阶段重新计算 response token 的 logprob。
3. 被 optimizer 更新。
```

### 3.2 Critic / Value Model：负责估计预期分

Critic 不生成回答，也不直接评价完整回答质量。

它估计的是：

```text
在当前状态下继续生成，未来大概能拿多少 reward？
```

也就是：

```text
V(s)
```

其中 state `s` 在 LLM 里可以理解成：

```text
prompt + 已经生成的 response 前缀
```

例如：

```text
prompt: 解释什么是机器学习
已生成: 机器学习是
```

critic 要估计：

```text
从这个前缀继续生成下去，最终回答大概能拿多少分？
```

MiniMind PPO 里定义了一个 `CriticModel`：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

关键变化是：

```text
普通语言模型：
hidden_states -> lm_head -> vocab logits

critic：
hidden_states -> value_head -> scalar value
```

所以 critic 输出不是 `[B, T, V]`，而是：

```text
values: [B, T]
```

每个 token 位置都有一个 value 估计。

### 3.2.1 为什么 PPO 需要 Critic，而 GRPO 不需要

这里很容易和 GRPO 串线。

GRPO 的做法更直接：

```text
同一个 prompt 生成多个 response
-> 每个 response 打 reward
-> 在这一组 response 里比较谁更好
-> 高于组均值的 response 被鼓励
-> 低于组均值的 response 被压低
```

所以 GRPO 的 baseline 来自：

```text
同一个 prompt 下多个 response 的组内平均 reward
```

它回答的问题是：

```text
这条回答比同组其他回答更好吗？
```

PPO 的做法不同。

PPO 通常不依赖“同一个 prompt 生成多个 response”来建立参照，而是让 critic 估计每个生成位置的预期收益：

```text
生成到当前位置时，如果继续写下去，最终大概能拿多少分？
```

所以 PPO 的 baseline 来自：

```text
critic 输出的 value
```

它回答的问题是：

```text
这一步 token 选择，比 critic 原本预期的结果更好吗？
```

这也是为什么 PPO 里会出现：

```text
reward + value
-> GAE
-> return / advantage
```

而 GRPO 里更像是：

```text
reward - grouped_reward_mean -> advantage
```

注意一个关键时间顺序：

```text
PPO / GRPO 都会先 rollout。
rollout 会先让 actor.generate(...) 生成 completion_ids。
后面的 reward、logprob、value、KL 都是在这些已经生成出来的 completion_ids 上计算的。
```

所以文档里看到的 completion token 不是原始数据集自带的标准答案，而是 rollout 阶段模型现场生成出来的回答。

### 3.3 Reference Model：负责防跑偏

Reference model 是冻结的基准模型，通常来自 SFT 权重。

MiniMind 中：

```python
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)
```

它的作用是：

```text
计算当前 actor 和 reference 的差异。
通过 KL penalty 防止 actor 为了 reward 跑偏。
```

它不更新。

### 3.4 Reward Model：负责打分

Reward model 是裁判。

它看：

```text
prompt + response
```

输出：

```text
reward score
```

在 MiniMind PPO 中：

```python
reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
rewards = calculate_rewards(prompts, responses_text, reward_model)
```

MiniMind 的 reward 不是纯模型分数，而是：

```text
最终 reward = 规则分 + reward_model 分
```

规则分包括：

```text
回答长度是否合适
thinking 格式是否合理
重复惩罚
```

reward model 不更新。

## 4. PPO 的完整主流程

PPO 最容易混乱的地方，是 rollout 之后会出现很多变量：

```text
reward
value
advantage
return
old_logprob
new_logprob
ratio
ref_logprob
KL
```

不要把它们看成一条直线。

更清晰的理解方式是：

```text
阶段 1：rollout 采样
先让 actor 真正生成一批 response，并固定下来。

阶段 2：围绕同一批 prompt + response 分三条线计算训练材料
reward / critic 线：判断好不好，以及比预期好多少。
policy ratio 线：判断当前 actor 相比 rollout 时的 old actor 改了多少。
KL 线：判断当前 actor 相比 reference model 偏了多少。

阶段 3：loss 汇合与参数更新
三条线的结果汇合成 policy_loss、value_loss、KL penalty，然后只更新 actor 和 critic。
```

所以 PPO 的主线不是：

```text
prompt -> response -> loss
```

而是：

```text
prompt
-> rollout 生成 response
-> 基于同一批 response 分线计算 reward / value / ratio / KL
-> 汇合成 loss
-> 更新 actor + critic
```

完整流程可以先记成这张图：

```text
阶段 1：Rollout 采样，先生成并固定一批数据

prompts
-> tokenizer
-> enc.input_ids [B, P]
-> rollout_engine.rollout 使用当时的 actor 生成：
   gen_out [B, P + R]
   completion_ids [B, R]
   responses_text: list[str], 长度 B
   old_resp_logp [B, R]

阶段 2：Rollout 后、PPO 更新前，计算固定下来的训练材料

reward 线：
responses_text
-> reward_model / reward_rule
-> rewards [B]

old value 线：
gen_out
-> critic(gen_out)，此时不反传
-> old_resp_values [B, R]

reference 线：
gen_out
-> reference model，冻结不反传
-> ref_resp_logp [B, R]

advantage / return：
rewards + old_resp_values
-> advantages [B, R]
-> returns [B, R]

阶段 3：进入 PPO 更新循环，同一批 gen_out 会被重新送进当前 actor / critic

policy ratio 线：
current actor(gen_out[inds])
-> mb_resp_logp [mb, R]
-> mb_resp_logp - old_resp_logp[inds]
-> ratio [mb, R]

policy loss：
ratio + advantages[inds] + clip + KL
-> policy_loss

current value 线：
current critic(gen_out[inds])
-> mb_resp_values [mb, R]

value loss：
mb_resp_values + returns[inds]
-> value_loss

阶段 4：loss 汇合与参数更新

policy_loss + vf_coef * value_loss + aux_loss
-> backward
-> actor_optimizer.step()
-> critic_optimizer.step()

更新后：
actor / critic 参数变了
但这批 rollout 里的 gen_out / old_resp_logp / old_resp_values / ref_resp_logp / advantages / returns 不变
```

对应到 MiniMind 的关键变量：

```text
rollout_result.output_ids        -> gen_out
rollout_result.completion_ids    -> completion_ids
rollout_result.completions       -> responses_text
rollout_result.per_token_logps   -> old_resp_logp

rewards                          -> reward 线结果
values_seq / old_resp_values     -> critic 线结果
advantages / returns             -> reward + value 汇合结果

mb_resp_logp                     -> current actor 的 new logprob
ratio                            -> current actor / old actor 的概率比
ref_resp_logp                    -> reference model 的 logprob
kl_ref_penalty                   -> KL 约束项

policy_loss / value_loss / loss  -> 最终训练目标
```

这里最容易混的是 old 和 current。

可以这样固定：

```text
old_resp_logp：
rollout 生成 response 时保存下来的 actor logprob。
在同一批 rollout 的 PPO 更新循环里一直不变。
用于 ratio 的分母。

mb_resp_logp：
PPO 更新循环里，当前 actor 对同一批 gen_out 重新 forward 得到的 logprob。
会参与反传。
随着 actor_optimizer.step() 后 actor 参数变化，下一次重新 forward 可能会变。

old_resp_values：
rollout 后、更新前，critic 对这批 gen_out 算出的旧 value。
不反传，在这批 PPO 更新中固定。
用于计算 advantages / returns，也用于 value clip。

mb_resp_values：
PPO 更新循环里，当前 critic 对同一批 gen_out 重新 forward 得到的 value。
会参与反传。
用于 value_loss。

ref_resp_logp：
reference model 对这批 gen_out 的 logprob。
reference model 冻结，所以它也是固定对照。
用于 KL penalty。
```

一句话：

```text
gen_out 是同一批生成结果。
old 是这批结果生成时或更新前保存下来的固定值。
current / mb 是 PPO 更新循环里当前模型重新算出来、会反传的值。
```

## 5. 阅读后续代码前先统一符号

后面只沿一条执行主线展开。先固定 shape 符号：

```text
B：batch size
P：padding 后的 prompt 长度
R：padding 后的 response 长度
S：完整序列长度，S = P + R
C：hidden size
V：vocab size
mb：PPO 更新阶段的 mini-batch size
```

最常见的张量：

```text
prompt_ids:       [B, P]
completion_ids:   [B, R]
gen_out:          [B, S]
actor logits:     [B, S, V]
values_seq:       [B, S]
response logprob: [B, R]
advantages:       [B, R]
returns:          [B, R]
```

还要固定 LLM 强化学习里的状态和动作：

```text
状态 s_t：prompt + token_t 之前已经生成的 response 前缀
动作 a_t：在状态 s_t 下选择 response token_t
```

假设 response 是 `[A, B, C]`：

```text
动作 A 对应状态：prompt
动作 B 对应状态：prompt + A
动作 C 对应状态：prompt + A + B
```

因此，Actor 在相应前缀位置输出下一个 token 的概率，Critic 在同一个前缀位置输出未来累计回报的预期。二者可以使用相似的 Transformer backbone，但 head 和训练目标不同。

后续章节只按下面一次主线展开：

```text
第 6 节：rollout，生成并固定训练样本
第 7～9 节：reward / critic / GAE 数据线
第 10～13 节：new logprob / ratio / clip 数据线
第 14 节：reference / KL 数据线
第 15～17 节：value loss、总 loss 与更新
第 18 节：统一检查所有 shape
```

## 6. 阶段 A：rollout 生成 response

训练循环先拿到 prompt：

```python
prompts = batch["prompt"]
enc = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=args.max_seq_len,
    padding_side="left",
).to(args.device)
```

此时：

```text
input_ids:      [B, P]
attention_mask: [B, P]
```

然后 actor 进行 rollout：

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=enc.input_ids,
    attention_mask=enc.attention_mask,
    num_generations=1,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

PPO 通常每个 prompt 生成一个 response：

```text
num_generations = 1
```

这和 GRPO 很不同，GRPO 的核心是同一个 prompt 生成多个 response 做组内比较。

rollout 后得到：

```python
gen_out = rollout_result.output_ids
completion_ids = rollout_result.completion_ids
prompt_lens = rollout_result.prompt_lens.to(args.device)
responses_text = rollout_result.completions
old_resp_logp = rollout_result.per_token_logps.to(args.device)
```

shape：

```text
gen_out:         [B, P + R]
completion_ids: [B, R]
old_resp_logp:  [B, R]
prompt_lens:    [B]
responses_text: list[str]，长度 B
```

这里最关键的是：

```text
old_resp_logp
```

它记录的是：

```text
rollout 生成这些 response token 的时候，当时 actor 对这些 token 的 logprob。
```

后面更新 actor 时，会拿当前 actor 的新 logprob 和它比较。

## 7. reward：完整 response 的评分信号

rollout 得到 response 文本后，计算 reward：

```python
rewards = calculate_rewards(prompts, responses_text, reward_model)
```

shape：

```text
rewards: [B]
```

注意：

```text
reward 是 response 级别的标量。
```

也就是：

```text
一条 response 一个 reward。
```

不是：

```text
每个 token 一个 reward。
```

MiniMind PPO 后面会把这个 response-level reward 放到最后一个有效 response token 上：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]
```

shape：

```text
old_resp_logp: [B, R]
token_rewards: [B, R]
```

直觉：

```text
整段回答最终拿到一个分数。
这个分数作为生成完这段 response 后得到的终局奖励。
```

## 8. critic：估计每个状态的 value

先把一个容易沿用语言模型思维而产生的误区拆开：

```text
Transformer 的 hidden state 本身不等于“下一个 token 预测”。
hidden state 最终表示什么，取决于它后面接什么 head、使用什么训练目标。
```

同一个 `gen_out [B, S]` 分别送入 actor 和 critic 时：

```text
gen_out [B, S]
-> Transformer
-> hidden_states [B, S, C]

actor：
hidden_states -> lm_head(C -> V) -> logits [B, S, V]
目标：预测下一个 token。

critic：
hidden_states -> value_head(C -> 1) -> values_seq [B, S]
目标：预测每个前缀状态的未来累计回报。
```

所以 critic 虽然也读取 token 序列，但它不预测下一个 token。它使用不同的输出头和训练目标，输出的每个位置只有一个标量 value。

PPO 不直接只看 reward 更新 actor。

它还要问：

```text
这个 reward 是比预期好，还是比预期差？
```

所以需要 critic 估计 value：

```python
values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
```

shape：

```text
values_seq:      [B, P + R]
old_resp_values: [B, R]
```

`values_seq` 是完整序列每个前缀位置的 value。

但 PPO 只关心 response 中由 actor 采取的动作，所以用 `logp_pos` 取出每个 response token **生成之前** 的状态价值：

```text
old_resp_values: [B, R]
```

假设 response 是 `[A, B, C, D]`，对齐关系是：

```text
要评价的动作 A：状态 s0 = prompt                  -> V(s0)
要评价的动作 B：状态 s1 = prompt + A              -> V(s1)
要评价的动作 C：状态 s2 = prompt + A + B          -> V(s2)
要评价的动作 D：状态 s3 = prompt + A + B + C      -> V(s3)
```

因此：

```text
old_resp_values[:, 0] = V(prompt)
old_resp_values[:, 1] = V(prompt + A)
old_resp_values[:, 2] = V(prompt + A + B)
old_resp_values[:, 3] = V(prompt + A + B + C)
```

这里必须注意位置偏移：第一个 response token `A` 位于序列位置 `P`，但生成 `A` 之前的状态表示位于 `P - 1`。这也是 `logp_pos` 中要减一的原因。

`V(s0)` 也不是“生成 A 的预期分数”。在还没有生成 A 时，下一个 token 可能是 A、B、C 等多个选择，所以 `V(s0)` 是按照当前 policy 对所有可能后续求出的平均预期。真正采样出 A 后，状态变成 `s1 = prompt + A`，`V(s1)` 才是在已经确定 A 的条件下对剩余未来的预期。

## 9. advantage 和 return

### 9.1 为什么需要 advantage

如果只看 reward，会有问题：

```text
简单问题 reward = 8
困难问题 reward = 5
```

不能直接说困难问题回答更差。

PPO 更关心：

```text
本次 rollout 得到的回报比预期好多少？
```

这就是 advantage 的核心含义：

```text
advantage = 本次回报 - 预期回报
```

在最简化、暂时忽略 GAE 递推的理解里：

```text
advantage = reward - value
```

其中：

```text
reward：
本次完整 response 生成后得到的评分信号。
它是 reward model / 规则给出的训练依据，不代表绝对正确的客观真值。

value：
critic 估计的预期分数。
```

含义：

```text
advantage_t > 0：
这次在状态 s_t 选择 token_t 后带来的结果超过当时预期，应该提高 P(token_t | s_t)。

advantage_t < 0：
这次选择 token_t 后带来的结果低于当时预期，应该降低 P(token_t | s_t)。

advantage ≈ 0：
差不多，训练信号弱。
```

### 9.2 MiniMind 中的 GAE

MiniMind PPO 用的是 GAE 风格的递推：

```python
gen_len = old_resp_values.size(1)
lastgaelam = torch.zeros(B, device=args.device)
advs_rev = []

for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)

advantages = torch.stack(advs_rev[::-1], dim=1)
returns = advantages + old_resp_values
```

对应公式：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
Return_t = A_t + V(s_t)
```

每个变量在 LLM 生成中的实际含义：

```text
s_t：
生成第 t 个 response token 之前的前缀状态。

token_t：
actor 在状态 s_t 下选中的动作。

r_t：
执行 token_t 这一步收到的即时 reward。
正文位置通常只有逐 token KL 奖惩，完整 response 的最终评分通常放在最后一个有效 token。

V(s_t)：
还没有选择 token_t 时，critic 对未来累计回报的平均预期。

V(s_{t+1})：
已经选择 token_t、前缀变长之后，critic 对剩余未来累计回报的新预期。

delta_t：
选择 token_t 后得到的“即时 reward + 新的未来预期”，相比选择前的旧预期高多少。

A_t：
把当前 delta 和后续若干步的 delta 按距离衰减后合并，得到 actor 使用的 token 级相对优势。

Return_t：
结合 rollout 结果得到的 value 学习目标，供 critic 回归。
```

为什么 `V(s_{t+1})` 不包含刚生成的 token，仍然能帮助评价这个 token？因为一步动作的完整目标本来就是：

```text
当前动作的即时收益 r_t + 进入新状态后的未来收益 gamma * V(s_{t+1})
```

token 本身改变了后续前缀。即使 `V(s_{t+1})` 只预测 token 之后的未来，它也是在“已经选择了这个 token”的条件下预测，因此包含这个选择对后续回答质量造成的影响。

例如：

```text
s0 = prompt
V(s0) = 3.9                 # 尚未确定下一个 token 时，对各种可能后续的平均预期

实际选择 A，进入 s1
即时 reward r_A = 0
V(s1) = 8.0                 # 已知选择 A 后，对剩余未来的条件预期

delta_A = 0 + 8.0 - 3.9 = 4.1
```

这个正 delta 表示选择 A 后，未来收益预期明显高于选择前的平均基线。GAE 再把后续位置的 delta 一起折扣传播回来，得到更稳定的 `advantage_A`。

shape：

```text
token_rewards:    [B, R]
old_resp_values:  [B, R]
advantages:       [B, R]
returns:          [B, R]
```

最后抓住训练用途：

```text
advantages：
告诉 actor，本次在对应前缀下选择该 token 后的结果，相比当前 policy 的平均预期更好还是更差。

returns：
critic 应该学习逼近的目标。
```

也就是：

```text
actor 用 advantages 训练。
critic 用 returns 训练。
```

需要保留一个边界：advantage 是基于采样轨迹、reward model 和 critic 得到的信用分配估计。它说明“这个选择之后的结果相对基线更好或更差”，但不能严格证明某个 token 对最终 reward 具有独立的因果贡献。

### 9.3 advantage 标准化

MiniMind 还会对 advantage 做标准化：

```python
adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

作用：

```text
让 advantage 的尺度更稳定。
避免某一批 reward 尺度过大导致更新过猛。
```

## 10. 阶段 B：PPO 多轮更新

PPO 的一个特点是：

```text
同一批 rollout 数据，会重复更新多轮。
```

MiniMind 中：

```python
for ppo_epoch in range(args.ppo_update_iters):
    b_inds = torch.randperm(B, device=args.device)
    for i in range(0, B, mb_size):
        inds = b_inds[i:i + mb_size]
        ...
```

也就是：

```text
rollout 一次。
把这批数据切成 mini-batch。
对 actor/critic 更新 ppo_update_iters 轮。
```

为什么可以重复更新？

因为 rollout 很贵，生成 response 需要自回归。

但不能无限重复更新，因为数据来自旧 policy，更新太多会让当前 policy 和旧 policy 差太远。

这就是 PPO 需要：

```text
ratio
clip
approx_kl early stop
```

## 11. 当前 actor 重新计算 new logprob

PPO 更新时，会把同一批 `gen_out` 再喂给当前 actor：

```python
res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1) \
    .gather(2, labels[inds].unsqueeze(-1)) \
    .squeeze(-1) \
    .gather(1, logp_pos[inds])
```

shape：

```text
logits:        [mb, P + R, V]
mb_resp_logp:  [mb, R]
```

这里的 `mb_resp_logp` 是：

```text
当前 actor 对同一批 response token 的新 logprob。
```

它要和 rollout 时保存的旧 logprob 比较：

```text
old_resp_logp: [B, R]
```

## 12. ratio：当前 policy 和 old policy 的概率比

PPO 计算：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
ratio = torch.exp(log_ratio)
```

因为：

```text
mb_resp_logp = log 当前 actor 概率
old_resp_logp = log rollout 时旧 actor 概率
```

所以：

```text
ratio = exp(log 当前概率 - log 旧概率)
      = 当前概率 / 旧概率
```

含义：

```text
ratio > 1：
当前 actor 比 rollout 时更愿意生成这个 token。

ratio < 1：
当前 actor 比 rollout 时更不愿意生成这个 token。
```

ratio 要和 advantage 一起看：

```text
advantage > 0：
希望 ratio 变大，提高好路径概率。

advantage < 0：
希望 ratio 变小，降低差路径概率。
```

## 13. PPO 为什么要 clip

如果没有限制，actor 可能因为一次 reward 信号就更新太猛。

例如某个 response reward 很高，actor 可能过度提高它的概率，导致：

```text
模型行为突然偏移
输出变得模式化
原有能力被破坏
训练不稳定
```

PPO 使用 clip 控制更新幅度：

```python
torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
```

如果：

```text
clip_epsilon = 0.2
```

那么 ratio 的有效范围大致是：

```text
[0.8, 1.2]
```

MiniMind policy loss：

```python
policy_loss = (
    (
        torch.max(
            -advantages[inds] * ratio,
            -advantages[inds] * torch.clamp(
                ratio,
                1.0 - args.clip_epsilon,
                1.0 + args.clip_epsilon
            )
        )
        * resp_policy_mask[inds]
    ).sum() / resp_policy_mask[inds].sum().clamp(min=1)
    + args.kl_coef * kl_ref_penalty
)
```

不用被 `max` 绕晕。

它等价于 PPO 常见写法的负号形式：

```text
最大化：
min(ratio * advantage, clipped_ratio * advantage)

训练时最小化 loss：
-min(...)
```

直觉：

```text
advantage 决定方向。
ratio 表示当前 policy 相对 old policy 变化了多少。
clip 限制变化不要太猛。
```

## 14. KL reference penalty

除了 clip，MiniMind PPO 还用 reference model 做 KL 约束：

```python
kl_ref_penalty = (
    (
        torch.exp(ref_resp_logp[inds] - mb_resp_logp)
        - (ref_resp_logp[inds] - mb_resp_logp)
        - 1.0
    )
    * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

shape：

```text
ref_resp_logp: [B, R]
mb_resp_logp:  [mb, R]
```

它限制的是：

```text
当前 actor 不要偏离 reference model 太远。
```

这里要分清两个“旧”：

```text
old policy：
rollout 生成 response 时的 actor。
用 old_resp_logp 固化下来。
用于 ratio。

reference model：
冻结的 SFT 基准模型。
用于 KL penalty。
```

它们不是一回事。

可以这样记：

```text
ratio / old policy：
管这次 PPO 更新相对 rollout 时不要变化太猛。

KL / reference model：
管整体训练不要偏离 SFT 基准模型太远。
```

## 15. value loss：训练 critic

Actor 用 policy loss 更新。

Critic 用 value loss 更新。

critic 当前输出：

```python
mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])
```

shape：

```text
mb_resp_values: [mb, R]
returns:        [B, R]
```

MiniMind 的 value loss：

```python
value_loss = 0.5 * (
    torch.max(
        (mb_resp_values - returns[inds]) ** 2,
        (
            torch.clamp(
                mb_resp_values,
                old_resp_values[inds] - args.cliprange_value,
                old_resp_values[inds] + args.cliprange_value
            )
            - returns[inds]
        ) ** 2
    )
    * resp_value_mask[inds]
).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

直觉：

```text
critic 预测的 value 要接近 returns。
```

为什么 value 也 clip？

因为 critic 更新太猛也会让 advantage 估计不稳定。

所以：

```text
policy clip：
限制 actor 更新幅度。

value clip：
限制 critic 更新幅度。
```

## 16. approx_kl 和 early stop

PPO 还会监控当前 actor 和 old policy 的变化：

```python
approx_kl = (
    0.5 * (log_ratio ** 2) * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

如果变化太大：

```python
if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
```

含义：

```text
如果当前 actor 已经离 rollout 时的 old policy 太远，就停止继续用这批 rollout 数据更新。
```

这是因为：

```text
PPO 是 on-policy / 近似 on-policy。
rollout 数据来自旧 policy。
如果当前 policy 已经变化太大，这批旧数据就不再可靠。
```

## 17. 总 loss 和更新对象

MiniMind PPO 的核心 loss：

```python
loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
loss.backward()
```

其中：

```text
policy_loss：
更新 actor，让高 advantage token 概率上升，低 advantage token 概率下降。

value_loss：
更新 critic，让 value 估计更接近 returns。

aux_loss：
MoE 相关辅助 loss，非 MoE 时为 0。
```

随后：

```python
clip_grad_norm_(actor_model.parameters(), args.grad_clip)
clip_grad_norm_(critic_model.parameters(), args.grad_clip)
actor_optimizer.step()
critic_optimizer.step()
actor_scheduler.step()
critic_scheduler.step()
actor_optimizer.zero_grad()
critic_optimizer.zero_grad()
```

更新：

```text
actor_model
critic_model
```

冻结：

```text
ref_model
reward_model
```

## 18. PPO 完整 shape 总表

约定：

```text
B：batch size
P：prompt token 长度
R：response token 长度
S：完整序列长度，S = P + R
V：vocab size
mb：PPO mini-batch size
```

完整数据流：

```text
阶段 0：DataLoader / tokenizer
prompts:        list[str]，长度 B
input_ids:      [B, P]
attention_mask: [B, P]

阶段 1：rollout 采样产物
gen_out:          [B, S]
completion_ids:  [B, R]
completion_mask: [B, R]
prompt_lens:     [B]
responses_text:  list[str]，长度 B
old_resp_logp:   [B, R]

阶段 1.5：从完整序列定位 response token
full_mask:        [B, S]
labels:           [B, S - 1]
resp_idx:         [B, R] 或可广播到 [B, R]
logp_pos:         [B, R]
resp_policy_mask: [B, R]
resp_value_mask:  [B, R]

线 1：reward / critic / advantage
rewards:          [B]
token_rewards:    [B, R]
values_seq:       [B, S]
old_resp_values:  [B, R]
advantages:       [B, R]
returns:          [B, R]

线 2：policy ratio
inds:             [mb]
mb_gen_out:       [mb, S]
actor logits:     [mb, S, V]
new_resp_logp:    [mb, R]
old_resp_logp_mb: [mb, R]
ratio:            [mb, R]

线 3：KL reference
ref logits:       [B, S, V]
ref_resp_logp:    [B, R]
ref_resp_logp_mb: [mb, R]
kl_ref_penalty:   scalar

loss 汇合
current critic values: [mb, R]
policy_loss:           scalar
value_loss:            scalar
total loss:            scalar
```

## 19. PPO 和 GRPO 的边界

为了避免串线，最后单独放 PPO 和 GRPO 的区别。

共同部分：

```text
都有 actor / policy。
都有 rollout。
都有 reward model 或 reward 规则。
都有 old logprob。
都有 ratio。
都可以有 KL / reference model。
都用 advantage 指导提高或降低 response token 的概率。
```

PPO 特有：

```text
有 critic / value model。
critic 输出 values。
用 reward + value 计算 advantages / returns。
有 value loss。
同时更新 actor 和 critic。
通常每个 prompt 生成 1 个 response。
```

GRPO 特有：

```text
没有单独 critic。
同一个 prompt 生成多个 response。
用 grouped_rewards 按 prompt 分组。
用组内 mean/std 计算 advantage。
只更新 policy。
```

一句话区分：

```text
PPO 的 advantage 来自 critic/value。
GRPO 的 advantage 来自同 prompt 多 response 的组内相对 reward。
```

再具象一点：

```text
PPO 问的是：
这一步 token 的选择，是否比 critic 对未来收益的预期更好？

GRPO 问的是：
这条 response 的整体表现，是否比同一个 prompt 下其他 response 更好？
```

所以 PPO 需要 value model 预测“继续写下去的预期分”，GRPO 通常不需要。

再换成流程：

```text
PPO：
prompt
-> actor 生成 response
-> reward model 打分
-> critic 估计 value
-> reward/value 得到 advantage 和 return
-> ratio + clip + KL
-> policy loss 更新 actor
-> value loss 更新 critic

GRPO：
prompt
-> policy 生成多个 response
-> reward model / 规则打分
-> grouped_rewards 组内比较
-> 得到 advantage
-> ratio + KL
-> 只更新 policy
```

## 20. PPO 易混淆速查

这一节只回答读完主线后仍然容易串线的问题。完整推导以前文为准。

### 20.1 Reward、Value、Return、Advantage 分别是什么

```text
Reward：
完整 response 生成后，reward model / 规则给出的已观测评分信号。

Value V(s_t)：
生成 token_t 之前，Critic 对当前前缀未来累计回报的平均预期。

Return_t：
结合本次 rollout 的 reward、逐 token 奖惩和 GAE 得到的 Critic 学习目标。

Advantage_t = Return_t - V(s_t)：
本次选择 token_t 后得到的结果，相比状态 s_t 下原本平均预期好多少。
```

Reward 是当前训练采用的评分信号，不是绝对正确的客观真值；Reward Model 本身也可能产生误判。

### 20.2 Value 为什么不是下一个 token 预测

Transformer 只负责得到每个前缀的表示：

```text
hidden_states: [B, S, C]
```

输出含义由 head 和训练目标决定：

```text
Actor：
hidden_states -> lm_head(C -> V) -> logits [B, S, V]
预测下一个 token。

Critic：
hidden_states -> value_head(C -> 1) -> values_seq [B, S]
预测每个前缀状态的未来累计回报。
```

Critic 读取完整 `gen_out`，不代表它在每个位置看到了未来 token。由于 causal mask，位置 `t` 的 hidden state 只能编码截至该位置的前缀。

### 20.3 V(s_t) 和 V(s_{t+1}) 为什么可以比较

假设在状态 `s_t` 下选择 token `A`，进入 `s_{t+1}`：

```text
V(s_t)：
尚未确定 A 时，对当前 policy 各种可能后续的平均预期。

V(s_{t+1})：
已经确定选择 A 后，对剩余未来的条件预期。
```

一步动作的目标是：

```text
r_t + gamma * V(s_{t+1})
```

因此 TD error 为：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

`V(s_{t+1})` 虽然不再包含生成 A 这个动作本身，却包含“选择 A 后进入的新前缀”对后续结果造成的影响；A 当下的奖励或惩罚由 `r_t` 表示。

### 20.4 Advantage 到底如何训练 Actor

这是 Actor 更新最重要的方向信号：

```text
advantage_t > 0：
本次在 s_t 下选择 token_t 后的结果超过当时预期，
提高 P(token_t | s_t)。

advantage_t < 0：
本次选择 token_t 后的结果低于当时预期，
降低 P(token_t | s_t)。
```

Advantage 是根据采样轨迹、Reward Model 和 Critic 做出的信用分配估计。它不能严格证明某个 token 对最终 Reward 具有独立的因果贡献。

### 20.5 Return 和 Advantage 为什么不相等

二者关系是：

```text
Advantage_t = Return_t - V(s_t)
Return_t = Advantage_t + V(s_t)
```

用途不同：

```text
Actor 使用 Advantage：决定 token 概率提高还是降低。
Critic 使用 Return：通过 value loss 学习更准确的 V(s_t)。
```

### 20.6 Critic 会不会永远拟合 old policy

每轮 rollout 的 response 确实来自当时的 Actor，但 Critic 学的是这些轨迹得到的 Return，不是 old policy 的 token 概率。

```text
actor_v1 rollout -> 训练 Actor / Critic
actor_v2 rollout -> 得到新 response、reward 和 return
actor_v3 rollout -> 继续刷新训练分布
```

PPO 只在同一批 rollout 上更新有限轮，然后重新 rollout，因此 Critic 会随当前 Actor 附近的数据分布持续更新。

### 20.7 Ratio 为什么不是自己除自己

rollout 刚结束时：

```text
new_logprob ≈ old_logprob
ratio ≈ 1
```

但 `old_logprob` 会被冻结保存，而 PPO 会在同一批数据上更新 Actor 多轮。更新后：

```text
ratio = exp(new_logprob - old_logprob)
```

表示当前 Actor 相比 rollout 时的 Actor，对同一个已生成 token 的概率改变了多少：

```text
ratio > 1：当前 Actor 更愿意生成该 token。
ratio < 1：当前 Actor 更不愿意生成该 token。
```

Ratio 只描述概率变化幅度；Advantage 决定变化方向；clip 限制变化不要过大。

## 21. train_ppo.py 真实变量地图

这一节把上面的 PPO 概念直接映射到 MiniMind 的 `trainer/train_ppo.py`。

### 21.1 模型初始化

MiniMind PPO 初始化了四类模型：

```python
actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)
critic_model = critic_model.to(args.device)

reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
```

对应关系：

```text
actor_model：
当前要训练的策略模型，负责生成和更新。

ref_model：
冻结的 SFT / base 锚点模型，只负责 KL 约束。

critic_model：
value model，负责输出 values_seq。

reward_model：
外部打分模型，只负责给 response 打 reward。
```

只有：

```text
actor_model
critic_model
```

会被 optimizer 更新。

### 21.2 rollout 阶段变量

代码：

```python
prompts = batch["prompt"]
enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_seq_len, padding_side="left").to(args.device)

rollout_result = rollout_engine.rollout(
    prompt_ids=enc.input_ids,
    attention_mask=enc.attention_mask,
    num_generations=1,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

变量含义：

```text
prompts:
原始 prompt 文本，list[str]，长度 B。

enc.input_ids:
prompt token ids，[B, P]。

enc.attention_mask:
prompt padding mask，[B, P]。

rollout_result:
actor 自回归生成 response 后的结果包。
```

rollout 产物：

```python
gen_out = rollout_result.output_ids
completion_ids = rollout_result.completion_ids
prompt_lens = rollout_result.prompt_lens.to(args.device)
responses_text = rollout_result.completions
old_resp_logp = rollout_result.per_token_logps.to(args.device)
```

对应：

```text
gen_out:
prompt + response 的完整 token ids，[B, P + R]。

completion_ids:
只包含 response 的 token ids，[B, R]。

prompt_lens:
每条样本真实 prompt 长度，[B]。

responses_text:
解码后的 response 文本，list[str]，长度 B。

old_resp_logp:
rollout 时 old actor 对 response token 的 logprob，[B, R]。
```

这里要牢记：

```text
completion_ids 不是数据集 label。
它是 actor 现场生成出来的 assistant response。
```

### 21.3 response 位置和 mask

代码：

```python
full_mask = (gen_out != tokenizer.pad_token_id).long()
labels = gen_out[:, 1:].clone()
B = len(prompts)
resp_labels = completion_ids
resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
```

shape：

```text
full_mask:     [B, P + R]
labels:        [B, P + R - 1]
resp_labels:   [B, R]
resp_idx:      [1, R]
logp_pos:      [B, R]
resp_pad_mask: [B, R]
```

为什么有 `logp_pos`？

因为语言模型输出位置 `t` 的 logits，用来预测位置 `t + 1` 的 token。

所以 response 第一个 token 的概率，来自：

```text
prompt 最后一个 token 位置的 logits
```

这就是：

```python
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
```

### 21.4 reward / critic / advantage 线

第一步，reward model 给完整 response 打分：

```python
rewards = calculate_rewards(prompts, responses_text, reward_model)
```

shape：

```text
rewards: [B]
```

第二步，critic 输出完整序列每个位置的 value：

```python
values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
```

shape：

```text
values_seq:      [B, P + R]
old_resp_values: [B, R]
```

第三步，把完整 response 的 reward 放到最后一个有效 response token 上：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]
```

shape：

```text
token_rewards: [B, R]
```

第四步，GAE 从后往前算 advantage 和 return：

```python
delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
lastgaelam = delta + args.gamma * args.lam * lastgaelam
advantages = torch.stack(advs_rev[::-1], dim=1)
returns = advantages + old_resp_values
```

shape：

```text
advantages: [B, R]
returns:    [B, R]
```

这一条线的目标：

```text
把完整 response 的 reward，转成每个 response token 的训练信号。

advantages 给 actor 用。
returns 给 critic 用。
```

### 21.5 ratio / policy loss 线

PPO 更新时，当前 actor 重新计算同一批 response token 的 logprob：

```python
res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1) \
    .gather(2, labels[inds].unsqueeze(-1)) \
    .squeeze(-1) \
    .gather(1, logp_pos[inds])
```

shape：

```text
res.logits:    [mb, P + R, V]
mb_resp_logp:  [mb, R]
```

然后和 rollout 时保存的 old logprob 算 ratio：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
ratio = torch.exp(log_ratio)
```

shape：

```text
old_resp_logp[inds]: [mb, R]
log_ratio:           [mb, R]
ratio:               [mb, R]
```

这一条线的目标：

```text
衡量当前 actor 相比 old actor，
对同一批 response token 的概率改了多少。
```

### 21.6 KL / reference 线

reference model 在 rollout 后也对同一条 `gen_out` 打分：

```python
ref_resp_logp = F.log_softmax(
    ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1],
    dim=-1
).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

shape：

```text
ref_resp_logp: [B, R]
```

PPO 更新时：

```python
kl_ref_penalty = (
    (
        torch.exp(ref_resp_logp[inds] - mb_resp_logp)
        - (ref_resp_logp[inds] - mb_resp_logp)
        - 1.0
    )
    * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

这一条线的目标：

```text
限制当前 actor 不要为了 reward model 分数，
偏离冻结的 SFT / base reference 太远。
```

### 21.7 actor loss 和 critic loss 汇合

actor 的 policy loss：

```python
policy_loss = (
    (
        torch.max(
            -advantages[inds] * ratio,
            -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
        )
        * resp_policy_mask[inds]
    ).sum() / resp_policy_mask[inds].sum().clamp(min=1)
    + args.kl_coef * kl_ref_penalty
)
```

critic 的 value loss：

```python
mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

value_loss = 0.5 * (
    torch.max(
        (mb_resp_values - returns[inds]) ** 2,
        (
            torch.clamp(
                mb_resp_values,
                old_resp_values[inds] - args.cliprange_value,
                old_resp_values[inds] + args.cliprange_value
            )
            - returns[inds]
        ) ** 2
    )
    * resp_value_mask[inds]
).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

最后：

```python
loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
loss.backward()
```

注意：

```text
MiniMind 的 policy_loss 里面已经加了 kl_coef * kl_ref_penalty。

所以总 loss 不要再额外加一次 KL。
```

更新对象：

```python
actor_optimizer.step()
critic_optimizer.step()
```

冻结对象：

```text
ref_model 不更新。
reward_model 不更新。
```

### 21.8 一句话变量链路

把 MiniMind PPO 串成一句话就是：

```text
prompts
-> tokenizer 得到 enc.input_ids [B, P]
-> rollout_engine.rollout 得到 gen_out [B, P + R] / completion_ids [B, R] / old_resp_logp [B, R]
-> reward_model 得到 rewards [B]
-> critic(gen_out) 得到 old_resp_values [B, R]
-> rewards + old_resp_values 得到 advantages [B, R] / returns [B, R]
-> current actor(gen_out) 得到 mb_resp_logp [mb, R]
-> mb_resp_logp - old_resp_logp 得到 ratio [mb, R]
-> reference model 得到 ref_resp_logp [B, R] 并构造 KL
-> advantage + ratio + clip + KL 得到 policy_loss
-> current critic values + returns 得到 value_loss
-> 更新 actor 和 critic
```

## 22. DPO、GRPO、PPO 横向对照：同样在优化回答，训练信号完全不同

这三个阶段都不是在学习“唯一标准答案”，而是在让模型的回答更符合人类偏好。

但它们拿到的训练数据、好坏信号和优化方式不同。最容易记住的总览是：

```text
DPO：数据集已经告诉你谁好谁坏。

GRPO：模型现场生成多份回答，在同组里比较谁更好。

PPO：模型现场生成回答，由 reward model 打分，并由 critic 估计预期收益。
```

### 22.1 一张表分清三者

| 维度 | DPO | GRPO | PPO |
| --- | --- | --- | --- |
| 训练前已有的数据 | `prompt + chosen + rejected` | 主要是 `prompt` | 主要是 `prompt` |
| response 从哪里来 | 数据集直接提供 | 当前 policy rollout 生成多条 | 当前 actor rollout 生成 |
| 是否现场生成 | 否 | 是 | 是 |
| 好坏信号 | `chosen` 优于 `rejected` 的相对关系 | reward model / 规则分数的组内比较 | reward model / 规则分数 |
| advantage / baseline | 没有显式 advantage | 同 prompt 多个 response 的组均值 | critic 对每个前缀的 value 估计 |
| 是否需要 critic | 否 | 否 | 是 |
| 是否需要 reference model | 是 | 是 | 是 |
| reference 的作用 | 衡量 chosen/rejected 偏好相对基座的变化 | KL 约束，防止 policy 学歪 | KL 约束，防止 actor 学歪 |
| 主要更新对象 | policy | policy | actor + critic |
| 训练复杂度 | 较低 | 中等 | 最高 |

注意：这里的 `policy`、`actor` 本质上都是“当前正在训练的语言模型”。PPO 只是沿用强化学习的术语，把它称为 actor。

### 22.2 三条最短数据流

```text
DPO：
prompt + chosen/rejected
-> policy/ref 分别计算两种回答的 logprob
-> 让 policy 更偏向 chosen、远离 rejected
-> 更新 policy
```

```text
GRPO：
prompt
-> policy rollout 同题多答
-> reward model / 规则分别打分
-> 同组 reward 归一化得到 advantage
-> 当前 policy 计算 new logprob，ref 提供 KL 约束
-> 更新 policy
```

```text
PPO：
prompt
-> actor rollout 得到 response、old logprob
-> reward model 给完整 response 打分
-> critic 给每个 response 前缀估计 value
-> GAE 得到 token-level advantage / return
-> actor 的 ratio + clip + KL 得到 policy loss
-> critic 的 value 与 return 得到 value loss
-> 更新 actor + critic
```

### 22.3 DPO 为什么最像“离线偏好学习”

DPO 的 `chosen` 和 `rejected` 已经被人或其他模型准备好。训练时模型并不需要自己生成新回答，也不需要给回答打一个绝对 reward。

它只学习一个排序关系：

```text
同一个 prompt 下，chosen 的概率应该高于 rejected。
```

所以 DPO 很适合已有高质量偏好对的场景：实现相对简单、训练稳定、成本较低；但它无法直接利用“模型当前最常犯的错误”，因为训练答案不是当前模型现场生成的。

### 22.4 GRPO 为什么介于 DPO 和 PPO 之间

GRPO 已经进入在线强化学习：response 来自当前 policy 的 rollout，而不是静态数据集。

但它不训练 critic。它用同一个 prompt 的多份答案建立临时参照：

```text
advantage_i = reward_i - 同组平均 reward
```

因此它问的是：

```text
这份回答相对同题其他回答，是更好还是更差？
```

它用更多 rollout 样本换掉 critic 的价值估计，工程上比 PPO 少一个需要训练和维护的模型。

### 22.5 PPO 为什么多出 Critic

PPO 不依赖同题多答的组均值，而是要判断：

```text
生成到当前 token 时，最终结果比这个前缀原本预期的结果好多少？
```

这就需要 critic 的 `value` 作为 baseline。GAE 再把完整 response 的 reward 和每个位置的 value 串起来，得到 `[B, R]` 的 token-level `advantages`。

换句话说：

```text
GRPO 的参照物：同题的其他回答。

PPO 的参照物：critic 对当前回答前缀的预期。
```

PPO 的优点是 token 级信用分配更细、可以通过 critic 降低策略梯度方差；代价是 actor、critic、reference、reward 四个角色同时存在，训练和调参明显更复杂。

### 22.6 不要把三种“reference”混为一谈

下面三个概念名称相近，但职责不同：

| 名称 | 出现在哪 | 是否更新 | 作用 |
| --- | --- | --- | --- |
| `ref model` | DPO、GRPO、PPO | 否 | 作为冻结锚点，约束当前 policy 不要偏离 SFT/base 太远 |
| `old policy` / `old logprob` | GRPO、PPO | 当前 rollout 后固定 | 记录采样这批 response 时的策略，用于计算 ratio |
| `critic value` | PPO | 是 | 估计当前前缀的未来 reward，用于构造 advantage 和训练 critic |

尤其是 PPO 中：

```text
old logprob：回答生成当时的 actor 概率快照。

ref model：长期冻结的 SFT/base 行为锚点。

critic value：对未来 reward 的可学习预测。
```

三者不能互相替代。

### 22.7 选择哪种方法的实践直觉

```text
有明确的标准答案：优先 SFT。

只有“回答 A 比 B 好”的偏好对：优先 DPO。

能够定义可自动打分的任务，且希望模型探索多种答案：可以用 GRPO。

需要更细的 token-level 信用分配，并愿意承担 critic 训练和稳定性成本：使用 PPO。
```

现实项目通常不是四选一，而是分阶段组合：

```text
Pretrain -> SFT -> DPO 或 GRPO / PPO
```

其中 DPO、GRPO、PPO 都属于“让模型更符合偏好或 reward”的后训练手段，只是监督信号的来源不同。

## 23. 现阶段你需要掌握到什么程度

当前阶段不要求你完整手推 PPO 论文。

你需要能讲清楚：

```text
1. Actor/policy 是生成 token 的模型。
2. Reward model 是给完整 response 打分的模型。
3. Critic/value model 是估计每个前缀未来累计回报的模型。
4. Advantage 表示本次 token 选择后的回报相比当前状态预期好多少。
5. Return 是 critic 要学习逼近的目标。
6. PPO 用 reward + value 通过 GAE 构造 advantage 和 return。
7. ratio 是当前 actor 概率 / rollout 时 old actor 概率。
8. clip 是为了限制 actor 更新幅度。
9. value loss 是为了训练 critic。
10. KL penalty 是为了限制 actor 偏离 reference。
11. PPO 更新 actor 和 critic，不更新 reference 和 reward model。
12. PPO 和 GRPO 最大区别是 critic/value 是否存在，以及 advantage 的来源不同。
```

能讲清楚这些，就可以继续看 `train_ppo.py` 的逐行实现，也能更清楚地理解 GRPO 为什么要去掉 critic。

## 24. 检查题

1. PPO 里的 actor 负责什么？
2. PPO 里的 critic 负责什么？
3. reward model 和 critic 有什么区别？
4. reference model 和 old policy 是一回事吗？
5. `old_resp_logp` 为什么要在 rollout 时保存？
6. reward 为什么通常是 `[B]`，而 advantage 是 `[B, R]`？
7. MiniMind 为什么把 reward 放到最后一个有效 response token 上？
8. `advantages` 和 `returns` 分别给谁用？
9. `ratio = exp(new_logp - old_logp)` 表示什么？
10. PPO 为什么要 clip ratio？
11. value loss 为什么训练 critic？
12. KL reference penalty 防止什么问题？
13. PPO 为什么同一批 rollout 数据不能无限更新？
14. PPO 和 GRPO 的 advantage 来源分别是什么？
15. PPO 更新哪些模型，冻结哪些模型？
