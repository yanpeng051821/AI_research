# 第 12 周：Agentic RL 与教育 Agent 迁移

## 1. 这篇文档解决什么问题

前面的 GRPO / PPO 通常是单轮 rollout：

```text
prompt -> model.generate -> response -> reward -> 更新 policy
```

Agentic RL 把一段 response 扩展为会与环境交互的多轮轨迹：

```text
用户任务
-> 模型生成 action（普通文本或 tool_call）
-> harness 解析并执行工具
-> 工具返回 observation
-> observation 回填上下文
-> 模型生成下一步 action
-> ...
-> 最终回答
-> 对整条轨迹打 reward
-> 用 GRPO / CISPO 更新 policy
```

这周不追求搭建工业级 Agent 框架，而是建立可落到 MiniMind 源码的认识：模型负责生成 token；harness 负责执行循环；环境负责提供真实 observation；reward 评价任务是否完成；RL 提高好轨迹的概率。

对应文件：

```text
dataset/lm_dataset.py -> AgentRLDataset
trainer/train_agent.py
scripts/eval_toolcall.py
trainer/rollout_engine.py
```

## 2. 从 GRPO 到 Agentic RL：rollout 的边界变大了

普通 GRPO：

```text
state s0 = prompt
policy 生成 completion
得到一条 response
```

Agentic RL：

```text
state s0 = 用户问题 + 工具说明
-> action a1 = <tool_call>...</tool_call>
-> observation o1 = 工具结果
-> state s1 = s0 + a1 + o1
-> action a2 = 最终回答，或下一次 tool_call
```

一条完整过程是 trajectory：

```text
tau = (s0, a1, o1, a2, o2, ..., aT)
```

其中 `state` 是消息历史，`action` 是模型生成 token，`observation` 是工具返回，`trajectory` 是任务从开始到结束的完整 action / observation 历史。模型本体仍是自回归语言模型；它新增的不是新网络，而是通过上下文读取环境反馈并改变后续决策的能力。

## 3. Model、Harness、Tool、Environment 的边界

| 角色 | 负责什么 | MiniMind 对应 |
| --- | --- | --- |
| Policy model | 根据上下文生成下一 token 或 tool_call | `model` / `rollout_engine` |
| Harness | 维护 messages、解析调用、决定循环是否继续 | `rollout_single`、`run_case` |
| Tool / Environment | 接收参数、执行操作、返回 observation | `execute_tool`、`MOCK_RESULTS` |
| Reward function | 评价完整 trajectory | `calculate_rewards` |

边界要记牢：

```text
模型不会直接执行 Python、访问数据库或天然拥有实时信息。
Harness 不替模型思考，只把模型表达出的调用意图变成可执行操作。
工具结果不更新参数，只作为下一轮上下文。
RL 不会凭空创造工具能力，只能优化已有工具、数据和 reward 范围内的策略。
```

这也对应你之前的 `model + harness` 思考：模型会越来越擅长“何时调用什么、参数怎么写、结果怎么利用”；但权限控制、持久化、审计、重试和真实副作用必须留在参数之外。

## 4. 工具调用的三个对象：Schema、Call、Response

### 4.1 Tool schema：模型事前看到的能力说明

`train_agent.py` 的 `TOOLS` 为每个工具声明名称、描述、参数类型、必填字段：

```python
{
  "type": "function",
  "function": {
    "name": "calculate_math",
    "parameters": {
      "type": "object",
      "properties": {"expression": {"type": "string"}},
      "required": ["expression"]
    }
  }
}
```

它回答的是“可选动作有哪些、参数结构是什么”，并不执行工具。

### 4.2 Tool call：模型生成的 action

模型按 chat template 生成：

```text
<tool_call>
{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}
</tool_call>
```

这仍然只是自回归生成的 token。模型学会的是调用协议和参数组织，不是执行数学函数。

### 4.3 Tool response：环境返回的 observation

Harness 解析 JSON 后执行：

```python
result = execute_tool(name, args)
messages.append({"role": "tool", "content": json.dumps(result)})
```

下一轮模板会把它组织成：

```text
<tool_response>
{"result": "9472"}
</tool_response>
```

`<tool_call>` 是 policy 的 action；`<tool_response>` 是环境提供的 observation。两者都是特殊 token，但训练角色完全不同。

## 5. Chat Template：messages 如何变成 Agent prompt

Agent 的输入先是结构化消息：

```python
messages = [{"role": "user", "content": "帮我算 256 乘以 37"}]
```

调用：

```python
tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, tools=tools
)
```

简化后的文本为：

```text
<|im_start|>system
# Tools
<tools>
{...calculate_math schema...}
</tools>
请在 <tool_call>...</tool_call> 中返回工具名和 JSON 参数
<|im_end|>
<|im_start|>user
帮我算 256 乘以 37
<|im_end|>
<|im_start|>assistant
```

工具 schema 本身也是上下文 token。工具越多、描述越长，留给任务和轨迹的上下文越少；这就是实际系统会做工具检索、筛选和压缩的原因。

## 6. 一条完整 trajectory

任务：“生成一个 1 到 1000 的随机数，然后计算平方。”

```text
state s0：用户问题 + random_number / calculate_math schema

action a1：
<tool_call>{"name":"random_number","arguments":{"min":1,"max":1000}}</tool_call>

observation o1：
<tool_response>{"result":71}</tool_response>

action a2：
<tool_call>{"name":"calculate_math","arguments":{"expression":"71**2"}}</tool_call>

observation o2：
<tool_response>{"result":"5041"}</tool_response>

action a3：随机数是 71，它的平方是 5041。
```

关键不是模型把流程“说出来”，而是第二次调用里的 `71` 来自真实 observation `o1`，不是模型凭空猜的内容。

## 7. `rollout_single`：多轮循环如何实现

### 7.1 Agent RL 数据先提供什么

`AgentRLDataset` 读取每条样本的 `conversations` 和 `gt`。它会从 system message 中取出 `tools`，并返回：

```text
messages：初始任务上下文，不包含最后一条目标回复。
tools：本任务可调用的工具 schema。
gt：最终可验证目标。
```

代码中的 `messages[:-1]` 正是在避免把数据集中最后的 assistant 目标提前泄露给 rollout。它和 SFT 的区别非常直接：

```text
SFT：数据集给出完整 assistant 回复，模型模仿它。

Agent RL：数据集给出初始状态、工具和 gt，模型现场生成调用过程与最终回复。
```

### 7.2 多轮 rollout 循环

`train_agent.py` 的 `rollout_single` 是本周最重要的函数。它最多循环 `max_turns=3` 次：

```text
messages -> chat template -> context
context -> rollout_engine.rollout -> new_text

没有 tool_call：停止，new_text 是最终回答

有 tool_call：解析 JSON -> 执行工具 -> assistant 输出和 tool 结果 append 回 messages -> 下一轮
```

它返回的也不只是 completion：

| 产物 | 含义 |
| --- | --- |
| `final_output` | 最后一轮模型输出，通常为最终回答 |
| `final_context` | 含工具结果的完整上下文，主要用于调试 |
| `prompt_ids` | 首轮生成前的 token |
| `response_ids` | 模型输出与工具 observation 拼接后的 token |
| `response_mask` | 哪些 token 是模型 action，哪些来自工具 |
| `response_old_logps` | rollout 时 action token 的 old logprob |
| `all_outputs` | 每轮模型输出 |
| `unfinished` | 是否达到最大轮数仍在调用工具 |

`max_turns` 是 harness 的安全边界，不是模型能力上限。它限制无限循环和训练成本。

## 8. 最关键的工程细节：为什么 tool response 不参与 policy loss

`rollout_single` 用 `response_mask` 区分 action 与 observation：

```python
# 模型生成：policy 的 action
response_ids.extend(new_ids)
response_mask.extend([1] * len(new_ids))
response_old_logps.extend(new_logps)

# 工具返回：environment 的 observation
response_ids.extend(obs_delta)
response_mask.extend([0] * len(obs_delta))
response_old_logps.extend([0.0] * len(obs_delta))
```

所以一条轨迹的 mask 是：

```text
prompt token：0
assistant 的 tool_call token：1
tool_response token：0
assistant 的最终回答 token：1
padding：0
```

只有模型选择的 token 是 action，才应被 reward 强化或抑制。工具结果不是模型生成的，不能要求模型“提高生成工具返回值的概率”。但它仍然会进入下一轮 state，强烈影响之后的工具选择和回答。

## 9. Agent trajectory 的 shape 与训练数据流

记：

```text
B：DataLoader 的原始任务数。
G：每任务生成轨迹数，`num_generations`。
N = B * G：实际更新的轨迹数。
P：首轮 prompt 长度。
S：拼接并 padding 后的完整轨迹长度。
V：词表大小。
```

```text
AgentRLDataset
-> messages_batch / tools_batch / gt_batch，均按 B 个任务组织
-> rollout_batch(..., G)
-> N 条 trajectory

prompt_ids + response_ids
-> input_ids [N, S]
-> full_response_masks [N, S]
-> old_per_token_logps [N, S - 1]

current policy
-> logits [N, S, V]
-> per_token_logps [N, S - 1]

reference policy
-> ref_per_token_logps [N, S - 1]

calculate_rewards
-> rewards [N]
-> grouped_rewards [B, G]
-> advantages [N]
```

`advantages [N]` 会 broadcast 到 token 维度，并由 `completion_mask [N, S - 1]` 排除 prompt、tool response 和 padding。当前实现给一条 trajectory 的所有 action token 同一个 trajectory-level advantage，因此延迟 reward 的信用分配较粗：成功时整条调用过程被鼓励，失败时整条过程被压低。

## 10. Delayed Reward：为什么要为整条轨迹评分

Agent 的最终回答看似正确，不代表过程正确：它可能调用了错误工具、伪造了结果，或在无意义循环。因此 `calculate_rewards` 组合评价：

```text
R(tau)
= 工具名和参数是否合法
+ 调用数量是否与 gt 对齐
+ 最终回答是否命中 gt
+ 非工具回答的格式 / think 标签 / reward model 分数
- 未闭合 tool_call 标签
- 达到最大轮数仍未完成
- 重复文本

最后 clip 到 [-3, 3]
```

`gt` 是 Agent RL 数据相对普通对话数据新增的字段：普通 SFT 给标准回答；Agent RL 给可验证目标，模型自己探索调用和回答过程。数学题可用 `5041` 做 gt；教育 Agent 则可用“计划覆盖三项薄弱知识点且无时间冲突”等结构化检查结果。

reward 最难的是是否代表你真正想要的行为。只奖励工具调用会导致滥用工具；只奖励关键词命中会导致猜答案；只奖励文本很长会导致啰嗦。这就是 reward hacking。

## 11. 与 GRPO、PPO 的关系

MiniMind 的 `train_agent.py` 使用 GRPO / CISPO 风格，而不是 PPO：

```text
同一任务 rollout G 条 trajectory
-> 每条 trajectory 一个 reward
-> 组内标准化得到 advantage
-> ratio + clipped objective / CISPO + reference KL
-> 更新 policy
```

| 维度 | 普通 GRPO | MiniMind Agentic RL |
| --- | --- | --- |
| rollout 产物 | 单轮 response | 多轮 trajectory |
| action | response token | 文本、tool_call、最终回答 token |
| observation | 通常没有 | tool_response 回填上下文 |
| reward | 回答质量 / 规则 | 工具合法性、gt、格式、完成度、回答质量 |
| advantage | 同 prompt 多 response | 同任务多 trajectory |
| 更新位置 | response token | 只更新 action token |
| critic | 没有 | MiniMind 当前没有 |

因此 Agentic RL 不是新 loss，而是将 GRPO / CISPO 的策略优化应用到多轮工具轨迹。`ref_model` 仍然冻结，用 KL 防止 policy 为追逐 reward 而偏离 SFT 基座太远。

## 12. `train_agent.py` 的完整训练闭环

```text
AgentRLDataset 取出 messages / tools / gt
-> rollout_batch 生成 G 条多轮轨迹
-> packing 拼接 prompt、action、tool observation
-> response_mask 标记 action token
-> calculate_rewards 为每条轨迹打分
-> grouped rewards 得 trajectory-level advantage
-> current policy 得 new logprob
-> ref policy 得 ref logprob
-> ratio + GRPO/CISPO objective + KL
-> completion_mask 屏蔽非 action token
-> policy loss
-> 更新 policy
-> rollout_engine.update_policy(model) 同步最新权重
```

日志重点：

```text
Reward：平均轨迹得分。
GrpStd：同题轨迹 reward 是否有差异；接近 0 时 GRPO 信号弱。
AdvStd：优势信号是否存在。
KL：当前 policy 相对 reference 的漂移。
AvgLen：平均 action token 数，排查无限调用或过长输出。
```

## 13. `eval_toolcall.py`：训练和推理使用同一交互协议

评测脚本的 `run_case` 也是同一个循环：

```text
messages
-> apply_chat_template(..., tools=tools)
-> model.generate
-> parse_tool_calls
-> execute_tool
-> messages.append(assistant)
-> messages.append(tool result)
-> 继续或结束
```

它支持本地 MiniMind 权重和 OpenAI 兼容 API。训练并不是把工具函数塞进参数，而是让模型学会在这套 schema、消息角色和调用格式下正确行动。

## 14. 映射到教育 Agent：先设计环境，再谈训练

教育 Agent 的第一版工具建议：

| 工具 | 输入 | observation | 目的 |
| --- | --- | --- | --- |
| `get_learner_profile` | `user_id` | 基础、目标、偏好、薄弱点摘要 | 个性化 |
| `get_learning_history` | `user_id`, `time_range` | 时长、完成率、错题主题 | 判断进度 |
| `query_knowledge_state` | `user_id`, `concept_ids` | 掌握度及证据 | 诊断 |
| `search_learning_resources` | `topic`, `level` | 资源元数据 | 推荐资料 |
| `create_study_plan` | `user_id`, `tasks`, `schedule` | plan_id 或冲突信息 | 落地计划 |
| `record_learning_event` | `user_id`, `event` | 写入确认 | 更新学习记录 |

模型负责理解意图、选工具、抽参数、读 observation 后继续规划；系统负责权限、数据库读写、冲突检测、幂等、审计和重试。真实世界状态与副作用不能只依赖模型参数记忆。

最小计划任务可定义为：

```text
用户：根据我本周的空闲时间和薄弱知识点，制定 5 天学习计划。

R_profile：正确读取画像 / 历史。
R_schedule：计划无时间冲突。
R_coverage：覆盖目标薄弱知识点。
R_action：成功创建计划，而不是只在文本中声称已创建。
R_explanation：最终解释清楚、不过度承诺。
- R_invalid：缺参、非法工具、越权。
- R_loop：无意义重复调用。
```

先将 reward 绑定到系统可验证的事实，再逐步加入 judge model 对解释质量的评价。否则模型容易写出“看起来专业”的计划，却没有真正读用户状态或创建计划。

## 15. 用户画像与长期记忆的边界

教育 Agent 的长期记忆不应是无限追加的 prompt，而应分层：

```text
事件层：完成练习、错题、停留时长、反馈。
事实层：可追溯的稳定结论，如偏好中文解释、周末时间更多。
画像层：对事件和事实聚合后的工作摘要。
会话层：本次任务需要的临时上下文。
```

模型可以提出画像更新候选，但不应自行无审计写入长期事实：

```text
模型提出 candidate
-> 系统校验来源、置信度与权限
-> 写入事实层或待确认队列
-> 后台聚合新的画像摘要
```

## 16. MiniMind 实现的边界与本周学习顺序

MiniMind 很适合学习核心闭环，但不等于生产方案：工具多为 `MOCK_RESULTS`；`max_turns=3`；训练是同步 rollout 后更新；reward 主要靠规则和 gt；示例中的 `eval` 绝不能原样暴露到生产环境。

后续按此顺序读代码：

```text
1. TOOLS：schema 如何告诉模型可用能力。
2. parse_tool_calls + execute_tool：文本 action 如何变为 observation。
3. rollout_single：messages 如何多轮增长与停止。
4. response_mask / packing：为什么只训练模型 action token。
5. calculate_rewards：延迟 reward 如何评价整条轨迹。
6. GRPO / CISPO loss：reward 如何更新 policy。
7. eval_toolcall.py：训练后的模型如何运行和评测。
8. 教育 Agent：工具、状态、reward、记忆的最小设计。
```

学完后应能回答：tool schema、call、response 的区别；为何 observation 不参与 policy loss；为何 reward 在整条 trajectory 后结算；以及教育 Agent 中哪些能力属于模型、哪些必须由 harness / 系统承担。
