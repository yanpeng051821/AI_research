# Week 12 补充：Agentic RL 实践代办与执行指南

## 1. 这份文档的目标

Agentic RL 的概念和源码主线已经完成，接下来只补实践闭环：

```text
准备运行条件
-> 跑一次工具调用评估
-> 用 debug_mode 观察真实多轮轨迹
-> 跑一次最小 Agent RL 训练
-> 人工构造错误轨迹，核对 reward
```

完成这份代办后，应当能回答：

```text
模型实际生成了什么 tool_call？
Harness 执行了什么工具？
observation 如何回填上下文？
一条轨迹为什么得到当前 reward？
一批 B * G 轨迹何时组成 loss？
训练后的模型是否比训练前更会正确调用工具？
```

## 2. 当前机器状态

截至 2026-07-12，本机检查结果：

```text
GPU：NVIDIA GeForce RTX 4060 Ti
显存：8 GB
CUDA GPU 当前可用

缺少：
D:\pythonlearning\minimind\dataset\agent_rl.jsonl
D:\pythonlearning\minimind\out\full_sft_768.pth
reward model 目录 internlm2-1_8b-reward
```

因此当前不能直接启动完整 `train_agent.py`。先完成前置资源准备，再执行训练。

还有一个 Windows 兼容性问题：

```python
signal.signal(signal.SIGALRM, ...)
signal.alarm(1)
```

Windows Python 通常没有 `SIGALRM`。当前 `execute_tool` 会捕获异常并返回 `None`，导致工具调用被当作执行失败。

正式训练前需要把超时逻辑改成跨平台实现，或在 Windows 下跳过 `SIGALRM`。修复后至少手动验证：

```python
execute_tool("calculate_math", {"expression": "256 * 37"})
```

应返回类似：

```python
{"result": "9472"}
```

## 3. 推荐执行顺序

- [ ] P0：准备模型权重、Agent RL 数据和 reward model。
- [ ] P1：先运行 `eval_toolcall.py`，验证推理侧工具闭环。
- [ ] P2：运行一次最小 `train_agent.py --debug_mode`。
- [ ] P3：阅读并记录一条真实多轮 trajectory。
- [ ] P4：构造错误工具、错误参数和未完成轨迹，核对 reward。
- [ ] P5：保存实验记录，对比训练前后表现。

不要一上来跑完整训练。先确认：

```text
权重能加载
chat template 能注入工具 schema
模型能生成 tool_call
工具能在 Windows 正常执行
observation 能回填 messages
reward 能区分正确和错误轨迹
```

## 4. P0：准备运行条件

### 4.1 检查 Python / PyTorch / CUDA

在 PowerShell 中：

```powershell
cd D:\pythonlearning\minimind
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

预期：

```text
torch.cuda.is_available() 为 True
设备名包含 NVIDIA GeForce RTX 4060 Ti
```

### 4.2 准备基础权重

`train_agent.py` 默认：

```text
--from_weight full_sft
--hidden_size 768
--num_hidden_layers 8
```

因此 `init_model` 需要能找到与配置匹配的 full SFT 权重。先检查：

```powershell
Get-ChildItem D:\pythonlearning\minimind\out
```

不要用结构不匹配的权重。至少确认：

```text
hidden_size 一致
num_hidden_layers 一致
use_moe 一致
权重前缀和 --from_weight 一致
```

验收标准：单独执行一次模型初始化，不出现 missing key、unexpected key 或 shape mismatch。

### 4.3 准备 `agent_rl.jsonl`

默认路径：

```text
D:\pythonlearning\minimind\dataset\agent_rl.jsonl
```

每行至少包含：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个可以使用工具的助手",
      "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"description\":\"计算数学表达式\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"
    },
    {
      "role": "user",
      "content": "帮我计算 256 乘以 37"
    },
    {
      "role": "assistant",
      "content": "参考答案，AgentRLDataset 会在 rollout 前去掉这一条"
    }
  ],
  "gt": ["9472"]
}
```

数据检查：

```powershell
cd D:\pythonlearning\minimind
python -c "from dataset.lm_dataset import AgentRLDataset; print('dataset module ok')"
```

后续再用实际 tokenizer 实例化 Dataset，并检查：

```text
len(dataset) > 0
sample["messages"] 不包含最后一条参考 assistant
sample["tools"] 是 list
sample["gt"] 是 list
```

### 4.4 准备 reward model

默认参数：

```text
--reward_model_path ../../internlm2-1_8b-reward
```

相对 `trainer` 目录运行时，这个路径会指向项目目录之外。建议正式运行时传绝对路径：

```powershell
--reward_model_path D:\models\internlm2-1_8b-reward
```

验收标准：

```text
tokenizer 能加载
reward model 能加载
LMForRewardModel.get_score 能返回 float
显存不会在 actor + reference + reward model 同时加载时 OOM
```

8 GB 显存同时加载多个模型风险较高。第一次实践优先减小：

```text
batch_size = 1
num_generations = 2
max_gen_len = 128
max_total_len = 512
```

## 5. P1：阅读并运行 `eval_toolcall.py`

### 5.1 先理解评估闭环

文件：

```text
scripts/eval_toolcall.py
```

主流程：

```text
TEST_CASES 选择 prompt 和工具子集
-> run_case
-> generate / chat_api
-> parse_tool_calls
-> execute_tool
-> tool result 追加到 messages
-> 再次 generate
-> 直到模型不再生成 tool_call
```

这和训练时 `rollout_single` 使用同一种协议：

```text
assistant tool_call
-> tool observation
-> assistant next action
```

但 `eval_toolcall.py` 不计算 loss，也不更新参数。

### 5.2 本地模型运行命令

权重准备好后，从项目根目录执行：

```powershell
cd D:\pythonlearning\minimind
python scripts\eval_toolcall.py `
  --backend local `
  --load_from .\model `
  --save_dir .\out `
  --weight full_sft `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --device cuda `
  --max_new_tokens 256 `
  --temperature 0.8 `
  --top_p 0.9
```

脚本启动后选择：

```text
[0] 自动测试
```

### 5.3 需要记录什么

每个测试用例记录：

```text
prompt
提供了哪些 tools
模型第一次输出
是否生成合法 <tool_call>
工具名是否正确
arguments 是否可解析
工具返回 observation
模型是否继续生成最终回答
是否发生重复调用或死循环
```

建议写入：

```text
experiments/exp_agent_toolcall_baseline.md
```

### 5.4 P1 完成标准

- [ ] 至少运行 3 个自动测试用例。
- [ ] 至少看到 1 条成功 tool_call -> observation -> final answer。
- [ ] 能指出 `run_case` 中 messages 在每轮如何变化。
- [ ] 保存一份训练前 baseline。

## 6. P2：最小化运行 `train_agent.py`

### 6.1 为什么先用极小配置

Agent RL 一步包含：

```text
B 条任务
每条 G 条轨迹
每条轨迹最多 max_turns 轮生成
actor current forward
reference forward
reward model forward
backward
```

8 GB 显存下，默认配置很可能过重。第一次目标只是观察一条真实轨迹和一次 loss，不追求训练质量。

### 6.2 建议首次命令

从 `trainer` 目录执行，保持源码相对路径语义：

```powershell
cd D:\pythonlearning\minimind\trainer
python train_agent.py `
  --data_path ..\dataset\agent_rl.jsonl `
  --from_weight full_sft `
  --reward_model_path D:\models\internlm2-1_8b-reward `
  --device cuda:0 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 1 `
  --num_generations 2 `
  --max_seq_len 384 `
  --max_gen_len 128 `
  --max_total_len 512 `
  --num_workers 0 `
  --accumulation_steps 1 `
  --save_interval 1000 `
  --log_interval 1 `
  --debug_mode `
  --debug_interval 1 `
  --thinking_ratio 0.0 `
  --rollout_engine torch
```

说明：

```text
batch_size=1：一次只取 1 个任务。
num_generations=2：同一任务生成 2 条轨迹，仍可计算 group advantage。
max_gen_len=128：限制单轮生成长度。
max_total_len=512：限制完整多轮轨迹长度。
num_workers=0：Windows 下先排除 DataLoader 多进程问题。
debug_interval=1：每个 step 都打印轨迹。
thinking_ratio=0：先排除 thinking 格式变量。
```

如果 `num_generations=1`，组内标准差为 0，advantage 接近 0，几乎没有有效 GRPO 学习信号。因此最小训练也应使用 `G >= 2`。

### 6.3 运行时重点观察日志

应当看到：

```text
[DEBUG] CONTEXT_BEGIN / CONTEXT_END
[DEBUG] COMPLETION_BEGIN / COMPLETION_END
reward=...
Reward:...
KL:...
GrpStd:...
AdvStd:...
Loss:...
AvgLen:...
```

重点判断：

```text
GrpStd 是否长期为 0
AdvStd 是否有非零变化
KL 是否突然增大
Loss 是否为 NaN
AvgLen 是否持续顶到 max_total_len
```

### 6.4 常见失败与处理

`FileNotFoundError: agent_rl.jsonl`：

```text
数据文件尚未准备，检查 --data_path。
```

`FileNotFoundError` 或权重 shape mismatch：

```text
检查 --from_weight、hidden_size、num_hidden_layers、use_moe。
```

reward model 加载失败：

```text
改用绝对路径，确认模型目录完整。
```

CUDA out of memory：

```text
先减 max_total_len / max_gen_len。
确认 batch_size=1、num_generations=2。
关闭其他 GPU 程序。
仍然 OOM 时，先只做 eval，不做完整训练。
```

工具全部返回 `tool not found`：

```text
先检查工具名和 MOCK_RESULTS。
Windows 下检查 SIGALRM 兼容问题。
```

### 6.5 P2 完成标准

- [ ] 成功加载 actor、reference、reward model 和 Dataset。
- [ ] 成功生成至少 `B * G = 2` 条轨迹。
- [ ] 成功打印非空 reward / advantage / KL / loss。
- [ ] 至少完成一次 `loss.backward()` 和 `optimizer.step()`。
- [ ] 没有 NaN、OOM 或无限工具调用。

## 7. P3：用 debug mode 读一条真实多轮轨迹

选择一条包含工具调用的日志，按下面模板拆解：

```text
任务：

初始 messages：

可用 tools：

Turn 1 action：
是否为合法 tool_call：
工具名：
arguments：

Turn 1 observation：

Turn 2 action：
是下一次 tool_call 还是 final answer：

最终 reward：
unfinished：
```

再从 tensor 角度核对：

```text
prompt_ids：初始状态 P
response_ids：A1 + O1 + A2
response_mask：A1=1、O1=0、A2=1
response_old_logps：A1/A2 为 rollout logprob，O1 为占位 0
```

建议临时在 debug 分支增加只读打印时，观察：

```text
len(prompt_ids)
len(response_ids)
sum(response_mask)
len(response_old_logps)
```

必须满足：

```text
len(response_ids) == len(response_mask)
len(response_ids) == len(response_old_logps)
```

P3 完成标准：能够从一条日志完整讲清：

```text
state -> action -> observation -> next state -> final action -> reward
```

## 8. P4：验证错误轨迹的 reward

这一步不要求先完成长时间训练。目标是验证 reward function 是否真的区分正确和错误行为。

### 8.1 设计四组轨迹

对同一数学任务：

```text
prompt：帮我计算 256 乘以 37
tools：calculate_math
gt：["9472"]
```

准备四种输出。

正确轨迹：

```text
<tool_call>{"name":"calculate_math","arguments":{"expression":"256 * 37"}}</tool_call>
最终回答包含 9472。
```

错误工具：

```text
<tool_call>{"name":"get_current_weather","arguments":{"location":"北京"}}</tool_call>
```

错误参数：

```text
<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>
```

未完成轨迹：

```text
最后一轮仍输出 tool_call，达到 max_turns，没有 final answer。
```

### 8.2 预期 reward 排序

至少应满足：

```text
正确轨迹 reward
> 错误参数轨迹 reward
> 未知工具或未完成轨迹 reward
```

具体数值可能受这些项影响：

```text
tool_call 标签完整性
工具名是否合法
参数是否通过 CHECK_ARGS
有效调用数量与 gt 数量的差异
gt 是否出现在工具结果或最终回答中
unfinished 惩罚
重复惩罚
[-3, 3] 总分裁剪
```

### 8.3 建议记录表

| 场景 | 标签合法 | 工具合法 | 参数合法 | GT 命中 | unfinished | Reward |
| --- | --- | --- | --- | --- | --- | --- |
| 正确轨迹 | 是 | 是 | 是 | 是 | 否 | 待填写 |
| 错误工具 | 是 | 否 | 不适用 | 否 | 否 | 待填写 |
| 错误参数 | 是 | 是 | 否 | 否 | 否 | 待填写 |
| 未完成 | 是 | 视情况 | 视情况 | 否 | 是 | 待填写 |

如果错误轨迹反而得分更高，不要继续训练。先修 reward，避免 reward hacking。

### 8.4 P4 完成标准

- [ ] 四种轨迹都得到可解释的 reward。
- [ ] 正确轨迹分数最高。
- [ ] 错误工具和错误参数受到惩罚。
- [ ] unfinished 受到额外惩罚。
- [ ] 每个分数都能对应到 `calculate_rewards` 的具体分项。

## 9. P5：训练前后评估闭环

训练前：

```text
full_sft 权重
-> eval_toolcall.py
-> 保存 baseline
```

最小训练后：

```text
agent 权重
-> eval_toolcall.py --weight agent
-> 使用相同 TEST_CASES 和采样参数
```

对比维度：

```text
工具选择准确率
JSON 参数可解析率
必填参数完整率
工具调用成功率
最终答案命中率
平均调用轮数
无效/重复调用次数
未完成轨迹比例
```

不要只比较 loss。Agentic RL 的目标是任务行为改善，必须看真实 trajectory。

## 10. 最终验收清单

- [ ] 解决 Windows `SIGALRM` 工具执行问题。
- [ ] 准备 `agent_rl.jsonl`。
- [ ] 准备匹配的 full SFT 权重。
- [ ] 准备 reward model。
- [ ] 跑通 `eval_toolcall.py` baseline。
- [ ] 跑通最小 `train_agent.py --debug_mode`。
- [ ] 保存并解释一条多轮 trajectory。
- [ ] 验证四种错误轨迹的 reward。
- [ ] 完成训练前后同条件评估。

全部完成后，Agentic RL 可以从“源码理解完成”更新为：

```text
最小实践闭环完成
```
