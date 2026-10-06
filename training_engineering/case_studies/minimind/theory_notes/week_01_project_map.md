# 第 1 周：项目地图与环境理解

## 1. 本周学习目标

第 1 周不追公式、不训练模型、不深读 attention。目标是先建立 MiniMind 的工程地图，搞清楚：

- MiniMind 为什么适合从 0 到 1 学 LLM。
- 一个 LLM 项目通常有哪些阶段。
- MiniMind 的目录分别对应哪些阶段。
- 最小训练链路是什么。
- 后面每周要学的内容在项目中分别落在哪些文件。

本周完成后，你应该能用自己的话讲清楚：

```text
MiniMind 是一个从 tokenizer、数据、模型、pretrain、SFT、后训练、Agent 到部署都覆盖的小型 LLM 教学项目。
```

## 2. 今天先建立的主线

MiniMind 不是单纯的模型文件，而是一条完整链路：

```text
原始文本/对话数据
-> tokenizer
-> Dataset
-> MiniMind 模型
-> Pretrain
-> SFT
-> LoRA / Distillation / DPO / PPO / GRPO
-> Tool Use / Agentic RL
-> 推理 / API / Web Demo / 第三方部署
```

这条链路里，每个阶段解决的问题不同：

| 阶段 | 解决的问题 |
|---|---|
| tokenizer | 把文本变成 token id |
| Dataset | 把样本组织成 `input_ids` 和 `labels` |
| 模型结构 | 把 token 序列映射成下一个 token 的概率分布 |
| Pretrain | 学语言规律、基础知识和续写能力 |
| SFT | 学会按指令和对话格式回答 |
| LoRA | 低成本适配垂类任务 |
| 蒸馏 | 把大模型能力迁移给小模型 |
| DPO | 根据 chosen/rejected 偏好优化回答 |
| PPO/GRPO | 用 reward 优化生成策略 |
| Agentic RL | 学会多轮工具调用和延迟奖励任务 |
| 推理部署 | 把训练好的模型变成可交互服务 |

## 3. MiniMind 目录地图

```text
minimind/
├── README.md
├── requirements.txt
├── model/
├── dataset/
├── trainer/
├── scripts/
└── eval_llm.py
```

### README.md

项目说明书。第 1 周重点读它，不要求所有细节都懂，但要知道项目包含哪些阶段。

重点关注：

- 项目介绍。
- 快速开始。
- 数据介绍。
- 模型结构。
- 主要训练。
- 评估和部署。

### requirements.txt

依赖列表。它告诉我们这个项目需要哪些库，例如 PyTorch、transformers、datasets 等。

第 1 周只需要知道它是环境入口，不需要逐个库深挖。

### model/

模型相关代码。

```text
model_minimind.py
model_lora.py
tokenizer.json
tokenizer_config.json
```

后面第 3、4、8 周会重点学习这里。

### dataset/

数据读取与样本构造。

```text
lm_dataset.py
dataset.md
```

第 2 周会重点学习这里，尤其是 `input_ids`、`labels` 和 `-100`。

### trainer/

训练脚本集中地。

```text
train_tokenizer.py
train_pretrain.py
train_full_sft.py
train_lora.py
train_distillation.py
train_dpo.py
train_ppo.py
train_grpo.py
train_agent.py
rollout_engine.py
trainer_utils.py
```

它对应整个训练路线：

```text
tokenizer -> pretrain -> SFT -> LoRA/Distill/DPO/RL -> Agent
```

### scripts/

推理、服务、转换和工具调用评估脚本。

```text
chat_api.py
serve_openai_api.py
web_demo.py
eval_toolcall.py
convert_model.py
```

第 7 周会重点看推理，第 12 周会看 tool call。

### eval_llm.py

本地推理和测试入口之一。第 7 周学习。

## 4. 第 1 周必须理解的 4 个判断

### 4.1 MiniMind 的价值不是性能最强，而是链路完整

MiniMind 的参数量很小，不要把它和 Qwen、Llama、DeepSeek 正面比效果。它的价值在于：

```text
小到能学习，完整到能看见 LLM 全流程。
```

对你来说，这比直接上手巨大模型更合适。

### 4.2 Pretrain 和 SFT 是主线中的主线

README 里说快速复现 MiniMind Zero，默认只需要：

```text
pretrain_t2t_mini.jsonl
sft_t2t_mini.jsonl
```

对应训练阶段：

```text
Pretrain -> SFT
```

这是最小闭环。后面的 LoRA、DPO、GRPO、Agentic RL 都建立在这个基础上。

### 4.3 数据、模型、训练脚本是一一对应的

不是随便一个脚本读随便一个数据。

大致关系是：

| 数据/任务 | Dataset/脚本 |
|---|---|
| pretrain 文本 | `PretrainDataset` + `train_pretrain.py` |
| SFT 对话 | `SFTDataset` + `train_full_sft.py` |
| LoRA 垂类对话 | `SFTDataset` + `train_lora.py` |
| DPO 偏好对 | `DPODataset` + `train_dpo.py` |
| RLAIF | `RLAIFDataset` + `train_grpo.py` / `train_ppo.py` |
| Agent 工具轨迹 | `AgentRLDataset` + `train_agent.py` |

### 4.4 第 1 周只需要建立地图，不需要理解所有算法

看到 PPO、GRPO、CISPO、Agentic RL 不懂是正常的。现在只要知道它们在链路后面，属于后训练和 Agent 阶段。

本周最重要的是这张图：

```text
dataset/
  负责把数据变成模型可训练的样本

model/
  负责定义模型如何从 token 算 logits

trainer/
  负责定义不同阶段如何用 loss/reward 更新模型

scripts/ 和 eval_llm.py
  负责把模型拿出来推理、服务和评估
```

## 5. 本周 Checklist

- [ ] 读完 README 的项目介绍。
- [ ] 读完 README 的快速开始。
- [ ] 读完 README 的数据介绍。
- [ ] 读完 README 的模型结构介绍。
- [ ] 读完 README 的主要训练流程。
- [ ] 列出每个核心文件的职责。
- [ ] 画出 MiniMind 从数据到推理的流程图。

## 6. 已掌握的最小训练闭环

你已经能把 MiniMind 的 LLM 训练过程和 CV 图像分类训练建立对应关系：

```text
CV 图像分类：
图片 -> transforms -> 模型 -> logits -> 类别标签 -> cross entropy -> 更新参数

LLM 预训练：
文本 -> tokenizer -> Transformer -> logits -> 下一个 token -> cross entropy -> 更新参数
```

### 6.1 tokenizer 到 input_ids

一段文本进入 MiniMind 后，会先经过 tokenizer。

tokenizer 根据已经训练好的 BPE/ByteLevel 词表，把文本切成 token，并转成 token id。

需要注意：

```text
token 不一定等于自然语言里的“词”。
一个词可能被切成多个 token，标点、空格、特殊符号也可能成为 token。
```

### 6.2 input_ids、labels、logits 的 shape

如果：

```text
batch_size = B
seq_len = T
hidden_size = C
vocab_size = V
```

那么：

```text
input_ids: [B, T]
labels:    [B, T]
embedding 后: [B, T, C]
Transformer 后: [B, T, C]
lm_head 后 logits: [B, T, V]
```

以 MiniMind 主线配置举例：

```text
B = 4
T = 512
C = 768
V = 6400

input_ids: [4, 512]
labels:    [4, 512]
embedding: [4, 512, 768]
logits:    [4, 512, 6400]
```

### 6.3 labels 和 shift

在 `PretrainDataset` 中：

```python
labels = input_ids.clone()
labels[input_ids == tokenizer.pad_token_id] = -100
```

这里没有显式移动 labels。真正的 shift 发生在模型 forward 里：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

也就是：

```text
第 0 个位置的 logits 预测第 1 个 token
第 1 个位置的 logits 预测第 2 个 token
第 2 个位置的 logits 预测第 3 个 token
```

### 6.4 当前最重要的一句话

```text
LLM 训练就是：给模型一串 token，让它在每个位置预测下一个 token，然后用交叉熵更新参数。
```

## 7. 当前需要校准的小点

你复述时写到：

```text
经过 Transformer 后 [B, T, C]
然后经过 fnn 后 [B, T, C]
最后通过 lm_head 得到 [B, T, V]
```

这里需要稍微调整：

```text
Transformer block 内部本身就包含 Attention 和 FFN。
```

所以更准确的链路是：

```text
input_ids
-> embedding
-> 多层 Transformer block
   每个 block 内部包含 Attention + FFN + 残差连接 + Norm
-> final norm
-> lm_head
-> logits
-> shift 后计算 cross entropy loss
```

也就是说，FFN 不是整个 Transformer 之后额外单独接的一层，而是每个 Transformer block 里面都有一个 FFN。

## 8. 当前进度

状态：进行中

已完成：

- [x] 建立第 1 周学习笔记。
- [x] 梳理 MiniMind 的整体阶段。
- [x] 梳理核心目录职责。
- [x] 完成 README 初读，能用自己的话说明 MiniMind 是缩小版完整 LLM 训练项目。
- [x] 理解 tokenizer、input_ids、labels、logits、cross entropy 的最小训练闭环。
- [x] 能写出 `[B,T] -> [B,T,C] -> [B,T,V]` 的基础 shape 链路。

待完成：

- [x] 你亲自读 README 的项目介绍和快速开始。
- [x] 我们一起复述 MiniMind 的最小训练闭环。
- [ ] 我们一起完成项目结构图。

## 9. 今日小结

今天先记住一句话：

```text
MiniMind = 一个适合学习的缩小版 LLM 全流程工程。
```

它的学习顺序不是从 attention 直接开啃，而是：

```text
项目地图 -> 数据/tokenizer -> 模型结构 -> pretrain/SFT -> 推理 -> 后训练 -> Agent
```
