# MiniMind 从 0 到 1 学习计划

## 1. 学习背景与目标

你的背景是本科物理，后来跨考计算机，并在北邮学习计算机视觉方向。你已经做过图像分类相关项目，例如奢侈品鉴伪，因此对深度学习任务的基本形态有实践经验。但在毕业找工作阶段，你发现自己的 Python 基础、PyTorch 基础和深度学习基础还不够牢固，所以现在正在 `/Users/yanpeng/Documents/PythonProjects` 下系统补齐 Python 与中高级编程能力。

现在的目标是围绕 MiniMind 项目完成一次从 0 到 1 的 LLM 学习。这个目标不只是“能把项目跑起来”，而是要真正理解一个小语言模型如何从数据、tokenizer、模型结构、训练循环、后训练、推理部署，逐步发展到 Tool Use 和 Agentic RL。

MiniMind 适合你当前阶段的原因是：

- 它足够小，可以在个人设备或低成本 GPU 上复现。
- 它足够完整，覆盖 tokenizer、pretrain、SFT、LoRA、DPO、PPO、GRPO、Agentic RL、蒸馏和部署。
- 它没有完全依赖 `transformers`、`trl`、`peft` 等高级封装，很多关键训练逻辑是用 PyTorch 原生代码实现的。
- 它能把你已有的 CV 深度学习经验迁移到 LLM 领域，同时补上 PyTorch、训练循环、语言模型目标函数和 Agent 训练的知识链条。

最终学习目标是：

- 理解 MiniMind 的完整项目结构。
- 理解一个文本样本如何变成 token、label 和 loss。
- 理解 MiniMind 模型结构中每一层的作用和 tensor shape。
- 理解 pretrain、SFT、LoRA、蒸馏、DPO、PPO、GRPO、Agentic RL 的训练目标差异。
- 能够读懂主要源码，并能修改一个小模块完成实验。
- 能把 MiniMind 的小模型和 Agent 训练思路迁移到未来 AI 教育 Agent 方向。

## 2. 当前已掌握内容与进度

你之前已经在另一个 AI 软件上学过一部分 LLM forward 主线：

```text
input -> tokenizer -> embedding -> pre_norm -> attention -> pre_norm -> ffn -> out_proj -> logits -> 自回归生成 + KV cache
```

这部分正好对应 MiniMind 的核心模型文件：

```text
model/model_minimind.py
```

当前进度评估如下：

| 模块 | 当前状态 | 后续任务 |
|---|---|---|
| Python 基础 | 进行中 | 保持练习，重点补文件读写、类、迭代器、装饰器、异常、包管理 |
| PyTorch 基础 | 入门到进阶过渡 | 补 tensor shape、autograd、Dataset、DataLoader、optimizer、AMP、DDP |
| Transformer forward 主线 | 已入门 | 用 MiniMind 源码复盘每个模块的 shape 和计算逻辑 |
| Tokenizer 概念 | 已入门 | 深入 BPE、ByteLevel、special tokens、chat template |
| 自回归生成 | 已入门 | 复盘 top-k、top-p、temperature、repetition penalty、KV cache |
| 训练循环 | 待系统学习 | 从 pretrain 和 SFT 脚本开始 |
| 后训练 | 待学习 | 按 LoRA、蒸馏、DPO、PPO、GRPO 顺序推进 |
| Agentic RL | 后期学习 | 等前面基础稳固后再进入 |

整体进度可以暂定为：

```text
MiniMind 完整链路进度：25%
```

这里的 25% 不是低评价，而是因为你已经学过模型 forward 主干，但还没有系统打通数据、训练、后训练和 Agent 链路。LLM 学习真正困难的部分，往往不是单个 Transformer block，而是“数据如何组织、loss 如何定义、训练目标如何改变模型行为”。

## 3. MiniMind 项目地图

MiniMind 的核心目录如下：

```text
minimind/
├── model/
│   ├── model_minimind.py
│   ├── model_lora.py
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── dataset/
│   ├── lm_dataset.py
│   └── dataset.md
├── trainer/
│   ├── train_tokenizer.py
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   ├── train_lora.py
│   ├── train_distillation.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   ├── train_grpo.py
│   ├── train_agent.py
│   ├── rollout_engine.py
│   └── trainer_utils.py
├── scripts/
│   ├── chat_api.py
│   ├── serve_openai_api.py
│   ├── web_demo.py
│   ├── eval_toolcall.py
│   └── convert_model.py
├── eval_llm.py
└── README.md
```

学习时不要一开始就平均阅读所有文件。推荐顺序是：

```text
README.md
-> dataset/lm_dataset.py
-> trainer/train_tokenizer.py
-> model/tokenizer_config.json
-> model/model_minimind.py
-> trainer/train_pretrain.py
-> trainer/train_full_sft.py
-> eval_llm.py
-> model/model_lora.py
-> trainer/train_lora.py
-> trainer/train_distillation.py
-> trainer/train_dpo.py
-> trainer/train_grpo.py
-> trainer/train_agent.py
-> trainer/rollout_engine.py
```

这个顺序是阶梯式的：先看项目说明，再看数据和 tokenizer 如何把文本变成模型输入，然后看模型结构，再看训练循环，最后进入后训练和 Agent。这样不会出现“已经在读 attention，但还不清楚 `input_ids` 和 `labels` 从哪里来”的跳跃。

## 4. 总体学习逻辑

学习 MiniMind 不应该按“文件列表”机械推进，而应该按一条因果链推进：

```text
文本
-> tokenizer
-> input_ids
-> Dataset 生成 labels
-> Embedding
-> Transformer blocks
-> logits
-> loss
-> backward
-> optimizer step
-> checkpoint
-> generate
-> SFT 改变回答方式
-> LoRA 适配垂类任务
-> 蒸馏把大模型能力压缩到小模型
-> DPO/RL 改变偏好和策略
-> Agentic RL 学会工具调用与多轮轨迹
```

每一阶段都要回答三个问题：

- 输入是什么？
- 输出是什么？
- 为什么这样设计？

例如，学习 SFT 时不要只问“代码怎么写”，而要问：

- 为什么 user 和 system 的 token 不参与 loss？
- 为什么只训练 assistant 的回复？
- 为什么同样是交叉熵，pretrain 和 SFT 训练出来的行为不一样？
- chat template 如何影响模型学到的对话格式？

这类问题会让你真正理解训练目标，而不是停留在跑脚本。

这套体系的节奏要遵循一个原则：

```text
先理解数据如何进入模型
再理解模型如何计算 logits
再理解 loss 如何推动参数更新
最后理解不同后训练目标如何改变模型行为
```

如果某一阶段还不能讲清楚“输入、输出、loss、shape”，不要急着进入下一阶段。MiniMind 的价值不在于快速浏览所有脚本，而在于把一条完整训练链路真正打通。

## 5. 阶段一：Python 与 PyTorch 补强

### 5.1 Python 必备能力

MiniMind 源码阅读中会频繁遇到：

- 类与继承
- `__init__` 和 `forward`
- 列表、字典、生成器
- 文件路径和 `os.path`
- JSON / JSONL 读写
- 命令行参数 `argparse`
- 模块导入与包路径
- 上下文管理器
- 随机数与 seed

重点不是刷题，而是能读懂项目代码。

建议练习：

- 写一个读取 JSONL 文件的小脚本，逐行解析样本。
- 写一个简单 `Dataset` 类，返回 `input_ids` 和 `labels`。
- 写一个带 `argparse` 的训练入口，支持 `--batch_size`、`--learning_rate`、`--epochs`。

### 5.2 PyTorch 必备能力

你需要重点掌握：

- `torch.Tensor` 的 shape 变换
- `view`、`reshape`、`transpose`、`permute`
- `nn.Module`
- `nn.Linear`
- `nn.Embedding`
- `nn.ModuleList`
- `F.cross_entropy`
- `F.softmax` 和 `F.log_softmax`
- `torch.gather`
- `loss.backward()`
- `optimizer.step()`
- `optimizer.zero_grad()`
- `Dataset` 和 `DataLoader`
- 混合精度 `autocast` 和 `GradScaler`
- 梯度裁剪 `clip_grad_norm_`

MiniMind 中最重要的 shape 直觉是：

```text
input_ids: [batch, seq]
embedding: [batch, seq, hidden]
q/k/v: [batch, seq, heads, head_dim]
attention scores: [batch, heads, seq, seq]
hidden_states: [batch, seq, hidden]
logits: [batch, seq, vocab]
labels: [batch, seq]
loss: scalar
```

只要 shape 直觉稳了，读 LLM 源码会轻松很多。

## 6. 阶段二：Tokenizer 与数据格式

对应文件：

```text
trainer/train_tokenizer.py
dataset/lm_dataset.py
model/tokenizer.json
model/tokenizer_config.json
```

### 6.1 Tokenizer 的意义

Tokenizer 是语言模型的“输入接口”。它负责把自然语言转换成 token id，也负责把模型输出的 token id 解码回文本。

MiniMind 使用 BPE + ByteLevel tokenizer，词表较小，默认 vocab size 是 6400。词表小的好处是 embedding 层和输出层参数量更少，适合 MiniMind 这种小模型；坏处是中文和复杂文本的压缩率不如大模型 tokenizer。

需要理解：

- 为什么 LLM 不能直接处理字符串？
- token id 是如何进入 embedding 层的？
- vocab size 为什么影响 embedding 和 lm_head 参数量？
- 为什么 tokenizer 改了，原来的模型权重通常就不能直接复用？

### 6.2 Special Tokens 与 Chat Template

MiniMind tokenizer 中包含：

```text
<|im_start|>
<|im_end|>
<tool_call>
</tool_call>
<tool_response>
</tool_response>
<think>
</think>
```

这些 token 决定了模型如何理解对话、思考标签和工具调用。

学习重点：

- `system/user/assistant/tool` 如何被拼成字符串？
- `apply_chat_template` 做了什么？
- 为什么 tool call 需要稳定的 XML/JSON 格式？
- 为什么 `<think>` 这种标签可以改变模型输出结构？

### 6.3 PretrainDataset

对应类：

```text
PretrainDataset
```

它的核心逻辑是：

```text
读取 text
-> tokenizer 编码
-> 加 bos/eos
-> padding 到 max_length
-> labels = input_ids.clone()
-> pad 位置 label 置为 -100
```

这里的 `-100` 很重要，因为 PyTorch 的 `F.cross_entropy` 可以通过 `ignore_index=-100` 忽略这些位置，不参与 loss。

你需要能解释：

- 为什么 pretrain 的 labels 可以直接等于 input_ids？
- 为什么 forward 里会用 `logits[..., :-1, :]` 对齐 `labels[..., 1:]`？
- 为什么 pad token 不参与 loss？

### 6.4 SFTDataset

对应类：

```text
SFTDataset
```

SFT 的关键不是数据长得像对话，而是 loss mask 不一样。

在 SFT 中，模型输入包含 system、user、assistant 的完整上下文，但 loss 通常只计算 assistant 回复部分。

原因是：

- system 和 user 是条件，不是模型要学习生成的目标。
- assistant 回复才是模型要学会的输出。
- 如果把 user 也纳入 loss，模型会学着生成用户问题，训练目标会混乱。

需要重点阅读：

```text
generate_labels
```

理解它如何找到 assistant 起始 token，并只给 assistant 内容设置 label，其余位置为 `-100`。

## 7. 阶段三：模型结构复盘

对应文件：

```text
model/model_minimind.py
```

这是你已经学过一部分的主线，但需要用源码重新复盘。

### 7.1 MiniMindConfig

需要理解这些参数：

```text
hidden_size
num_hidden_layers
vocab_size
num_attention_heads
num_key_value_heads
head_dim
intermediate_size
max_position_embeddings
rope_theta
use_moe
```

重点问题：

- `hidden_size` 决定什么？
- `num_hidden_layers` 增加会带来什么？
- `num_attention_heads` 和 `num_key_value_heads` 为什么可以不同？
- `intermediate_size` 为什么通常大于 hidden size？
- `vocab_size` 为什么直接影响输入 embedding 和输出 lm_head？

### 7.2 RMSNorm

MiniMind 使用 RMSNorm，而不是 LayerNorm。

RMSNorm 的作用是稳定激活分布，让训练更稳定。它不减均值，只根据均方根做归一化，计算更简单。

你需要掌握：

- 为什么 Transformer block 前要做 norm？
- pre-norm 和 post-norm 有什么区别？
- 为什么现代 LLM 常用 pre-norm？

### 7.3 RoPE 位置编码

对应函数：

```text
precompute_freqs_cis
apply_rotary_pos_emb
```

RoPE 的作用是把位置信息注入 q/k，使 attention 能感知 token 的相对位置。

需要理解：

- 为什么 self-attention 本身不包含顺序信息？
- RoPE 为什么作用在 q/k 上，而不是 v 上？
- MiniMind 中 YaRN scaling 是为了什么？
- 长上下文外推为什么可能不稳定？

### 7.4 GQA Attention

对应类：

```text
Attention
```

MiniMind 中：

```text
num_attention_heads = 8
num_key_value_heads = 4
```

这意味着 q 有 8 个 head，但 k/v 只有 4 个 head，需要通过 `repeat_kv` 复制，让 k/v 和 q 的 head 数对齐。

这就是 GQA 的思想：减少 k/v head 数，从而降低 KV cache 成本。

需要掌握：

- MHA、MQA、GQA 的区别。
- 为什么 KV cache 显存主要和 k/v 有关？
- `repeat_kv` 为什么能让 attention shape 对齐？
- `scaled_dot_product_attention` 和手写 attention 有什么区别？

### 7.5 FeedForward 与 SwiGLU

对应类：

```text
FeedForward
```

MiniMind 使用类似 SwiGLU 的结构：

```text
down_proj(act(gate_proj(x)) * up_proj(x))
```

它不是简单的两层 MLP，而是带门控的 FFN。

需要理解：

- FFN 在 Transformer 中的作用是什么？
- attention 负责 token 间交互，FFN 负责每个 token 内部的非线性变换。
- gate 分支为什么能增强表达能力？

### 7.6 MoE FeedForward

对应类：

```text
MOEFeedForward
```

MoE 的核心是：

```text
每个 token 通过 gate 选择 top-k 个 expert
只激活部分 expert
总参数量增加，但单 token 激活参数不同比例增加
```

需要理解：

- Dense 模型和 MoE 模型的区别。
- `num_experts` 和 `num_experts_per_tok` 的意义。
- router auxiliary loss 为什么需要？
- 为什么 MoE 训练可能因为 expert 分桶和 kernel 调度变慢？

### 7.7 MiniMindBlock

一个 block 的结构是：

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

这就是你之前学过的 pre-norm Transformer block。

需要做到：

- 能画出 block 内部结构图。
- 能写出每一步 shape。
- 能解释 residual connection 的作用。

### 7.8 MiniMindForCausalLM

对应类：

```text
MiniMindForCausalLM
```

它在 MiniMindModel 后面加了：

```text
lm_head: hidden -> vocab
```

并计算 causal language modeling loss。

需要理解：

- 为什么 logits 是 `[batch, seq, vocab]`？
- 为什么 loss 要错位对齐？
- `tie_word_embeddings` 为什么可以让 embedding 和 lm_head 共享权重？

## 8. 阶段四：自回归生成与 KV Cache

对应方法：

```text
MiniMindForCausalLM.generate
```

自回归生成的流程是：

```text
输入 prompt
-> 模型预测下一个 token 分布
-> 采样或 argmax 得到 next_token
-> 拼回 input_ids
-> 继续预测下一个 token
-> 直到 eos 或 max_new_tokens
```

需要理解的采样参数：

- `temperature`：控制分布尖锐程度。
- `top_k`：只保留概率最高的 k 个 token。
- `top_p`：只保留累计概率达到 p 的 token 集合。
- `repetition_penalty`：降低重复 token 的概率。
- `do_sample`：是否采样；否则用 argmax。

KV cache 的意义是：

```text
没有 cache：每生成一个 token 都重新计算整个序列的 k/v
有 cache：旧 token 的 k/v 保存下来，只计算新 token
```

你需要能回答：

- 为什么 KV cache 能加速生成？
- 为什么训练时通常不用 KV cache？
- 为什么长上下文会显著增加 KV cache 显存？
- 为什么 GQA 可以降低 KV cache 成本？

## 9. 阶段五：Pretrain 训练循环

对应文件：

```text
trainer/train_pretrain.py
```

预训练的目标是 next token prediction。模型通过大量文本学习语言规律、事实知识、格式模式和基础推理模式。

训练循环核心流程：

```text
读取 batch
-> input_ids.to(device)
-> labels.to(device)
-> model(input_ids, labels=labels)
-> loss = CE loss + aux_loss
-> loss / accumulation_steps
-> backward
-> 梯度累积
-> 梯度裁剪
-> optimizer.step
-> scaler.update
-> zero_grad
-> 保存 checkpoint
```

需要掌握：

- `epochs`、`batch_size`、`learning_rate` 的意义。
- `accumulation_steps` 为什么可以模拟更大的 batch。
- `grad_clip` 为什么能防止梯度爆炸。
- `autocast` 和 `GradScaler` 为什么能节省显存、提升速度。
- `AdamW` 和普通 SGD 的区别。
- checkpoint 为什么要保存 model、optimizer、scaler、epoch、step。

关键理解：

```text
Pretrain 不是让模型学会“回答问题”，而是让模型学会“语言和世界的统计规律”。
```

## 10. 阶段六：SFT 指令微调

对应文件：

```text
trainer/train_full_sft.py
dataset/lm_dataset.py
```

SFT 的目标是让模型从“会续写文本”变成“会按人类指令回答”。

虽然 SFT 和 pretrain 的底层 loss 都是交叉熵，但数据格式和 label mask 不一样，因此训练出来的行为完全不同。

核心区别：

| 阶段 | 输入 | 训练目标 |
|---|---|---|
| Pretrain | 普通文本 | 预测下一个 token |
| SFT | 对话模板 | 只学习 assistant 回复 |

需要重点理解：

- chat template 如何组织 system、user、assistant。
- assistant 部分如何被设置为 label。
- system/user 部分为什么是上下文，而不是预测目标。
- SFT 为什么会显著改变模型的交互体验。

建议实践：

- 打印一个 SFT 样本的 prompt。
- 打印每个 token 对应的 label。
- 验证哪些位置是 `-100`，哪些位置参与 loss。

## 11. 阶段七：推理、评测与部署

对应文件：

```text
eval_llm.py
scripts/chat_api.py
scripts/serve_openai_api.py
scripts/web_demo.py
```

这一阶段的目标是理解训练好的模型如何被使用。

需要掌握：

- 权重如何加载。
- tokenizer 如何应用 chat template。
- prompt 如何进入模型。
- 输出如何流式打印。
- OpenAI API 兼容接口如何包装本地模型。
- `max_tokens`、`temperature`、`top_p` 如何影响输出。

建议实践：

- 对比 pretrain 权重和 SFT 权重的回答差异。
- 固定 prompt，改变 temperature 和 top_p，观察输出变化。
- 记录模型在哪些问题上容易幻觉。

## 12. 阶段八：LoRA 微调

对应文件：

```text
model/model_lora.py
trainer/train_lora.py
```

LoRA 的核心思想是：

```text
原始线性层：y = Wx
LoRA 后：y = Wx + BAx
```

其中 `A` 和 `B` 是低秩矩阵，参数量远小于原始矩阵。

MiniMind 的实现逻辑是：

- 找到部分 `nn.Linear` 层。
- 给它们挂载 `lora` 模块。
- monkey patch 原来的 `forward`。
- 冻结原始参数。
- 只训练 LoRA 参数。
- 保存时只保存 LoRA 权重。

需要理解：

- 为什么 LoRA 参数少？
- 为什么 B 通常初始化为 0？
- 为什么冻结原模型可以降低训练成本？
- LoRA 适合做什么类型的垂类适配？
- LoRA 和 full SFT 的区别是什么？

和你未来 AI 教育 Agent 的关系：

```text
LoRA 可以用于快速适配某类教学风格、学科语料、题型格式或用户画像任务。
```

## 13. 阶段九：知识蒸馏

对应文件：

```text
trainer/train_distillation.py
```

蒸馏的目标是让小模型学习大模型或 MoE 模型的输出分布。

普通 CE loss 只告诉学生模型正确 token 是什么；蒸馏 loss 会告诉学生模型 teacher 对所有 token 的概率分布怎么看。

核心公式直觉：

```text
总 loss = alpha * CE loss + (1 - alpha) * KL distillation loss
```

需要理解：

- teacher model 为什么不更新参数？
- student model 为什么要同时学 ground truth 和 teacher distribution？
- temperature 为什么能软化概率分布？
- KL loss 和 CE loss 的区别是什么？
- 蒸馏为什么适合小模型？

和你未来方向的关系：

```text
你想做小模型处理 Agent 的部分任务，例如上下文压缩、学习记录总结、用户画像更新。蒸馏是把大模型能力迁移到小模型的重要方式。
```

## 14. 阶段十：DPO 偏好优化

对应文件：

```text
trainer/train_dpo.py
dataset/lm_dataset.py
```

DPO 使用偏好对数据：

```text
prompt
chosen response
rejected response
```

目标是让 policy model 更偏向 chosen，而不是 rejected。

MiniMind DPO 中有两个模型：

- policy model：当前要训练的模型。
- reference model：冻结的参考模型。

需要理解：

- 为什么需要 reference model？
- chosen 和 rejected 如何拼成 batch？
- 如何计算每个 response 的 log probability？
- `beta` 控制什么？
- DPO 为什么比 PPO 简洁？

DPO 的核心意义：

```text
SFT 教模型“模仿好答案”。
DPO 教模型“在好答案和坏答案之间更偏向好答案”。
```

## 15. 阶段十一：PPO、GRPO、CISPO

对应文件：

```text
trainer/train_ppo.py
trainer/train_grpo.py
```

这一阶段建议不要一开始追公式细节，而是先理解强化学习在 LLM 中解决什么问题。

LLM 强化学习的一般结构：

```text
prompt
-> 当前模型生成 response
-> reward model 或规则给分
-> 根据 reward 更新模型
-> 用 KL 限制模型不要偏离 reference 太远
```

### 15.1 PPO

PPO 通常包含：

- Actor：生成回答。
- Critic：估计价值。
- Reward：评价回答。
- Advantage：判断比预期好还是差。
- KL penalty：限制模型漂移。

优点是经典、成熟；缺点是复杂、显存占用高、训练不稳定因素多。

### 15.2 GRPO

GRPO 的核心是：

```text
对同一个 prompt 生成多个回答
用组内平均 reward 当 baseline
高于平均的回答被鼓励
低于平均的回答被抑制
```

它不需要单独训练 critic，因此比 PPO 更轻。

需要理解：

- 为什么要 `num_generations`？
- 为什么同一个 prompt 要生成多个 response？
- 组内标准化 advantage 的意义是什么？
- 为什么 reward 方差太小时学习信号会消失？

### 15.3 CISPO

CISPO 可以理解成 GRPO/PPO 类 loss 的一个变体，重点在于改善 clip 之后梯度路径被截断的问题。现阶段你只需要知道它是对策略优化 loss 的改进，不需要一开始深挖公式。

## 16. 阶段十二：Agentic RL 与 Tool Use

对应文件：

```text
trainer/train_agent.py
trainer/rollout_engine.py
scripts/eval_toolcall.py
```

MiniMind 的 Agentic RL 是一个轻量版本，但很适合学习 Agent 训练的基本组成。

Agentic RL 的关键不是单轮回答，而是多轮轨迹：

```text
用户问题
-> 模型决定是否调用工具
-> 输出 tool_call
-> 环境执行工具
-> 返回 tool_response
-> 模型继续回答
-> 最终根据结果给 reward
```

需要理解：

- 工具 schema 如何写？
- 模型如何生成 `<tool_call>`？
- 工具执行结果如何作为下一轮上下文？
- reward 为什么是延迟的？
- 什么是 rollout？
- 为什么要做训推分离？

MiniMind 中的工具包括：

```text
calculate_math
unit_converter
get_current_weather
get_current_time
get_exchange_rate
translate_text
```

这些工具虽然是模拟的，但足够帮助你理解真实 Agent 系统。

和你未来 AI 教育 Agent 的关系：

```text
教育 Agent 也可以设计工具：
- 查询学习计划
- 写入学习记录
- 检索知识点
- 生成错题
- 更新用户画像
- 安排复习任务
- 检查作业答案
```

MiniMind 的 Agentic RL 可以作为你未来教育 Agent 工具调用训练的缩小版原型。

12 周执行路线和进度管理已单独沉淀到：

```text
model_training_12_weeks/minimind_12_week_roadmap.md
```

## 17. 面向 AI 教育 Agent 的迁移思路

你未来想做 AI 教育相关 Agent，目标是知识平权，让任何人都能轻松学习和管理学习任务。MiniMind 可以为这个方向提供底层训练和系统设计启发。

### 17.1 教育 Agent 不应该只是问答机器人

单纯的问答机器人很容易被通用大模型替代。更有价值的是学习操作系统：

```text
诊断当前水平
-> 制定学习路径
-> 解释知识点
-> 生成练习
-> 批改反馈
-> 记录错因
-> 安排复习
-> 更新用户画像
-> 长期追踪目标
```

### 17.2 小模型可以负责的任务

小模型不一定负责最终教学回答，而是可以负责大量后台任务：

- 意图识别
- 学科分类
- 知识点标签
- 错因分类
- 学习记录摘要
- 上下文压缩
- 用户画像更新
- 复习任务生成
- 多模态材料的轻量解析

这些任务高频、低成本、隐私敏感，适合小模型。

### 17.3 MiniMind 对教育 Agent 的启发

MiniMind 中的模块可以对应到教育 Agent：

| MiniMind 模块 | 教育 Agent 迁移 |
|---|---|
| SFT | 训练教学风格、答题格式、解释方式 |
| LoRA | 快速适配学科、考试、用户群体 |
| 蒸馏 | 把大模型教学能力压缩到小模型 |
| DPO | 让模型偏向更清晰、更耐心、更适合学习者的回答 |
| GRPO | 优化可验证任务，例如数学题、代码题、选择题 |
| Agentic RL | 训练模型调用学习工具、更新计划、查询错题、安排复习 |
| rollout_engine | 模拟学习过程中的多轮交互 |
| reward function | 根据答案正确性、解释质量、学习进度给反馈 |

### 17.4 一个最小教育 Agent 实验

可以设计如下工具：

```text
get_user_profile
update_user_profile
retrieve_knowledge_point
generate_exercise
check_answer
schedule_review
summarize_session
```

一个训练样本可以是：

```text
用户：我今天学了 Transformer attention，但是还是不理解 QKV。
Agent：
1. 查询用户画像，发现用户已学过矩阵乘法和 CNN。
2. 检索 attention 知识点。
3. 用 CNN 类比解释 QKV。
4. 生成一道小练习。
5. 根据用户回答更新薄弱点。
6. 安排明天复习。
```

这个任务的 reward 可以包括：

- 是否调用了正确工具。
- 是否正确识别薄弱知识点。
- 是否给出合适解释。
- 是否生成了难度合适的练习。
- 是否更新用户画像。
- 是否安排复习。

这就是 MiniMind Agentic RL 在教育场景中的迁移版本。

## 18. 每周复盘模板

每周学习结束后，建议写一次复盘：

```text
本周主题：

我读了哪些文件：

我真正理解的 3 个知识点：

我还不理解的 3 个问题：

我能解释的一条数据流：

我能解释的一条 tensor shape：

我能跑通或修改的代码：

这一周和 AI 教育 Agent 方向的关系：

下周计划：
```

这个模板的作用是防止学习变成“看了很多但没有沉淀”。每周至少留下一份可复用笔记。

## 19. 学习完成标准

当你能独立回答下面这些问题时，就说明 MiniMind 第一轮学习基本完成：

- 一个 JSONL 样本如何变成 `input_ids` 和 `labels`？
- `-100` 在 loss 中起什么作用？
- 为什么 SFT 只训练 assistant token？
- `input_ids` 进入模型后，每一步 shape 如何变化？
- RMSNorm、RoPE、GQA、SwiGLU 分别解决什么问题？
- KV cache 为什么能加速生成？
- Pretrain 和 SFT 的 loss 都是 CE，为什么模型行为不同？
- LoRA 为什么可以低成本适配垂类任务？
- 蒸馏中的 temperature 有什么意义？
- DPO 为什么需要 reference model？
- GRPO 为什么可以不用 critic？
- Agentic RL 中 rollout 是什么？
- Tool Use 的 reward 为什么是延迟的？
- MiniMind 的哪些模块可以迁移到 AI 教育 Agent？

如果这些问题能用自己的话讲清楚，并且能指到对应源码位置，就说明你已经不是“会跑项目”，而是开始真正理解 LLM 的训练闭环了。

## 20. 建议的学习态度

这条路线不要急着追求“最快跑出效果”。你的目标是从 CV 迁移到 LLM 和 Agent，小模型只是入口，真正要建立的是底层理解：

```text
数据如何塑造模型
结构如何承载能力
loss 如何改变行为
reward 如何改变策略
工具如何扩展模型
记忆和上下文如何形成长期智能
```

MiniMind 是一个很好的练习场。它小，但链路完整；简单，但不玩具。把它吃透之后，你再去看 Qwen、Llama、DeepSeek、Claude Code、Agent framework、小模型蒸馏和教育 Agent，很多概念都会自然连起来。
