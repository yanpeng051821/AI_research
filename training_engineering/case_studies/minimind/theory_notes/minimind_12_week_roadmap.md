# MiniMind 12 周学习路线图与进度管理

> 2026-09-27：本文保留第一轮理论学习与历史进度。本轮独立实现、数据实践和亲自运行实验的
> 安排见 [实践路线](../PRACTICE_ROADMAP.md)；历史完成标记不自动计入本轮实践验收。

## 1. 使用方式

这份文档是 MiniMind 学习计划的执行版，用来管理每周进度、产出、阻塞点和复盘记录。完整知识体系见：

```text
model_training_12_weeks/minimind_learning_plan.md
```

建议每周只推进一个主题。每周结束时，更新本文件中的：

- `状态`
- `完成日期`
- `本周产出`
- `阻塞点`
- `复盘记录`

状态建议使用：

```text
未开始 / 进行中 / 已完成 / 需复盘 / 暂停
```

节奏建议：

- 每周只保证完成“核心 checklist”，加餐任务可以顺延。
- 如果某周只能投入较少时间，优先完成“能讲清楚数据流或 shape”的产出。
- 第 3、4、11 周概念密度较高，允许拆成两周完成，不影响总路线。
- 每 3 周做一次小复盘，确认前一阶段的输入、输出、loss 和 shape 已经讲得清楚。
- 不建议为了赶进度跳过阻塞点。LLM 学习最怕“概念链断了但继续往后看”。

## 2. 总进度看板

| 周次 | 主题 | 状态 | 完成日期 | 核心产出 |
|---|---|---|---|---|
| 第 1 周 | 项目地图与环境理解 | 已完成 | 2026-06-19 | 项目结构图、文件职责笔记 |
| 第 2 周 | Tokenizer 与数据样本 | 已完成 | 2026-06-19 | pretrain/SFT 样本解析 |
| 第 3 周 | 模型主干复盘 | 已完成 | 2026-06-19 | forward shape 表 |
| 第 4 周 | Attention、RoPE、KV Cache | 已完成 | 2026-06-20 | attention shape 图、KV cache 说明 |
| 第 5 周 | Pretrain 训练循环 | 已完成 | 2026-06-20 | `train_epoch` 逐行解释 |
| 第 6 周 | SFT 训练循环 | 已完成 | 2026-06-24 | assistant-only loss 说明 |
| 第 6.5 周 | 训练代码手撕复盘 | 已完成 | 2026-06-30 | pretrain/SFT 训练骨架、Dataset/labels 补充 |
| 第 7 周 | 推理与采样 | 已完成 | 2026-06-30 | 自回归生成、采样参数、KV cache、streamer、baseline 评估 |
| 第 7.5 周 | 实验工程与评估流水线 | 已完成 | 2026-06-30 | baseline、指标、复现实验记录 |
| 第 8 周 | LoRA | 已完成 | 2026-07-04 | LoRA 公式、参数量解释、模块注入、forward patch |
| 第 9 周 | 知识蒸馏 | 已完成 | 2026-07-05 | teacher/student、CE/KL、alpha、temperature |
| 第 10 周 | DPO | 已完成 | 2026-07-06 | chosen/rejected、policy/reference、DPO loss、完整数据流 |
| 第 11 周 | GRPO 与 RL 入门 | 进行中 |  | rollout、reward、advantage、KL、PPO 前置 |
| 第 12 周 | Agentic RL 与教育 Agent 迁移 | 未开始 |  | 教育 Agent 最小实验设计 |

## 3. 阶段节奏与缓冲规则

这份路线图按 12 周设计，但它不是考试倒计时。更推荐把它当成 12 个学习单元：

```text
输入链路：第 1-2 周
模型主干：第 3-4 周
训练闭环：第 5-7 周
实验工程：第 7.5 周
后训练：第 8-11 周
教育 Agent 迁移：第 12 周
多模态与视觉小模型：MiniMind 主线完成后的后续支线
```

每个阶段结束时，如果里程碑检查没有通过，就把下一周改成“复盘周”，而不是继续推进新主题。

建议复盘节点：

| 复盘节点 | 位置 | 检查重点 |
|---|---|---|
| 第 1 次复盘 | 第 2 周后 | 文本如何变成 `input_ids`、`labels`、`loss mask` |
| 第 2 次复盘 | 第 4 周后 | 模型 forward shape、attention、RoPE、KV cache |
| 第 3 次复盘 | 第 7.5 周后 | pretrain/SFT/generate/experiment 的完整闭环 |
| 第 4 次复盘 | 第 11 周后 | LoRA、蒸馏、DPO、GRPO 的目标差异 |

任务优先级分为两类：

```text
核心任务：本周必须完成，否则不建议进入下一周。
加餐任务：理解更深会更好，但可以顺延。
```

## 4. 当前已计入进度

你已经学过 LLM forward 主线的一部分：

```text
input -> tokenizer -> embedding -> pre_norm -> attention -> pre_norm -> ffn -> out_proj -> logits -> 自回归生成 + KV cache
```

因此下面模块可以视为“已入门，但需要用 MiniMind 源码复盘”：

- Tokenizer 在输入链路中的位置。
- Embedding 如何把 token id 映射到 hidden state。
- Pre-norm Transformer block 的基本结构。
- Attention 和 FFN 在 block 中的位置。
- Logits 如何对应 vocab 分布。
- 自回归生成的基本流程。
- KV cache 的基本作用。

换句话说，第 3 周和第 4 周不是从零开始，而是把已学知识落实到 MiniMind 源码和 tensor shape 上。

## 5. 第 1 周：项目地图与环境理解

状态：已完成  
完成日期：2026-06-19  

### 目标

- 读完 `README.md`。
- 搞清楚每个目录的作用。
- 画出 MiniMind 全流程图。
- 明确本项目为什么适合从 0 到 1 学 LLM。

### 学习文件

```text
README.md
requirements.txt
```

### 必须理解

- MiniMind 项目的定位是什么。
- MiniMind 包含哪些训练阶段。
- `model/`、`dataset/`、`trainer/`、`scripts/` 分别负责什么。
- 为什么本项目强调不用高级框架封装核心训练逻辑。

### Checklist

- [x] 读完 README 的项目介绍。
- [x] 读完 README 的快速开始。
- [x] 读完 README 的数据介绍。
- [x] 读完 README 的模型结构介绍。
- [x] 读完 README 的主要训练流程。
- [x] 列出每个核心文件的职责。
- [x] 画出 MiniMind 从数据到推理的流程图。

核心任务：

- [x] 能说清楚 `model/`、`dataset/`、`trainer/`、`scripts/` 的职责。
- [x] 能画出“数据 -> 模型 -> loss -> 推理”的粗流程。

加餐任务：

- [ ] 阅读 README 中 RL、Agentic RL、部署相关章节，先建立整体印象。

### 本周产出

```text
1. 一张项目结构图
2. 一份“每个文件负责什么”的笔记
```

### 阻塞点

```text

```

### 复盘记录

```text

```

## 6. 第 2 周：Tokenizer 与数据样本

状态：已完成  
完成日期：2026-06-19  

### 目标

- 理解 tokenizer。
- 理解 JSONL 数据格式。
- 理解 `input_ids`、`labels`、`-100`。
- 理解 pretrain 样本和 SFT 样本的区别。

### 学习文件

```text
trainer/train_tokenizer.py
dataset/lm_dataset.py
model/tokenizer.json
model/tokenizer_config.json
```

### 必须理解

- 为什么语言模型不能直接处理字符串。
- BPE + ByteLevel tokenizer 的基本意义。
- special tokens 如何支持 chat、thinking 和 tool call。
- `PretrainDataset` 如何构造 labels。
- `SFTDataset` 如何只训练 assistant 部分。
- `-100` 为什么能让某些 token 不参与 loss。

### Checklist

- [x] 理解 tokenizer encode/decode 的基本流程。
- [x] 理解 `apply_chat_template` 的作用。
- [x] 读懂 `PretrainDataset.__getitem__`。
- [x] 读懂 `SFTDataset.create_chat_prompt`。
- [x] 读懂 `SFTDataset.generate_labels`。
- [ ] 打印一个 pretrain 样本的 token 和 label。
- [ ] 打印一个 SFT 样本的 token、role 和 label mask。

核心任务：

- [x] 能解释 `input_ids`、`labels`、`-100`。
- [x] 能解释 pretrain 样本和 SFT 样本的 label 差异。

加餐任务：

- [ ] 阅读 tokenizer_config 中的 chat template，并标出 tool/thinking 相关片段。

### 本周产出

```text
1. pretrain 样本解析笔记
2. SFT 样本解析笔记
3. `-100` 和 assistant-only loss 说明
```

### 阻塞点

```text
当前 shell 环境缺少 transformers，暂未打印真实 tokenizer 输出。该任务顺延到环境配置或训练前检查阶段。
```

### 复盘记录

```text
已能解释 conversations -> chat_template -> prompt -> tokenizer -> input_ids -> generate_labels -> labels 的完整链路。
已能区分 pretrain 只屏蔽 PAD，SFT 屏蔽 user/system/role/pad，只学习 assistant 内容和结束符。
```

## 7. 第 3 周：模型主干复盘

状态：已完成  
完成日期：2026-06-19  

### 目标

- 复盘 MiniMindConfig、RMSNorm、Embedding、Block、LM head。
- 写出每一步 tensor shape。
- 把你之前学过的 forward 主线和 MiniMind 源码对应起来。

### 学习文件

```text
model/model_minimind.py
```

### 必须理解

- `MiniMindConfig` 中关键参数的意义。
- `nn.Embedding` 如何处理 `input_ids`。
- `MiniMindModel` 如何堆叠多个 block。
- `MiniMindBlock` 中 attention 和 FFN 的顺序。
- `MiniMindForCausalLM` 如何从 hidden state 得到 logits。
- `labels` 和 logits 如何错位计算 CE loss。

### Checklist

- [x] 读懂 `MiniMindConfig`。
- [x] 读懂 `RMSNorm`。
- [x] 读懂 `MiniMindBlock`。
- [x] 读懂 `MiniMindModel.forward`。
- [x] 读懂 `MiniMindForCausalLM.forward`。
- [x] 写出 `[B,T] -> [B,T,C] -> [B,T,V]` 的完整 shape。
- [x] 解释为什么 logits 和 labels 要错位。

核心任务：

- [x] 能写出 `input_ids -> embedding -> blocks -> norm -> lm_head -> logits -> loss` 的 shape。
- [x] 能解释 residual、RMSNorm、lm_head 的作用。

加餐任务：

- [ ] 追一次 `tie_word_embeddings` 对参数量的影响。

### 本周产出

```text
1. forward shape 表
2. 一个最小 batch 的手动前向追踪笔记
```

### 阻塞点

```text

```

### 复盘记录

```text
已能复述 MiniMindForCausalLM / MiniMindModel / MiniMindBlock 的分工。
已能解释 input_ids [B,T] -> embedding [B,T,C] -> blocks [B,T,C] -> lm_head [B,T,V] -> shift loss。
已能说明 residual 是增量更新和梯度通路，RMSNorm 用于稳定尺度，lm_head 用于映射到词表空间。
```

## 8. 第 4 周：Attention、RoPE、KV Cache

状态：已完成  
完成日期：2026-06-20  

### 目标

- 理解 q/k/v。
- 理解 GQA。
- 理解 RoPE。
- 理解 generate 中的 KV cache。

### 学习文件

```text
model/model_minimind.py
```

### 必须理解

- q/k/v 的 shape 如何变化。
- `num_attention_heads` 和 `num_key_value_heads` 为什么可以不同。
- `repeat_kv` 如何让 GQA 的 k/v 和 q 对齐。
- RoPE 为什么作用在 q/k 上。
- `past_key_values` 如何保存历史 k/v。
- KV cache 为什么能加速生成。

### Checklist

- [x] 读懂 `Attention.forward`。
- [x] 读懂 `repeat_kv`。
- [ ] 读懂 `precompute_freqs_cis`。
- [x] 读懂 `apply_rotary_pos_emb`。
- [x] 读懂 `generate` 中的 `past_key_values` 更新。
- [x] 画出 attention scores 的 shape。
- [x] 解释 GQA 如何降低 KV cache 成本。

核心任务：

- [x] 能写出 q/k/v 和 attention scores 的 shape。
- [x] 能解释 RoPE、GQA、KV cache 分别解决什么问题。

加餐任务：

- [ ] 阅读 YaRN scaling 相关代码，理解它和长上下文外推的关系即可，不要求推公式。

### 本周产出

```text
1. Attention shape 图
2. KV cache 加速原理说明
3. RoPE 和 GQA 的源码注释笔记
```

### 阻塞点

```text

```

### 复盘记录

```text
已能写出 x [B,T,C] -> q/k/v -> RoPE -> repeat_kv -> scores [B,H,T,T] -> output [B,T,C] 的主线。
已能解释 GQA 中 q heads 多、k/v heads 少，通过 repeat_kv 让多个 q head 共享 k/v，从而降低 KV cache 成本。
已能解释 RoPE 作用在 q/k 而非 v，KV cache 只缓存历史 k/v 而不缓存 q。
YaRN scaling 公式细节暂不深挖，后续长上下文专题再补。
```

## 9. 第 5 周：Pretrain 训练循环

状态：已完成  
完成日期：2026-06-20  

### 目标

- 理解 pretrain dataset。
- 理解训练循环。
- 理解 optimizer、lr、grad accumulation、AMP。

### 学习文件

```text
trainer/train_pretrain.py
trainer/trainer_utils.py
```

### 必须理解

- `train_epoch` 每一步在做什么。
- `loss / accumulation_steps` 的意义。
- 什么时候执行 `backward`，什么时候执行 `optimizer.step`。
- 为什么要做梯度裁剪。
- 为什么要保存 checkpoint 和 resume checkpoint。
- `get_lr` 的余弦学习率变化。

### Checklist

- [x] 读懂参数解析部分。
- [x] 读懂模型和 tokenizer 初始化。
- [x] 读懂 DataLoader 构造。
- [x] 读懂 `train_epoch`。
- [x] 读懂 checkpoint 保存和恢复。
- [x] 解释 AMP 的作用。
- [x] 解释梯度累积的作用。

核心任务：

- [x] 能逐行讲清楚 `train_epoch` 的主流程。
- [x] 能解释 loss、backward、optimizer.step 的关系。

加餐任务：

- [ ] 理解 DDP 和 resume checkpoint 的实现细节。

### 本周产出

```text
1. `train_epoch` 逐行解释
2. loss 如何产生、梯度如何更新参数的说明
```

### 阻塞点

```text

```

### 复盘记录

```text
已能解释 Dataset 构造 input_ids/labels，DataLoader 组成 batch，model 前向得到 loss，loss/accumulation_steps 后 backward，累积够 step 后 unscale、clip、optimizer.step、zero_grad。
已能解释 autocast、GradScaler、gradient accumulation、clip_grad_norm_、get_lr 的作用。
已能区分 out/*.pth 是模型成果，checkpoints/*_resume.pth 是训练现场快照。
已对比 train_pretrain.py 和 train_full_sft.py：训练引擎相似，核心差异在 Dataset、数据文件、默认 from_weight、学习率和训练目标。
服务器环境已初步检查：Tesla T4 16GB、CUDA 12.2、torch 2.4.1、transformers 4.57.6、datasets 3.6.0，mini 数据文件已存在。下一步建议做受控 smoke test 和正式训练参数规划。
```

## 10. 第 6 周：SFT 训练循环

状态：已完成  
开始日期：2026-06-20  
完成日期：2026-06-24  

### 目标

- 理解 SFT 与 pretrain 的区别。
- 理解 assistant-only loss。
- 理解 chat template 对训练行为的影响。

### 学习文件

```text
trainer/train_full_sft.py
dataset/lm_dataset.py
model_training_12_weeks/week_06_sft_training_loop.md
```

### 必须理解

- 为什么 SFT 仍然使用 CE loss。
- 为什么 SFT 的数据格式和 label mask 会改变模型行为。
- system/user/assistant 在训练中的不同角色。
- full SFT 和 pretrain 训练脚本为什么很像，但训练目标不同。

### Checklist

- [x] 对比 `train_pretrain.py` 和 `train_full_sft.py`。
- [x] 对比 `PretrainDataset` 和 `SFTDataset`。
- [x] 解释 assistant-only loss。
- [x] 解释 chat template 如何影响模型输出格式。
- [ ] 打印 SFT 样本并标出参与 loss 的 token。

核心任务：

- [x] 能解释为什么 SFT 只训练 assistant token。
- [x] 能对比 pretrain 和 SFT 的样本、label、训练目标。

加餐任务：

- [ ] 观察 tool call 样本在 SFT 数据中如何被模板化。

### 本周产出

```text
1. pretrain 和 SFT 的对比表
2. assistant-only loss 说明
3. 为什么同样是 CE loss，模型行为不同的解释
4. `model_training_12_weeks/week_06_sft_training_loop.md`
```

### 阻塞点

```text
真实 SFT 训练已跑完。真实 SFT 样本 token/label mask 打印任务可在第 6.5 周或第 7 周作为补充实验完成。
```

### 复盘记录

```text
已确认 SFT 与 pretrain 的训练引擎基本一致，核心区别在 SFTDataset、chat_template、assistant-only labels、较小学习率和 from_weight 加载 pretrain 权重。
服务器 SFT 已跑完。下一步先插入第 6.5 周训练代码手撕复盘，再进入 eval_llm.py 和采样参数。
```

## 10.5 第 6.5 周：训练代码手撕复盘

状态：已完成  
开始日期：2026-06-24  
完成日期：2026-06-30

### 目标

- 按执行顺序剖开 `train_pretrain.py` 和 `train_full_sft.py`。
- 建立自己的训练脚本骨架。
- 理解训练主线和工程细节的层级关系。

### 学习文件

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/trainer_utils.py
model_training_12_weeks/week_06_5_training_code_dissection.md
```

### 必须理解

- `main` 函数如何串起 config、model、dataset、optimizer、loader。
- `train_epoch` 中一个 batch 如何完成 forward、loss、backward、step。
- gradient accumulation、autocast、GradScaler、clip、lr schedule 分别解决什么问题。
- `from_weight` 和 `from_resume` 在训练流程中的位置。
- `out/*.pth` 和 `checkpoints/*_resume.pth` 的保存内容和用途。

### Checklist

- [x] 新增 `model_training_12_weeks/week_06_5_training_code_dissection.md`。
- [x] 按执行顺序讲清楚 `train_full_sft.py`。
- [x] 对照指出 `train_pretrain.py` 与 `train_full_sft.py` 的最小差异。
- [x] 能手写一个极简训练循环骨架。
- [x] 能解释训练工程细节的先后顺序。

核心任务：

- [x] 能独立画出 Dataset -> DataLoader -> model -> loss -> backward -> optimizer.step -> save 的训练闭环。
- [x] 能解释后续 LoRA / DPO / GRPO 为什么会复用这套训练骨架。

加餐任务：

- [ ] 打印一个真实 SFT 样本的 token 与 label mask，补齐第 6 周遗留实验。

### 本周产出

```text
1. `model_training_12_weeks/week_06_5_training_code_dissection.md`
2. 一份极简训练脚本骨架
3. pretrain/SFT 训练代码差异表
4. PretrainDataset / SFTDataset 与 labels 构造补充
```

### 阻塞点

```text

```

### 复盘记录

```text
已决定先进行训练代码手撕复盘，再进入第 7 周推理与采样。这样后续学习 LoRA、蒸馏、DPO、GRPO 时会有更扎实的工程底座。
已完成 pretrain 和 SFT 训练代码手撕复盘，能够完整理解 `train_pretrain.py` / `train_full_sft.py` 的主流程、训练工程细节、checkpoint/resume、以及 Dataset -> DataLoader -> model -> loss -> backward -> optimizer.step -> save 的完整闭环。

训练主线已完成：能解释 `train_full_sft.py` 的执行顺序、pretrain/SFT 的最小差异、from_weight/from_resume、普通权重和 resume checkpoint 的区别。

已补充 PretrainDataset / SFTDataset、chat_template、assistant-only labels、模型内部 shift loss、autocast、GradScaler 等关键概念。

真实 SFT 样本 token 与 label mask 打印作为加餐任务顺延到第 7.5 周实验工程。
```

## 11. 第 7 周：推理与采样

状态：已完成
完成日期：2026-06-30

### 目标

- 理解模型如何生成回答。
- 理解采样参数。
- 理解流式输出。

### 学习文件

```text
eval_llm.py
scripts/chat_api.py
scripts/serve_openai_api.py
model_training_12_weeks/week_07_inference_sampling.md
```

### 必须理解

- 推理时如何加载模型和 tokenizer。
- prompt 如何通过 chat template 进入模型。
- `temperature`、`top_p`、`top_k` 如何改变输出。
- repetition penalty 如何减少重复。
- stream 输出和非 stream 输出的区别。

### Checklist

- [x] 新增 `model_training_12_weeks/week_07_inference_sampling.md`。
- [x] 读懂 `eval_llm.py` 的推理入口。
- [x] 读懂 `scripts/chat_api.py` 的客户端请求主线。
- [x] 初步读懂 `scripts/serve_openai_api.py` 的流式输出主线。
- [x] 理解 `max_tokens` 和 `max_new_tokens`。
- [x] 对比不同采样参数下的输出。
- [x] 记录模型容易重复或幻觉的场景。

核心任务：

- [x] 能解释自回归生成循环。
- [x] 能解释 temperature、top-p、top-k、repetition penalty 的作用。
- [x] 能解释 KV cache / past_key_values 的作用。
- [x] 能解释 streamer 如何实现边生成边输出。

加餐任务：

- [ ] 尝试对比 pretrain 权重和 SFT 权重的输出风格差异。

### 本周产出

```text
1. `model_training_12_weeks/week_07_inference_sampling.md`
2. eval_llm_report.py
3. eval_reports/full_sft_768_20260630_183627.md
4. 采样参数实验记录
5. 模型幻觉和重复问题总结
```

### 阻塞点

```text

```

### 复盘记录

```text
已完成 eval_llm.py 推理入口、generate 自回归循环、KV cache、attention_mask/causal mask/labels mask、temperature/top-k/top-p/repetition penalty 的学习。
已完成推理主线学习：模型加载、prompt/chat_template、`add_generation_prompt`、自回归生成、logits/softmax、KV cache、streamer。
新增结构化评估脚本 eval_llm_report.py，并使用 full_sft_768 跑出 safe/balanced/creative 三组 baseline。
结论：当前 full_sft_768 已具备基本对话格式能力和一定实时边界意识，但专业知识、代码能力、教学解释和指令约束仍较弱。safe 参数最稳，creative 更容易幻觉。
下一步进入第 7.5 周实验工程，把这次报告作为后续 LoRA/DPO/数据改进的 baseline。
```

## 11.5 第 7.5 周：实验工程与评估流水线

状态：已完成
开始日期：2026-06-30  
完成日期：2026-06-30

### 目标

- 建立最小可用实验流水线。
- 学会设置 baseline、变量、指标和复现实验记录。
- 为后续 LoRA、蒸馏、DPO、GRPO 准备统一评估方法。

### 学习文件

```text
model_training_12_weeks/week_07_5_experiment_engineering.md
eval_llm.py
trainer/train_pretrain.py
trainer/train_full_sft.py
```

### 必须理解

- 实验不是只跑模型，而是固定条件下比较变量。
- 没有 baseline，就很难判断一个改动是否真的有效。
- 训练指标、推理指标、任务指标分别回答不同问题。
- 复现实验需要记录代码版本、数据、权重、命令、参数和输出样例。

### Checklist

- [x] 建立 `model_training_12_weeks/experiments/` 实验记录目录。
- [x] 写一份 pretrain vs SFT 的对比实验记录。
- [x] 固定一组评估 prompts。
- [x] 新建采样参数实验记录。
- [x] 跑完一次采样参数实验并填写结果。
- [x] 总结至少 3 个模型失败案例。

核心任务：

- [x] 能设计 baseline vs experiment 的对比。
- [x] 能说明一个实验中的固定变量和变化变量。
- [x] 能同时记录质量、速度、显存和失败案例。

加餐任务：

- [ ] 为 AI 教育 Agent 设计一个 20 条 prompt 的小评测集。

### 本周产出

```text
1. `model_training_12_weeks/week_07_5_experiment_engineering.md`
2. `model_training_12_weeks/experiments/` 实验记录目录
3. `model_training_12_weeks/experiments/exp_001_pretrain_vs_sft.md`
4. `model_training_12_weeks/experiments/exp_002_sampling_params.md`
```

### 阻塞点

```text

```

### 复盘记录

```text
第 7.5 周完成了两个最小实验：

1. exp_001_pretrain_vs_sft
   目标：比较 pretrain 和 SFT 的能力差异。
   结论：pretrain 更像续写模型；SFT 更像助手模型，但并不能保证知识、代码和停止控制正确。

2. exp_002_sampling_params
   目标：比较不同采样参数对输出稳定性的影响。
   结论：采样参数决定“怎么说”，不决定“会不会”。代码能力不足、事实错误、复读倾向不能主要靠 temperature/top_p 解决。

本周确认了 3 类典型失败案例：

失败案例 1：复读和停不下来
表现：机器学习解释中多次重复“从数据中学习，并从中学习”，没有严格遵守“三句话”。
原因猜测：模型规模小、SFT 数据可能存在模板化表达、停止 token/长度控制能力弱。
后续方向：使用更好的 SFT 数据；尝试 repetition_penalty；后续用 DPO 偏好“简洁、不复读”的答案。

失败案例 2：代码格式像代码，但语义错误
表现：斐波那契函数输出了代码块，但混入 factorial、欧几里得算法、伪代码或乱码式代码。
原因猜测：SFT 学到了“代码请求要输出代码块”的表面格式，但没有足够代码知识和可执行逻辑能力。
后续方向：代码数据 SFT、代码垂类 LoRA、执行器辅助评估、用可运行性作为实验指标。

失败案例 3：实时信息与工具能力边界
表现：天气问题基本能说明无法获取实时天气，但有时会说“我可以帮助查询”，暗示自己有工具能力。
原因猜测：普通 SFT 中没有严格区分“无工具回答”和“工具调用回答”。
后续方向：Agent 阶段引入真实工具调用；普通对话数据中加入更严格的实时信息边界样本。

第 7.5 周能力进展：

已经能把模型训练结果放进实验框架里比较，而不是只凭感觉判断模型好坏。
已经能区分模型能力问题和采样参数问题。
已经能把失败案例转化成后续 LoRA、DPO、Agent 工具调用的实验目标。
```

## 12. 第 8 周：LoRA

状态：已完成
开始日期：2026-06-30  
完成日期：2026-07-04

### 目标

- 理解低秩适配。
- 理解冻结原模型、只训练 LoRA 参数。
- 理解 LoRA 权重保存与合并。

### 学习文件

```text
model/model_lora.py
trainer/train_lora.py
scripts/convert_model.py
model_training_12_weeks/week_08_lora.md
```

### 必须理解

- LoRA 的 `W + BA` 形式。
- 为什么 LoRA 参数量小。
- 为什么 B 通常初始化为 0。
- MiniMind 如何 monkey patch `Linear.forward`。
- 为什么 LoRA 适合垂类适配。

### Checklist

- [x] 读懂 `LoRA` 类。
- [x] 读懂 `apply_lora`。
- [x] 读懂 `save_lora`。
- [x] 读懂 `merge_lora`。
- [x] 读懂 `train_lora.py` 如何冻结非 LoRA 参数。
- [x] 计算一次 LoRA 参数占比。

核心任务：

- [x] 能解释 `W + BA`。
- [x] 能解释 LoRA 接入原模型的位置和方式。
- [x] 能解释为什么只训练 LoRA 参数可以适配垂类任务。

加餐任务：

- [ ] 设计一个 AI 教育场景的 LoRA 数据样本。

### 本周产出

```text
1. LoRA 公式手写说明
2. LoRA 参数量为什么小的解释
3. 一个教育垂类 LoRA 数据格式设计
```

### 阻塞点

```text
已完成 LoRA 主线：理解 LoRA 的 `W + BA` 低秩增量形式，知道为什么只训练少量 A/B 参数就能进行垂类适配。

已理解 MiniMind 的实现方式：`apply_lora` 会遍历模型中的目标 Linear，为其挂载 `lora` 子模块，并 patch 原始 Linear 的 forward，让输出变成原始线性输出加 LoRA 增量。

已澄清 `setattr(module, "lora", lora)` 后 `model.named_modules()` 的行为：遍历时既会看到带有 `.lora` 属性的原 Linear，也会看到作为子模块注册进去的 LoRA 模块本身；训练时主要通过 `hasattr(module, "lora")` 找到父 Linear。

已理解 Linear 自身也有 forward，LoRA patch 的是被选中的 Linear 子模块 forward，不是整个模型的 forward。
```

### 复盘记录

```text

```

## 13. 第 9 周：知识蒸馏

状态：已完成

完成日期：2026-07-05

### 目标

- 理解 teacher/student。
- 理解 CE + KL。
- 理解 temperature。

### 学习文件

```text
model_training_12_weeks/week_09_distillation.md
trainer/train_distillation.py
```

### 必须理解

- teacher model 为什么冻结。
- student model 学的是什么。
- CE loss 和 distillation KL loss 的区别。
- temperature 如何软化 teacher 分布。
- 蒸馏为什么适合小模型。

### Checklist

- [x] 读懂 `distillation_loss`。
- [x] 读懂 teacher logits 和 student logits 如何对齐。
- [x] 读懂 `alpha` 如何平衡 CE 和 KL。
- [x] 读懂 `temperature` 的作用。
- [ ] 设计一个上下文压缩小模型蒸馏任务。

核心任务：

- [x] 能解释 teacher/student。
- [x] 能解释 CE loss 和 KL distillation loss 的区别。

加餐任务：

- [ ] 设计一个“学习记录总结小模型”的蒸馏数据格式。

### 本周产出

```text
1. 蒸馏适合小模型的解释
2. “用大模型蒸馏学习总结小模型”的实验思路
3. `model_training_12_weeks/week_09_distillation.md`
```

### 阻塞点

```text

```

### 复盘记录

```text
已完成蒸馏主线：teacher/student、hard label/soft label、CE loss、KL distillation loss、temperature 软化分布、alpha 平衡 CE 与 KL。

上下文压缩和学习总结小模型的任务设计顺延到后续 Agent/教育场景实验。
```

## 14. 第 10 周：DPO

状态：已完成

完成日期：2026-07-06

### 目标

- 理解偏好数据。
- 理解 chosen/rejected。
- 理解 policy/reference。

### 学习文件

```text
trainer/train_dpo.py
dataset/lm_dataset.py
model_training_12_weeks/week_10_dpo.md
```

### 必须理解

- DPO 数据如何组织。
- chosen 和 rejected 如何拼成 batch。
- policy logprob 和 reference logprob 如何计算。
- DPO 为什么不需要显式 reward model。
- `beta` 如何控制偏好优化强度。

### Checklist

- [x] 新增 `model_training_12_weeks/week_10_dpo.md`。
- [x] 读懂 `DPODataset`。
- [x] 读懂 `logits_to_log_probs`。
- [x] 读懂 `dpo_loss`。
- [x] 初步解释 reference model 的作用。
- [x] 初步解释 beta 的作用。

核心任务：

- [x] 能解释 chosen/rejected 偏好对。
- [x] 能解释 policy model 和 reference model 的区别。

加餐任务：

- [ ] 用教育问答场景设计一组 chosen/rejected 样本。

### 本周产出

```text
1. DPO 数据流图
2. beta 和 reference model 的作用说明
3. `model_training_12_weeks/week_10_dpo.md`
```

### 阻塞点

```text
```

### 复盘记录

```text
已完成 DPO 主线：能够解释 chosen/rejected 偏好数据，理解 DPODataset 返回 x/y/mask 的原因，知道 logits_to_log_probs 是把 [B,T,V] 的整词表预测抽取成 [B,T] 的目标 token logprob。

已理解 dpo_loss 的核心：先把 token logprob 通过 mask 合成整句 logprob，再比较 policy 与 reference 对 chosen 相对 rejected 的偏好差异。

需要注意：beta 不是简单让 loss 变小，而是放大或缩小偏好差异信号。beta 越大，policy 对 chosen/rejected 差异的优化越激进；beta 越小，更新更温和。

```

## 15. 第 11 周：GRPO 与 RL 入门

状态：已完成

完成日期：2026-07-11

### 目标

- 理解 reward。
- 理解 rollout。
- 理解 advantage。
- 理解 KL penalty。

### 学习文件

```text
trainer/train_grpo.py
trainer/rollout_engine.py
model_training_12_weeks/week_11_ppo_prerequisites.md
```

### 必须理解

- 为什么 RL 训练需要先生成 response。
- 为什么同一个 prompt 要生成多个回答。
- 组内 reward 平均值和标准差如何构造 advantage。
- KL penalty 为什么能限制模型偏离 reference。
- reward 方差太小时为什么学习信号弱。

### Checklist

- [x] 读懂 `calculate_rewards` 的 reward 组成。
- [x] 读懂 rollout 调用的主流程。
- [x] 读懂 grouped rewards。
- [x] 读懂 advantages 计算。
- [x] 读懂 KL 项。
- [x] 新增 PPO / RLHF / GRPO 前置知识文档。
- [x] 对比 GRPO 和 PPO 的主要差异。

核心任务：

- [x] 能解释 rollout、reward、advantage、KL penalty。
- [x] 能解释为什么同一个 prompt 要生成多个回答。

加餐任务：

- [x] 阅读 PPO 部分，能够说明 Actor/Critic/Advantage、GAE、ratio、clip 与 KL 的完整角色分工。

### 本周产出

```text
1. 为什么同一个 prompt 要生成多个回答的解释
2. reward 方差为什么影响训练信号的说明
3. GRPO 与 DPO 的对比表
4. `model_training_12_weeks/week_11_grpo.md`
5. `model_training_12_weeks/week_11_ppo_prerequisites.md`
```

### 阻塞点

```text
```

### 复盘记录

```text
已进入 GRPO/RL 入门：完成 rollout、reward、grouped rewards、advantage 的第一轮理解。

已补齐完整 GRPO 数据流：从 RLAIFDataset 文本 prompt、tokenizer、rollout、model.generate、old_per_token_logps、reward、advantage、policy/ref logprob、KL、ratio、per_token_loss 到最终 scalar loss。

已完成 KL 项理解：ref_per_token_logps 和 per_token_logps 都来自完整 outputs 的 policy/ref forward，再通过 log_softmax + gather 得到 response token logprob；per_token_kl 逐 token 计算，shape 保持 [B * G, R]；beta 控制 KL penalty 强度。

已完成 GRPO / PPO 对比：GRPO 用同 prompt 多 response 的组内 reward 构造 advantage；PPO 用 Critic / Value Model 与 GAE 构造 token-level advantage，并额外训练 Critic。

已完成 PPO 的完整数据流和 `train_ppo.py` 变量映射：rollout -> reward / critic / GAE -> ratio -> reference KL -> policy loss + value loss -> 更新 actor / critic。
```

## 16. 第 12 周：Agentic RL 与教育 Agent 迁移

状态：进行中
完成日期：  

### 目标

- 理解 Tool Use 轨迹。
- 理解多轮 rollout。
- 理解延迟 reward。
- 设计一个教育 Agent 的最小训练任务。

### 学习文件

```text
trainer/train_agent.py
scripts/eval_toolcall.py
model_training_12_weeks/week_12_agentic_rl.md
```

### 必须理解

- 工具 schema 如何定义。
- 模型如何生成 `<tool_call>`。
- 工具结果如何返回到上下文。
- 多轮 rollout 如何形成完整轨迹。
- reward 为什么在整轮结束后计算。
- 这个机制如何迁移到教育 Agent。

### Checklist

- [x] 建立 Agentic RL 前置知识与教育 Agent 迁移文档。
- [ ] 读懂 `TOOLS` 定义。
- [ ] 读懂 `parse_tool_calls`。
- [ ] 读懂 `execute_tool`。
- [ ] 读懂 `rollout_single`。
- [ ] 读懂 tool call reward。
- [ ] 设计教育 Agent 工具列表。
- [ ] 设计学习任务 reward。
- [ ] 设计用户画像更新数据格式。

核心任务：

- [ ] 能解释 Tool Use 多轮轨迹。
- [ ] 能设计一个教育 Agent 的工具列表和 reward。

加餐任务：

- [ ] 阅读 `rollout_engine.py`，理解训推分离的接口思想。

### 本周产出

```text
1. 教育 Agent 工具列表
2. 学习任务 reward 设计
3. 用户画像更新数据格式
4. 一个最小教育 Agent 训练样本
```

### 阻塞点

```text

```

### 复盘记录

```text

```

## 16.5 MiniMind 之后的后续方向：小模型与多模态 Agent

MiniMind 主线完成后，学习方向不建议立刻变成“重新训练一个更大的教学模型”。

更适合当前教育 Agent 目标的方向是：

```text
大模型负责复杂讲解、推理和最终回复。
小模型负责 Agent 系统里的细节整理、状态抽取、路由、压缩和质检。
视觉小模型负责截图、页面、题目图片、学习材料的理解。
```

也就是说，小模型不一定直接承担“教学主模型”的角色，而更像教育 Agent 的后台 worker。

### 16.5.1 文本小模型适合承担的任务

文本小模型可以优先用于这些低成本、高频、结构化任务：

```text
用户画像抽取：conversation -> user_profile.json
学习状态摘要：study_logs -> learning_state.json
偏好更新：new_message + old_profile -> profile_patch.json
上下文压缩：long_context -> compact_memory
意图分类：message -> intent_label
任务路由：message -> tool_or_agent_route
输出质检：answer -> quality_report
```

这类任务的特点：

- 不一定需要最强通用推理。
- 需要稳定、便宜、快速。
- 输出通常可以设计成 JSON。
- 很适合自己构造数据、自己评估。

### 16.5.2 视觉小模型适合承担的任务

视觉小模型是教育 Agent 从“聊天助手”走向“操作型学习伙伴”的关键部件。

可以重点关注：

```text
截图 -> 页面元素结构
题目图片 -> 结构化题面
学习材料截图 -> 知识结构
手写/笔记图片 -> OCR + 错误定位
App/Web 页面 -> 下一步操作建议
```

典型产品能力类似：

```text
识别页面中按钮、输入框、菜单和错误提示。
判断用户当前卡在哪一步。
给出步骤引导，甚至配合工具层直接操作页面。
```

因此视觉小模型可以被定位为：

```text
Agent 的眼睛。
Agent 的 UI 侦察兵。
Agent 的视觉记忆整理器。
```

### 16.5.3 后续学习路线建议

MiniMind 主线不要中途改掉。更合理的顺序是：

```text
1. 先完成 MiniMind 的 LLM 训练闭环。
2. 学完 LoRA、蒸馏、DPO、GRPO、Agentic RL。
3. 用 MiniMind 建立“数据 -> 训练 -> 评估 -> 工程复盘”的能力。
4. 再切到多模态小模型和视觉小模型微调。
5. 最后把文本 worker、视觉 worker、工具层和记忆系统组合成教育 Agent。
```

后续多模态支线可以设计为：

```text
第 A 周：多模态模型基础，理解 image encoder、projector、LLM 的连接方式。
第 B 周：视觉问答数据格式，理解 image-text instruction tuning。
第 C 周：视觉小模型微调，做题目图片 OCR/结构化实验。
第 D 周：页面理解实验，截图 -> UI 元素 JSON。
第 E 周：教育 Agent 多模态原型，图片题目/页面引导/学习材料总结。
```

当前阶段的结论：

```text
MiniMind 主线继续。
未来方向新增“文本小模型 worker + 视觉小模型 worker + 教育 Agent 系统设计”。
多模态学习放在 MiniMind 后训练主线完成之后，会更稳。
```

## 17. 每周复盘模板

每周结束时复制下面模板，填到对应周的复盘记录中。

```text
本周主题：

状态：

完成日期：

我读了哪些文件：

我真正理解的 3 个知识点：

1.
2.
3.

我还不理解的 3 个问题：

1.
2.
3.

我能解释的一条数据流：

我能解释的一条 tensor shape：

我能跑通或修改的代码：

这一周和 AI 教育 Agent 方向的关系：

下周计划：
```

## 18. 里程碑检查

### 第 1 阶段里程碑：读懂输入链路

覆盖周次：

```text
第 1 周 - 第 2 周
```

完成标准：

- [ ] 能解释 MiniMind 项目目录。
- [ ] 能解释 tokenizer 的作用。
- [ ] 能解释 pretrain 和 SFT 数据样本差异。
- [ ] 能解释 `input_ids`、`labels`、`-100`。

### 第 2 阶段里程碑：读懂模型主干

覆盖周次：

```text
第 3 周 - 第 4 周
```

完成标准：

- [ ] 能写出 forward shape。
- [ ] 能解释 RMSNorm、RoPE、GQA、SwiGLU。
- [ ] 能解释 KV cache。
- [ ] 能读懂 `model_minimind.py` 的主流程。

### 第 3 阶段里程碑：读懂训练闭环

覆盖周次：

```text
第 5 周 - 第 7 周
```

完成标准：

- [ ] 能解释 pretrain 训练循环。
- [ ] 能解释 SFT 的 assistant-only loss。
- [ ] 能解释采样参数。
- [ ] 能把一个样本如何变成 loss 讲清楚。

### 第 4 阶段里程碑：读懂后训练

覆盖周次：

```text
第 8 周 - 第 11 周
```

完成标准：

- [ ] 能解释 LoRA。
- [ ] 能解释蒸馏。
- [ ] 能解释 DPO。
- [ ] 能解释 GRPO。
- [ ] 能说明这些方法分别改变模型的哪一部分行为。

### 第 5 阶段里程碑：迁移到教育 Agent

覆盖周次：

```text
第 12 周
```

完成标准：

- [ ] 能解释 Tool Use 多轮轨迹。
- [ ] 能解释延迟 reward。
- [ ] 能设计教育 Agent 工具。
- [ ] 能设计学习任务 reward。
- [ ] 能说清楚 MiniMind 哪些模块能迁移到 AI 教育 Agent。
