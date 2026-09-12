# MiniCPM5-2B：Base Training 阶段

> 记录日期：2026-09-13
> 研究范围：只梳理 Base Training，不提前展开 Mid-training、SFT、RL 或 OPD
> 证据原则：区分官方明确披露、根据训练原理作出的解释，以及仍未公开的工程细节

## 1. 这一阶段要解决什么问题

Base Training 是 MiniCPM5-2B 完整训练链路的第一阶段。它从随机初始化或尚未具备语言
能力的参数出发，通过 next-token prediction 建立模型最基础的能力：

- 学会词和句子的统计规律；
- 学会根据左侧上下文预测下一个 token；
- 在大量通用语料中形成基础知识和语言表示；
- 建立后续 Mid-training、SFT 和 RL 可以继续优化的模型参数；
- 在扩大训练规模和上下文长度时保持训练稳定。

这一阶段的目标不是让模型直接成为聊天助手。Base checkpoint 即使能够续写文本，也不
等于已经学会遵循 `system/user/assistant` 协议、稳定回答指令或按照指定格式调用工具。
这些行为主要由后续阶段建立和强化。

## 2. 官方披露的阶段链路

MiniCPM5-2B 官方材料将完整训练分为 Base training、Mid-training 和 Post-training。
其中官方训练流程图把 Base Training 进一步画成：

```text
随机初始化或训练起点
        |
        v
Stable Training
        |
        v
Short Decay（4K）
        |
        v
Long Decay（32K -> 128K -> 512K）
        |
        v
MiniCPM5-2B-Base checkpoint
```

官方正文明确说明 Base Training 经历 stable training 和 decay training，用来建立核心
语言能力与训练稳定性。`4K`、`32K -> 128K -> 512K` 来自官方训练流程图，表示该图
标注的序列长度阶段；但公开文字尚未给出每一段的精确训练 token 数、持续 step 数和
切换条件。

## 3. 三段分别在做什么

### 3.1 Stable Training

这是 Base Training 的主体阶段。模型持续接收经过清洗、分词和切分的预训练 token，
执行标准 causal language modeling：

```text
原始文本
-> 数据清洗、质量分层、去重和混合
-> tokenizer
-> 固定或动态长度的 token sequence
-> causal LM forward
-> 每个位置预测下一个 token
-> 全局有效 token 平均 loss
-> backward
-> optimizer / scheduler step
-> 周期性 checkpoint 与评测
```

“Stable”不能简单理解成“不改变学习率”。更可靠的含义是：这一段需要让大规模训练在
数据、数值、梯度、吞吐和故障恢复层面稳定推进。具体会涉及何种 optimizer、学习率
schedule、warmup、gradient clipping、并行策略和异常样本处理，官方当前公开材料没有
给出足以复现 MiniCPM5-2B 原始训练的完整配置。

### 3.2 Short Decay（4K）

这一段仍属于预训练，而不是 SFT。官方图将其标记为 4K 长度的 short decay。结合常见
预训练语义，可以把它理解为在短上下文训练末段逐步降低学习率，使参数从大步探索进入
更小步的收敛阶段。

这里需要严格区分：

- **官方事实**：存在名为 `Short Decay 4K` 的阶段标记；
- **合理解释**：`decay` 通常指学习率进入衰减阶段；
- **尚不能确认**：衰减曲线、起止学习率、token 数、数据是否同时提质或改变混合比例。

因此，不能只看到“decay”就断言该阶段的增益全部来自学习率。数据分布和采样策略也
可能同时发生变化，必须等待配置、日志或更详细技术材料才能归因。

### 3.3 Long Decay（32K -> 128K -> 512K）

官方图显示模型随后进入逐级扩展长度的 long decay：

```text
32K sequence
-> 128K sequence
-> 512K sequence
```

逐级扩展而不是直接从 4K 跳到 512K，工程上通常有三个理由：

1. 长序列显著增加 attention、activation 和通信成本，需要逐级验证显存与吞吐；
2. 模型需要逐步适应更远距离的位置关系和新的长度分布；
3. 每次长度切换都可以设置独立的稳定性探针、checkpoint 和长上下文评测。

这些是对该流程的工程解释，不是官方已经公布的完整实现。尤其不能仅凭 Base 的
`max_position_embeddings=524288` 就断言模型在所有 512K 任务上都具有可靠能力：配置
表示允许的长度上限，实际能力还取决于训练样本的长度分布、token 预算和评测结果。

## 4. 一条预训练样本的数据流

与我们正在做的 assistant-only SFT 相比，Base Training 的监督构造更直接。假设某段
文本分词后得到：

```text
input_ids = [10, 11, 12, 13, 14]
```

causal LM 训练在逻辑上形成：

```text
模型输入： [10, 11, 12, 13]
预测目标： [11, 12, 13, 14]
```

实际工程里通常把完整 `input_ids` 送入模型，由模型内部或 loss 函数执行 causal shift：

```text
logits[:, :-1, :]  对齐  labels[:, 1:]
```

数据与 shape 的主链路是：

```text
文本样本或拼接后的 token stream
-> tokenizer / packing
-> input_ids                       [B, T]
-> attention_mask                  [B, T]
-> LlamaForCausalLM
-> hidden states                   [B, T, H]
-> LM head
-> logits                          [B, T, V]
-> causal shift
-> shift_logits                    [B, T-1, V]
-> shift_labels                    [B, T-1]
-> token cross-entropy             [有效 token 数]
-> token 级全局平均 loss           scalar
-> backward / optimizer step
```

Base 预训练通常监督语料中的几乎所有非 padding token。SFT 则可能把 system、user、工具
返回和 padding 位置标成 `-100`，只监督 assistant completion。两者都使用 next-token
cross-entropy，但“哪些 token 进入 loss”不同。

## 5. Base 与当前 SFT 工程的关键区别

| 维度 | Base Training | 我们当前的数学 SFT |
| --- | --- | --- |
| 起点 | 随机初始化或早期预训练参数 | 已有 Qwen3-0.6B Base 权重 |
| 数据 | 大规模通用/代码/数学等预训练语料 | 经过审计的指令-回答样本 |
| 数据协议 | 主要是连续文本/token stream | chat template 与角色消息 |
| 监督位置 | 通常监督几乎所有有效 token | 只监督 assistant completion |
| 主要目标 | 建立语言、知识和基础预测能力 | 建立目标任务行为、格式和回答方式 |
| 训练规模 | 通常极大，决定基础模型能力上限 | 相对较小，改变目标分布上的行为 |
| 长度课程 | 官方图给出 4K 到 512K 的阶段变化 | 当前实验冻结为 16K 上限 |
| 产物 | Base checkpoint | S1 checkpoint 与配对评测证据 |

所以 MiniCPM5 的 Base Training 不是“更大规模的聊天 SFT”。它们可以使用相似的模型
forward、交叉熵和优化器，但训练数据合同、监督 mask、预算、长度课程和评测目标不同。

## 6. 从训练工程角度应有哪些模块

虽然官方没有公开一份可以原样复现 Base Training 的完整配置，但一个可信工程至少需要：

| 模块 | 需要回答的问题 |
| --- | --- |
| 数据治理 | 来源是什么，如何清洗、去重、质量分层和防止评测污染？ |
| 数据混合 | Web、代码、数学等类别按什么比例采样，比例是否随阶段变化？ |
| tokenizer | 词表如何训练，特殊 token 如何定义，跨语言覆盖如何验证？ |
| sequence construction | 文档如何切分或 packing，边界处是否允许跨文档 attention/loss？ |
| sampler | 如何确定全局样本顺序，分布式 worker 如何避免重复和遗漏？ |
| loss | causal shift、padding mask、跨 rank 有效 token 归一化是否正确？ |
| optimizer | AdamW 参数分组、状态精度、梯度裁剪和权重衰减如何设置？ |
| scheduler | warmup、stable 段和 decay 段如何衔接？ |
| parallelism | 数据、张量、流水线、序列并行分别如何组合？ |
| 长度课程 | 4K、32K、128K、512K 何时切换，batch/token budget 如何调整？ |
| checkpoint | 模型、optimizer、scheduler、sampler、RNG 和数据位置是否都可恢复？ |
| telemetry | loss、梯度范数、学习率、吞吐、MFU、显存和异常样本如何记录？ |
| evaluation | 通用能力、代码、数学、长上下文与训练稳定性如何分层验证？ |

这张表是下一轮讨论 MiniCPM5 第一个训练工程细节的入口，不代表官方已经公开了每个
问题的答案。

## 7. 当前能确认和不能确认的内容

### 已确认

- MiniCPM5-2B 是标准 `LlamaForCausalLM` 架构；
- 官方将训练分为 Base、Mid-training 和 Post-training；
- Base Training 包含 stable training 与 decay training；
- 官方图标记了 `Short Decay 4K` 和 `Long Decay 32K -> 128K -> 512K`；
- 官方发布了 `MiniCPM5-2B-Base` checkpoint；
- 官方列出 Ultra-FineWeb、Ultra-FineWeb-L3、UltraX、UltraData-Code 和
  UltraData-Math 等相关训练语料入口。

### 尚未确认

- Stable、Short Decay、Long Decay 各自的精确 token 数和 optimizer steps；
- 各阶段的数据 mixture、质量层级和采样权重；
- 完整 optimizer、scheduler、batch size 和并行配置；
- 4K、32K、128K、512K 切换时是否同时改变数据分布；
- 原始训练的硬件规模、实际吞吐、故障恢复策略和逐阶段评测结果；
- `MiniCPM5-2B-Base` 配置中的 524,288 上限与实际长序列训练覆盖之间的完整证据链。

## 8. 下一轮讨论顺序

下一轮不一次展开整套预训练系统，先讨论第一个最基础、也最容易被忽略的工程问题：

```text
原始预训练文本
-> 清洗与质量分层
-> tokenizer
-> token stream
-> 文档切分 / packing
-> 可被 DataLoader 和 sampler 消费的训练样本
```

需要重点弄清：为什么预训练数据不天然是一条 JSON 对应一个训练样本、packing 如何减少
padding、文档边界如何处理，以及长度从 4K 扩大到 32K 后数据管道要改变什么。之后再
进入分布式 sampler、训练 step、稳定性监控和 checkpoint/resume。

## 9. 官方来源

- [MiniCPM 官方仓库与 MiniCPM5-2B Training Recipe](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM5-2B 官方模型卡](https://huggingface.co/openbmb/MiniCPM5-2B)
- [MiniCPM5-2B-Base checkpoint](https://huggingface.co/openbmb/MiniCPM5-2B-Base)
- [MiniCPM5 官方模型集合](https://huggingface.co/collections/openbmb/minicpm5)
- [UltraData 数据入口](https://ultradata.openbmb.cn/)
