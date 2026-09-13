# MiniCPM5-2B：Base Training 阶段

> 记录日期：2026-09-13
>
> 研究范围：只讨论 Base Training，不提前展开 Mid-training、SFT、RL 或 OPD
>
> 证据原则：分别标明官方事实、通用机制和仍未公开的实现，不用合理推断替代复现证据

## 1. 当前阶段在完整链路中的位置

MiniCPM5-2B 的公开训练链路是：

```text
Base Training
-> Mid-training
-> SFT
-> RL 专家模型
-> OPD 合并专家能力
-> MiniCPM5-2B 最终模型
```

Base Training 位于最前面。它解决的不是“如何让模型回答用户”，而是“如何让一组初始
参数成为具备基础语言、知识和预测能力的基础模型”。后续阶段可以调整能力分布、交互
行为和偏好，但都建立在 Base checkpoint 已经形成的表示和预测能力上。

本阶段可以先用一句话概括：

> 大规模文本经过治理、分词、切分和采样后，被组织成 causal language modeling 样本；
> 模型通过持续预测下一个 token 更新全部参数，并在训练后段逐步降低学习率、扩大上下文
> 长度，最终产出可供 Mid-training 使用的 MiniCPM5-2B-Base checkpoint。

## 2. 输入和输出

### 2.1 输入 checkpoint

Base Training 的概念起点通常是随机初始化的模型参数。对于 MiniCPM5-2B，公开材料没有
给出初始化 checkpoint、初始化脚本和随机种子，因此目前只能确认其 Base 阶段是形成
基础模型的阶段，不能把“从完全随机初始化开始”写成已经核验的官方运行事实。

模型结构本身是公开的。`MiniCPM5-2B-Base/config.json` 给出的关键配置包括：

```text
architecture:             LlamaForCausalLM
vocab_size:               130560
hidden_size:              2048
intermediate_size:        6144
num_hidden_layers:        42
num_attention_heads:      16
num_key_value_heads:      2
head_dim:                 128
tie_word_embeddings:      false
max_position_embeddings:  524288
```

这套配置定义了要训练的参数容器，但没有定义训练数据、采样顺序、优化器状态或训练进度。

### 2.2 输入数据

官方将 Ultra-FineWeb、Ultra-FineWeb-L3、UltraX、UltraData-Code 和 UltraData-Math 列为
Base/Mid-training 相关公开语料。它们覆盖通用网页、代码和数学等分布。

这里应把“官方公开的相关语料仓库”和“官方某次训练实际消费的数据流”区分开：公开仓库
不自动给出精确快照、混合比例、采样权重、去重版本和每个子阶段消耗的 token 数。因此，
我们能够研究数据形态和治理方法，但尚不能据此还原原始训练 batch 序列。

### 2.3 输出 checkpoint

本阶段的公开产物是 `openbmb/MiniCPM5-2B-Base`。它包含模型权重、模型配置、Tokenizer
和生成配置，可作为后续 Mid-training 的参数起点。

```text
Base Training 的输出
= 模型参数状态
+ 模型结构配置
+ Tokenizer 与特殊 token 约定
+ 推理所需生成配置
```

公开模型仓库不包含原训练 optimizer、scheduler、sampler、RNG 和数据游标状态，所以它
适合推理、评测和启动新的训练实验，不等于能够从官方最后一步无损恢复原 Base 训练。

## 3. 这个阶段解决什么能力问题

模型在这一阶段通过 next-token prediction 建立：

- 词、句子和篇章的统计规律；
- 根据左侧上下文预测后续 token 的能力；
- 从通用、代码和数学语料中形成的基础知识与表示；
- 供后续阶段继续塑造的模型参数；
- 在大规模训练和长度扩展过程中的数值与工程稳定性。

Base checkpoint 能够续写文本，不代表已经成为聊天助手。稳定遵循
`system/user/assistant` 协议、按照指令组织答案、调用工具和满足偏好，主要由后续的
Mid-training、SFT 和 RL 阶段建立或强化。

## 4. 从原始文本到一个训练 batch

Base Training 的第一条主线发生在模型 forward 之前：原始数据必须先变成可重复交付的
token batch。完整链路应理解为：

```text
网页、代码、数学文档
-> 解析与规范化
-> 内容质量过滤
-> 精确去重 / 近似去重
-> 评测污染检查
-> 质量层级和领域标签
-> 数据混合与采样策略
-> tokenizer
-> 每篇文档的 token ids
-> 文档边界标记，例如 EOS
-> token shard / index / cache
-> distributed sampler 确定样本顺序
-> sequence construction / packing
-> DataLoader 组装 batch
-> input_ids [B, T]
```

各模块承担不同职责：

| 模块 | 它决定什么 |
| --- | --- |
| 数据治理 | 哪些文本有资格进入候选训练集 |
| 质量与领域分层 | 样本属于什么分布，质量处于哪个层级 |
| mixture | 每个 optimizer step 看到各类数据的概率 |
| tokenizer | 文本如何离散为词表 ID |
| shard/index | 数据如何存储、定位和并行读取 |
| sampler | 各 rank、epoch 和恢复点消费哪些样本 |
| packing | 多篇短文档如何填满固定长度序列 |
| DataLoader/collator | 如何把样本组织成实际 batch tensor |

预训练数据不天然是“一行 JSON 等于一个训练样本”。一篇文档可能长于 `T`，需要切成
多段；许多短文档也可能被拼进同一条序列以减少 padding。是否允许不同文档之间相互
attention、文档边界是否进入 loss，以及 packing 后如何恢复数据位置，都会改变实际训练
合同。

MiniCPM5 官方公开了分层数据治理思路，但尚未公开足以还原原始 batch 的全部实现细节。

## 5. 一条 batch 在模型内部如何流动

假设 DataLoader 已交付：

```text
input_ids      [B, T]
attention_mask [B, T]     # 具体预训练实现也可能采用隐式 mask 或 packed metadata
```

模型内部的数据流是：

```text
input_ids [B, T]
-> token embedding
-> hidden states [B, T, H=2048]
-> 42 个 Llama decoder blocks
-> final RMSNorm
-> final hidden states [B, T, H]
-> LM head
-> logits [B, T, V=130560]
```

每个 decoder block 内部又包含两条主要计算路径：

```text
hidden states
-> RMSNorm
-> GQA attention
-> residual add
-> RMSNorm
-> gated MLP
-> residual add
-> 下一层 hidden states
```

Causal mask 保证位置 `t` 只能读取自己和左侧位置的信息，不能提前看到右侧答案。GQA 中
Q 使用 16 个 head，而 K/V 只使用 2 个 head，从而减少 K/V 投影、通信和 KV cache 成本。

## 6. 训练目标和 loss 是什么

Base Training 使用 causal language modeling。假设一条序列是：

```text
input_ids = [10, 11, 12, 13, 14]
```

逻辑监督关系是：

```text
看到 [10]             预测 11
看到 [10, 11]         预测 12
看到 [10, 11, 12]     预测 13
看到 [10, 11, 12, 13] 预测 14
```

工程上通常把完整序列同时作为 `input_ids` 和 labels 的来源，再进行 causal shift：

```text
logits[:, :-1, :]  [B, T-1, V]
labels[:, 1:]      [B, T-1]
```

接着对每个有效目标 token 计算交叉熵：

```text
token NLL
= -log P(target_token | left_context)
```

主链路为：

```text
logits [B, T, V]
-> causal shift
-> shift_logits [B, T-1, V]
-> shift_labels [B, T-1]
-> token cross-entropy
-> 对全局有效 token 求和
-> 除以全局有效 token 数
-> scalar loss
```

Base 预训练通常监督语料中的几乎所有非 padding token。Assistant-only SFT 则会把
system、user、工具返回和 padding 等位置标为 `-100`，只监督 assistant completion。
两者都使用 next-token cross-entropy，但进入 loss 的 token 合同不同。

官方尚未披露 MiniCPM5-2B Base 原始运行在跨 rank、梯度累积和 packed sequence 下的
精确 loss 归一化实现，因此这里说明的是需要核验的正确机制，不宣称是官方源码复述。

## 7. 哪些参数会更新，梯度如何到达它们

Base Training 通常是 full-parameter training，而不是 LoRA。loss.backward() 会沿前向
计算图反向传播到：

```text
LM head
<- final RMSNorm
<- 每层 MLP
<- 每层 attention 的 Q/K/V/O projection
<- 每层 RMSNorm
<- token embedding
```

因此，输入 embedding、42 层 Transformer 主体和独立 LM head 都是需要优化的模型参数。
`tie_word_embeddings=false` 表示输入 embedding 与 LM head 是两套参数，不共享同一块
权重。

一个完整 optimizer update 通常经历：

```text
若干 micro-batch forward
-> 每个 micro-batch 得到 loss numerator 和有效 token 数
-> backward 累积梯度
-> 跨数据并行 rank 汇总
-> 按全局有效 token 数归一化
-> 可选 gradient clipping
-> optimizer.step()
-> scheduler.step()
-> zero_grad()
```

MiniCPM5 官方尚未公开原始 Base 训练所用 optimizer、参数分组、状态精度、梯度裁剪阈值
和梯度累积配置，因此不能进一步写成确定参数。

## 8. Base Training 内部为什么还有多个阶段

官方流程图把 Base Training 标为：

```text
训练起点
-> Stable Training
-> Short Decay（4K）
-> Long Decay（32K -> 128K -> 512K）
-> MiniCPM5-2B-Base
```

### 8.1 Stable Training

这是 Base Training 的主体阶段。模型在大规模 token 上持续进行 causal language
modeling。这里的“Stable”不应被理解为“学习率不变化”，而应理解为整个训练需要在
loss、梯度、数据交付、吞吐、显存、通信和故障恢复方面保持可持续推进。

这一阶段至少要持续观察：

```text
训练 loss 与分领域 loss
梯度范数和异常梯度
学习率
有效 token / second
数据读取与等待时间
GPU 利用率、显存和 MFU
NaN / Inf / OOM
数据跳过、worker 或通信错误
checkpoint 保存和恢复结果
```

### 8.2 Short Decay（4K）

这一段仍然属于预训练。官方图确认存在 `Short Decay 4K`，但没有披露衰减曲线、起止
学习率、token 预算和数据 mixture。

`decay` 在训练语境中通常意味着学习率从较大的更新逐步下降，使参数进入更小步的收敛
阶段。但若数据质量层级和 mixture 同时变化，能力增益不能只归因于学习率。当前证据只能
支持“存在该阶段”，不能支持完整因果解释。

### 8.3 Long Decay（32K -> 128K -> 512K）

官方图随后给出逐级长度扩展：

```text
32K
-> 128K
-> 512K
```

逐级扩展在工程上可以降低一次跳到极长序列的风险：

- attention、activation 和通信成本会显著增加；
- batch size、梯度累积和 token budget 需要重新平衡；
- 数据管道必须交付足够多的真实长文档或合理拼接序列；
- 每次切换都需要显存、吞吐、loss 和长上下文能力探针；
- 训练并行可能需要加入 context/sequence parallel 等策略。

这些是由训练机制得到的工程要求，不是 MiniCPM5 已公开的具体实现。配置中的
`max_position_embeddings=524288` 只说明模型配置允许这一长度上限，不能单独证明每个
长度区间获得了多少训练 token，也不能证明所有 512K 任务上的可靠能力。

## 9. 训练基础设施需要承担什么

可信的 Base 训练基础设施至少要覆盖：

| 子系统 | 必须保证的合同 |
| --- | --- |
| 数据存储 | 大规模 shard 可定位、可校验、可并行读取 |
| sampler | 各 rank 不重复、不遗漏，并能从数据位置恢复 |
| DataLoader | CPU 处理和磁盘读取不长期饿死 GPU |
| 分布式训练 | 参数、梯度和 optimizer 状态按既定策略切分与同步 |
| 长序列训练 | attention、activation 和通信不突破显存与时延预算 |
| checkpoint | 保存模型、optimizer、scheduler、RNG、sampler 和数据游标 |
| telemetry | 将 loss、梯度、学习率、吞吐、显存和异常持久化 |
| evaluator | 在冻结协议下比较训练前后及阶段间能力 |

公开的 Meshy 是面向 RL 的服务化训练框架，不能据此断言 MiniCPM5 Base Training 使用了
Meshy。当前官方也没有公开 MiniCPM5-2B Base 的完整分布式拓扑、硬件规模和启动配置。

## 10. 用什么评测证明阶段目标达成

如果要证明 Base Training 成功，不能只展示 train loss 下降。至少需要四层证据：

```text
训练健康
-> held-out validation NLL / perplexity
-> 通用、代码、数学等能力评测
-> 长上下文能力与生成健康检查
```

对应问题分别是：

| 证据层 | 回答的问题 |
| --- | --- |
| 训练健康 | 是否稳定完成，有无 NaN、OOM、梯度异常和数据中断 |
| Validation NLL | 是否更会预测未参与训练的同分布 token |
| 下游 benchmark | 基础能力是否真实形成，而不只是记忆训练分布 |
| 长上下文评测 | 32K/128K/512K 配置是否转化为可测能力 |
| 生成健康 | 是否能正常终止，有无重复、截断和退化模式 |

最理想的公开证据是对初始化、Stable、Short Decay、各 Long Decay checkpoint 和最终 Base
做同协议评测。官方目前主要公开阶段终点 checkpoint 和最终模型评测，没有给出足以完成
上述逐阶段归因的完整结果，所以我们不能判断每一段分别贡献了多少能力。

## 11. 算力和工程成本目前能知道什么

模型规模可以精确核验：总参数量约 25.17 亿，BF16 权重文件约 5.03 GB。但训练成本不能
从权重大小直接推出。Base Training 还需要梯度、optimizer 状态、activation、通信 buffer、
数据管道和 checkpoint 空间；长序列尤其会放大 activation 与 attention 成本。

估算 dense Transformer 训练计算量通常还需要至少知道：

```text
模型参数量 N
训练 token 数 D
序列长度分布
实际硬件与并行拓扑
吞吐、MFU 和失败重跑时间
```

MiniCPM5-2B Base 的精确 token 预算、GPU 型号与数量、运行时长、吞吐、MFU 和费用没有在
当前官方材料中完整披露。因此本阶段只能确认“这是实验室规模的全参数预训练工程”，不能
给出可信的总 GPU-hours 或资金成本。

## 12. 当前 SFT 工程与这一阶段是什么关系

| 维度 | MiniCPM5 Base Training | 我们当前的数学 SFT |
| --- | --- | --- |
| 参数起点 | 未公开的初始化训练状态 | 已有 Qwen3-0.6B Base 权重 |
| 数据 | 大规模连续预训练语料 | 审计后的指令与回答样本 |
| 数据协议 | 文档、token stream、packing | chat template 与角色消息 |
| 监督位置 | 通常几乎所有有效 token | assistant completion token |
| 更新范围 | 通常全部模型参数 | 当前实验定义的全部可训练参数 |
| 主要目标 | 形成语言、知识和基础预测能力 | 改变数学回答行为和目标分布拟合 |
| 规模 | 大规模预训练 | 16K 样本的受控 SFT |
| 阶段产物 | Base checkpoint | S1 checkpoint 与 B0/S1 证据 |

两者共享 tokenizer、causal forward、cross-entropy、backward、optimizer、scheduler、
checkpoint 和 evaluation 等基础模块，但数据合同、loss mask、训练预算和成功标准不同。
因此 Base Training 不是“更大规模的聊天 SFT”。

## 13. 当前结论与证据边界

### 已确认

- 模型采用标准 `LlamaForCausalLM`，关键结构配置已公开；
- 官方训练链路包含 Base、Mid-training 和 Post-training；
- Base Training 包含 Stable Training 与 Decay Training；
- 官方图标记 `Short Decay 4K` 和 `Long Decay 32K -> 128K -> 512K`；
- `MiniCPM5-2B-Base` checkpoint 已公开；
- 多个与 Base/Mid-training 相关的数据集仓库已公开。

### 尚未确认

- 初始化 checkpoint、随机种子和初始化实现；
- Stable、Short Decay、Long Decay 的精确 token 数和 optimizer steps；
- 各阶段的数据快照、mixture、质量层级和采样权重；
- optimizer、scheduler、batch、loss 归一化和梯度裁剪配置；
- 数据 packing、文档边界、sampler 和恢复语义；
- 分布式并行拓扑、硬件规模、吞吐、MFU、时长和费用；
- 长度切换时是否同步改变数据分布和位置编码策略；
- 各子阶段 checkpoint 的同协议能力增量。

所以本阶段的结论仍是：公开材料足以解释 Base Training 的角色、主链路和关键工程问题，
也足以围绕公开 checkpoint 与数据做缩小实验，但不足以原样复现 MiniCPM5-2B Base 的
官方训练运行。

## 14. 下一轮从哪里继续

下一轮先沿本阶段链路的前半段深入：

```text
原始文档
-> 清洗与质量分层
-> tokenizer
-> token shard
-> 文档切分 / packing
-> sampler
-> DataLoader
-> input_ids [B, T]
```

重点讨论四个问题：预训练数据为什么不是一行 JSON 对应一个样本；packing 为什么能减少
padding；文档边界怎样影响 attention 和 loss；长度从 4K 增加到 32K 后数据交付、batch
大小和训练吞吐为什么必须一起改变。理解这一段后，再进入分布式训练 step、稳定性监控和
checkpoint/resume。

## 15. 官方来源

- [MiniCPM 官方仓库与 MiniCPM5-2B Training Recipe](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM5-2B 官方模型卡](https://huggingface.co/openbmb/MiniCPM5-2B)
- [MiniCPM5-2B-Base checkpoint](https://huggingface.co/openbmb/MiniCPM5-2B-Base)
- [MiniCPM5-2B-Base config.json](https://huggingface.co/openbmb/MiniCPM5-2B-Base/blob/main/config.json)
- [MiniCPM5 官方模型集合](https://huggingface.co/collections/openbmb/minicpm5)
- [UltraData 数据入口](https://ultradata.openbmb.cn/)
