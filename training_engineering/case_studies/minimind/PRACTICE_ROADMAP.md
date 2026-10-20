# MiniMind 从零实践路线

> 更新：2026-10-18

| 当前状态 | 当前任务 | 下一里程碑 |
| --- | --- | --- |
| 第 6 周教学工程验收完成：全项目 249 项测试通过 | 下一次从真实数据与实验合同准备开始，尚未启动正式训练 | 第 7 周：冻结真实 Pretrain 子集与短训合同 |

## 目标

本轮要亲手实现并运行一套教学规模的语言模型训练工程：

```text
数据 -> tokenizer -> Transformer -> Pretrain -> SFT
-> LoRA / 蒸馏 / DPO -> 在线 RL -> Tool Use / Agent RL
```

完成后，需要具备三项能力：

1. **算法**：能实现并验证训练阶段的核心计算。
2. **工程**：能独立搭建、启动、恢复、评测和排查训练任务。
3. **实验**：能设计受控对照，并说明结果支持和不支持什么结论。

第一轮理论笔记只作参考。本轮是否完成，以亲手编写的代码、实际运行记录和实验结果为准。

## 路线

18 周表示预计工作量，不要求严格对应 18 个日历周。每个阶段通过验收后再进入下一阶段。

| 阶段 | 周次 | 主要问题 | 阶段产物 |
| --- | --- | --- | --- |
| 数据入口 | 1–2 | 模型实际读到了什么？ | 审计 CLI、BPE、Dataset、collator |
| 模型系统 | 3–6 | loss 如何变成参数更新？ | Transformer、optimizer、训练与恢复系统 |
| 完整训练 | 7–9 | 训练是否真的改变了目标能力？ | Pretrain、SFT、生成与配对评测 |
| 离线后训练 | 10–12 | 不同监督信号怎样改变模型？ | LoRA、蒸馏、DPO |
| 在线强化学习 | 13–15 | 在线轨迹怎样进入策略更新？ | rollout、reward、GRPO/CISPO、PPO |
| Agent 与交付 | 16–18 | 模型怎样在工具环境中学习？ | Tool Use、Agent RL、独立复现报告 |

## 已完成：第 2 周

### 目标

把通过审计的文本转换为 token、labels 和 batch，解释每一步转换的依据。
首周实现与证据见 [第 1 周实践记录](practice_notes/week_01_data_audit.md)。
本周带练步骤与已完成结果见 [第 2 周实践教程](practice_notes/week_02_tokenizer_dataset.md)。

### 任务

- [x] 实现教学 BPE 的 pair 统计、合并、编码、解码及保存重载。
- [x] 加载官方 tokenizer，观察真实文本的 token、ID 和特殊 token。
- [x] 对照官方 PretrainDataset，实现文本截断、BOS/EOS、定长 padding 和 labels。
- [x] 将组批逻辑拆到 collator，验证与官方定长行为一致。
- [x] 追踪真实审计样本到 batch，检查 shape、dtype、padding 与监督位置。

### 产物

- 可保存重载的教学 tokenizer；
- 官方 tokenizer 适配与 token/ID 对照；
- Pretrain Dataset、collator 与测试；
- 一条真实样本的 text → tokens → labels → batch 记录。

### 验收

- 教学 tokenizer 保存重载前后编码一致，完整编码可还原原文。
- 使用相同官方 tokenizer、输入及长度时，input_ids 和 labels 与官方实现对齐。
- padding 位置不参与 loss；能说明 causal shift 由后续哪一层负责，避免重复 shift。

第二周验收结果：Pretrain 数据相关测试 10 passed，全项目 77 passed；20 条审计通过样本均可进入
Dataset，首个真实 batch 的 `input_ids`、`attention_mask` 和 `labels` 均为 `[2, 512]`。

### 说明

- 继续使用少量真实样本，在本机完成组件验证。
- SFT 模板和 assistant mask 的完整实现安排在第 8 周。
- 首周独立增加校验规则及测试的迁移验收仍待完成，不能用助手复核替代。

### 官方对应

参照冻结提交 `f659b55761b754d306bd140573493a6543cafd7f`，学习顺序按实现依赖拆分。

| 内容 | MiniMind 官方实现 | 本轮安排 |
| --- | --- | --- |
| BPE | `trainer/train_tokenizer.py` 使用 tokenizers 的 ByteLevel/BPE/BpeTrainer | 手写小型教学实现理解算法；不要求复刻官方词表 |
| 正式 tokenizer | `model/tokenizer.json` 和 `model/tokenizer_config.json` | 使用官方文件，保持 token ID 与模型权重兼容 |
| Pretrain Dataset | `dataset/lm_dataset.py::PretrainDataset` | 对齐截断到 max_length-2、添加 BOS/EOS、padding、labels 屏蔽的行为 |
| 组批 | 官方在 Dataset 内定长 padding，返回 input_ids 与 labels | 独立 collator 是教学工程拆分；先定长对齐，动态 padding 后续再比较 |

官方脚本也明确建议复用现成 tokenizer。教学 BPE 的产物单独保存，不覆盖官方文件。

## 已完成：第 3 周

进入模型系统，先实现不含 attention 的基础组件：配置、Embedding、RMSNorm、SwiGLU/MLP、残差连接、
参数初始化与参数量统计。每个组件先完成 shape、dtype、梯度和独立参考测试，再装配模块 smoke；
causal attention、RoPE 和完整 Decoder 留到第 4 周。

带练操作、测试与排错同步记录于[第 3 周实践教程](practice_notes/week_03_model_components.md)。
截至 2026-10-03，基础组件、前馈残差、配置、教学初始化与参数统计已完成，
组合的前向和反向已整理为正式 smoke 测试。全项目 `101 passed`，
其中本周新增24项。当前初始化规则尚未宣称与官方完整模型一致；
完整 attention、位置机制、语言模型输出头与生成留到第四周。

2026-10-07 [第 4 周实践](practice_notes/week_04_attention_and_decoder.md)已完成基础注意力函数、可学习单头与普通多头模块，
Attention 测试文件为 23 passed，GQA 的分组观察与 KV 扩展数值、梯度测试已完成。
2026-10-08 GQA 初始化与前向已完成，CPU/CUDA 两种模式的逐头参考、梯度及等头数 MHA 对照临时检查通过；
当时的下一步是将 GQA 检查保存为正式测试。2026-10-11，GQA 接入 RoPE 和 padding mask 后，当时记录完整项目为 168 passed；随后完成 Decoder、主干、输出头、loss、参数更新与生成。

### 学习衔接

2026-09-30 阅读学习者提供的《自回归语言模型的起源和发展路径》对话记录后，明确后续讲解目标：
以 MiniMind 为实践载体，串起神经网络的表示、计算、训练与架构选择，并通过小型迁移实验验证理解。
对话中其他 AI 的解释不作为学习者已经掌握的证据；以下是依据学习者提问确定的补课重点。

| 观察 | 后续重点 |
| --- | --- |
| 能描述输入、模型、输出、loss、梯度和优化器的基本链条 | 用一个可手算的小网络追踪参数、激活、loss、梯度和更新，说明各自属于什么 |
| 仍在确认训练任务、阶段、架构和 loss 是否一一对应 | 区分任务需求、数学 objective、target、网络结构和训练阶段；用相同骨干配不同输出头比较 |
| 开始把文字、语音和分类联系到共同训练过程 | 从 LLM 学到的通用原理出发，小范围比较分类、回归和序列预测的输出及监督差异 |
| 希望理解架构为什么如此设计 | 每个组件说明解决的问题、信息流、参数共享、计算成本和可以验证的限制 |

后续每个模型组件按同一顺序学习：

1. 先回到整条模型计算链，指出当前组件的输入、输出和作用。
2. 用小数值例子讲清计算，区分参数与中间激活。
3. 对应公式、张量 shape 和代码，并检查梯度经过哪些路径。
4. 由学习者实现，完成独立参考或行为测试。
5. 用一个小改动解释该组件与一般神经网络原理的关系，再返回 MiniMind 主线。

第 3 周开始前，先做最小网络的计算回顾；随后解释 token ID、可训练 Embedding、线性层和非线性。
进入第 4 周时，重点区分逐位置 MLP 与跨位置 attention，以及 causal mask、padding mask、loss mask。
训练系统阶段再连接链式法则、局部梯度与 optimizer 状态。

第三周的通用原理统一见[模型基础系列](../../../model_architecture_and_algorithms/01_model_architecture/README.md)。
它从预测任务讲到组件职责、整体结构和自动求导；逐周教程记录实现过程与实际验收。
当前先理解已有结构，不增加组件改造或消融任务。两类文档互相链接，不改变本路线的实践顺序。

迁移验证沿用已有小模型：比较语言模型头与分类头；比较保留上下文交互与仅逐位置计算；
比较 Pretrain 与 SFT 改变监督位置后、同一骨干的训练信号。先提出预期，再运行并解释结果。
这些练习嵌入现有周次，不另开一条完整训练路线；需要新增输入编码、对齐或生成机制的模态再按需深入。

## 已完成：第 4 周

实现步骤、排错与验收见[第 4 周实践教程](practice_notes/week_04_attention_and_decoder.md)。
Attention、RoPE、GQA、mask、完整 Decoder、共享词表输出、causal loss、初始化和 greedy 生成均已接通。
真实 JSONL batch 的单步更新与更新前后 loss 对照通过；生成函数 16 项测试通过，已有 mask 边界覆盖已补齐。

最后用 CPU tiny 模型重复学习人工序列 `[1,3,4,2]`：100 次 AdamW 更新后，loss 从约 2.99900
降至 0.00388；仅给前缀 `[1]`，生成 `[1,3,4,2]` 并在 EOS=2 提前停止。
2026-10-11 最后一次全项目回归为 221 passed。本项证明极小目标可学习，不证明自然语言能力或泛化。
第四周到此验收完成，不再增加生成策略。

## 已完成：第 5 周

实践过程见[第 5 周：单设备训练系统](practice_notes/week_05_training_system.md)。
2026-10-12，学习者编写 SGD/AdamW 单步对照并修复参数集合传入问题；助手复核运行通过。
相同初值、目标与学习率下，SGD 更新约 -0.4，AdamW 更新约 -0.1。已回顾梯度定义、参数梯度与反向传播。
随后加入梯度与状态打印，复核两条路径梯度均为 4.0；AdamW 的 step=1、m≈0.4、v≈0.016。
已解释参数对象作字典键的读取方式。2026-10-13，继续讲解 m/v 递推、零初始化偏差修正与首次更新。
学习者写入首次更新手算后，复核运行发现 update 遗漏 lr，预测约 1.0 与实际约 1.9 不符；
随后由学习者补上学习率，助手重新运行，预测参数约 1.900000001，与实际约 1.899999976 对齐。
随后学习者增加 `weight_decay=0.2` 的独立分支：梯度与 `m/v` 和无衰减分支一致，额外衰减为 0.04，
手算参数约 1.860000001 与实际约 1.860000014 对齐，完成了解耦权重衰减的标量单步验证。
通用梯度与优化器解释已归入算法目录；至此完成标量机制验证，随后进入真实模型参数分组。

2026-10-14，学习者完成真实 MiniMind 参数分组：15 个二维权重进入 decay 组，5 个 RMSNorm 权重进入 no-decay 组；对象身份检查确认组内无重复、组间无交叉且全部参数均被覆盖。随后以全零但非 None 的梯度隔离自适应更新，实际观察到 decay 组按 `before*(1-lr*weight_decay)` 变化、no-decay 组保持不变，逐参数数值对照全部通过。下一步进入学习率调度，完整训练循环尚未完成。

随后学习者使用 SGD 与 StepLR 完成四次更新，区分参数更新实际使用的 `lr_used` 和调度后供下一次更新使用的 `next_lr`；学习率序列和参数数值均与独立预期对齐。下一步实现 warmup 与线性衰减，完整训练循环尚未完成。

同日，学习者完成 warmup 与线性衰减倍率函数及 LambdaLR 接入，修复循环缺少 backward 和旧断言接口的问题。助手复核六次真实参数更新：两条学习率序列断言通过，参数由 2.0 更新至约 0.94012。学习率调度这一小节完成；下一步进入梯度累积与 token 平均。

随后学习者完成标量梯度累积与一次性平均 loss 的参考对照。助手复核两条路径的平均梯度均为 3、更新后参数均约为 1.7，两个断言通过。下一步验证不同有效 token 数 micro-batch 的全局 token 平均；裁剪及循环装配仍待完成。

最近验证（2026-10-15）：学习者新增 loss sum/count 接口，现有 loss 测试 12 项通过；语言模型两条序列有效 token 数为 2 和 6，分别对 loss sum backward、累计梯度除以 8 后，全部参数梯度与大 batch 参考路径对齐。错误的 micro-batch 等权平均 loss 为约 2.47768，而参考 loss 约为 2.90018，最大梯度差约为 0.19432，完成错误归一化观察。下一步进入梯度裁剪；tail 窗口和循环装配仍待完成。

2026-10-16，学习者实现 train_update，完成 token 平均、统一裁剪、AdamW 更新和 scheduler 推进的装配，并修复空列表、冻结参数与日志数值类型问题；助手清理日志键的冒号拼写。完整更新与大 batch 参考参数对齐，空输入拒绝和冻结权重不变的定向检查通过。下一步完成外层循环、最后不足一个窗口的处理和本周整体验收。

同日，学习者完成 `train_epoch` 与整批更新对照测试，修复参考路径未裁剪和浮点严格比较的问题。用户委托助手补齐尾部更新、空输入、冻结参数和非法累积步数测试后，训练测试 8 项通过，全项目 229 项通过（7.74 秒）。第五周单设备训练循环验收完成；具体调用链、测试理由和协助范围见实践记录第 10–13 节。下一步进入第六周的验证、训练状态保存与恢复，当前尚未启动真实语料预训练。

## 已完成：第 6 周

实践过程见[第 6 周：验证与训练恢复](practice_notes/week_06_validation_and_checkpoint.md)。学习者完成 `evaluate_nll` 与整批参考对照，助手确认 token 加权、模式恢复、参数和已有梯度保留的行为；其中正式对照已落入 pytest，其他边界为临时检查。2026-10-16，训练与验证定向测试 9 项通过（2.81 秒）。随后学习者实现基础 checkpoint 保存与加载，助手临时核验写入失败保留旧文件，以及相同后续输入下恢复后的下一次更新与连续路径完全一致。下一步补随机状态、数据位置和完整恢复对照。

2026-10-17，学习者完成 CPU 与 CUDA RNG 接入，按可用设备数至少覆盖已保存状态的约定检查环境，并将检查移到修改对象之前。助手在 RTX 4060 Ti 上运行已落盘的核验脚本，确认重新建模和加载后 CPU/CUDA 下一组随机数逐元素一致，其他训练状态也恢复。下一步开始单 epoch 的 sampler 顺序、提交位置及数据恢复；完整训练对照尚未完成。

2026-10-18，学习者完成 StatefulSampler、按实际样本数提交进度、checkpoint 数据状态接入，以及连续训练与中断恢复的联合对照。模型参数、AdamW m/v/step、参数组、调度器、sampler 和更新次数比较通过。随后完成 CPU 单 epoch 最小入口，支持新训练、已知兼容 checkpoint 恢复、结果 JSON、普通异常记录及非空输出目录保护。助手获授权整理回归与文档，新增 19 项定向测试通过；最终全项目结果见第六周实践记录第 9 节。

本周验收范围为确定性数据、num_workers=0、更新边界恢复。当前 CLI 只在 epoch 结束保存，没有周期保存，也没有数据/tokenizer/配置 hash 恢复门禁；第七周真实训练前应冻结身份并接入所需更新边界保存。这里不宣称生产级容错或正式预训练已经完成，不新增多卡、多 worker 和混合精度学习任务。

## 详细阶段

### 阶段一：数据入口

**目标**

把公开文件稳定地转换成模型训练所需的 batch，并能解释每个字段。

**任务**

| 周次 | 任务 | 产物 |
| --- | --- | --- |
| 第 1 周 | 建立工程；实现 JSONL reader、validator、审计 CLI 和基础测试 | 数据审计报告、拒绝样本、测试结果 |
| 第 2 周 | 实现教学 BPE；适配官方 tokenizer；实现 Pretrain Dataset 和 collator | token/label 对照表、可重载 tokenizer、首个 batch |

**验收**

- 一条真实样本可以追踪到 token、attention mask 和 label。
- 坏行、字段错误和无有效监督可以被稳定定位。
- tokenizer 保存重载后结果一致，padding 不进入 loss。

### 阶段二：模型系统

**目标**

实现从 forward 到 optimizer update 的完整计算，并支持可靠中断恢复。

**任务**

| 周次 | 任务 | 产物 |
| --- | --- | --- |
| 第 3 周 | Embedding、RMSNorm、SwiGLU、残差、初始化和参数统计 | 不含 attention 的模块 smoke |
| 第 4 周 | causal attention、RoPE、GQA、Decoder、lm_head、CE 和 greedy generation | 可更新的 tiny Transformer |
| 第 5 周 | AdamW、参数组、LR schedule、token 平均、梯度累积和裁剪 | 单设备训练循环 |
| 第 6 周 | sampler 状态、validation NLL、原子 checkpoint、resume 和异常记录 | 可恢复训练入口 |

**验收**

- 关键前向和梯度与独立参考对齐。
- 未来 token 不影响过去位置的 logits。
- tiny batch 可以过拟合。
- 确定性条件下，连续训练与中断恢复得到相同结果。

### 阶段三：完整训练

**目标**

用真实数据跑通 Pretrain 和 SFT，并完成训练前后的配对评测。

**任务**

| 周次 | 任务 | 产物 |
| --- | --- | --- |
| 第 7 周 | 冻结 Pretrain 子集，完成短训、恢复、验证和生成 | Pretrain 实验报告 |
| 第 8 周 | 实现 chat 模板、assistant mask、截断和 SFT | SFT checkpoint 与前后对照 |
| 第 9 周 | 实现 KV cache、采样、停止条件、批推理和输出归档 | 固定评测结果与采样对照 |

**验收**

- 数据、配置、起点权重和评测合同均已冻结。
- 保存逐样本输出，而不只保存汇总分数。
- 能说明训练改善、退化和未确定的部分。

### 阶段四：离线后训练

**目标**

在共同起点上比较参数适配、教师监督和偏好监督。

**任务**

| 周次 | 任务 | 产物 |
| --- | --- | --- |
| 第 10 周 | LoRA 注入、冻结、保存、加载与合并 | LoRA 与全参 SFT 对照 |
| 第 11 周 | CE+KL、温度、teacher 冻结与可选 logits 缓存 | CE-only 与蒸馏对照 |
| 第 12 周 | chosen/rejected collator、sequence log-prob、reference 和 DPO loss | DPO 短实验 |

**验收**

- 核心公式通过手算、边界和梯度方向测试。
- 比较使用共同起点，并记录监督信息和计算预算差异。
- LoRA 合并前后输出对齐，teacher/reference 不发生更新。

### 阶段五：在线强化学习

**目标**

生成在线轨迹，并证明这些轨迹真实参与了模型更新。

**任务**

| 周次 | 任务 | 产物 |
| --- | --- | --- |
| 第 13 周 | 生成后端、response mask、old/ref log-prob、reward 和轨迹缓存 | 可追踪 rollout |
| 第 14 周 | 分组采样、advantage、ratio/KL、GRPO 与 CISPO | 在线 RL 短实验 |
| 第 15 周 | value head、critic、GAE/returns、PPO loss 与状态恢复 | PPO 端到端短训 |

**验收**

- policy 版本、生成参数、终止原因和奖励分项可以追踪。
- 先用固定轨迹验证数值，再进行在线训练。
- 真实 rollout 进入 loss；不能只凭 reward 上升判断能力提升。

### 阶段六：Agent 与交付

**目标**

把模型放入受控工具环境，并独立交付一项完整实验。

**任务**

| 周次 | 任务 | 产物 |
| --- | --- | --- |
| 第 16 周 | tool schema、调用解析、工具执行、观测回填、多轮终止和轨迹落盘 | Tool Use baseline |
| 第 17 周 | action mask、trajectory reward 和 GRPO 更新 | Agent RL 对照实验 |
| 第 18 周 | 从干净环境复现；独立增加数据变体或故障处理 | 可复现实验与总结报告 |

**验收**

- 工具输出与模型动作分开记录，gt 不泄漏给 policy。
- 训练前后使用相同环境和任务评测。
- 能独立完成数据检查、训练、恢复、评测和结论。

## 执行方式

每周按固定顺序推进：

1. **全貌**：明确输入、输出、入口和调用链。
2. **数据**：让一条真实样本走完整链路。
3. **实现**：先写核心计算，再增加工程层。
4. **执行**：运行测试和短实验，检查日志与产物。
5. **复盘**：解释结果，独立完成一个小变体。

每周保留四类证据：

| 证据 | 内容 |
| --- | --- |
| 数据 | 原始记录、转换结果、batch 字段和拒绝原因 |
| 代码 | 自己实现的模块、边界测试和参考对照 |
| 运行 | 命令、退出状态、日志、指标和产物 |
| 实验 | 问题、基线、固定条件、改变项、结果和结论边界 |

学习者亲手完成环境、工程、代码、测试、运行、排错、分析和归档。助手负责讲解、合同讨论、代码
审查和必要提示。助手直接提供完整实现或代操作时，记录协助范围；学习者需独立补写相关变体后再验收。

## 工程约定

所有阶段持续演进同一套工程：

```text
implementation/
  pyproject.toml
  src/minimind_lab/
    data/                     # 审计、Dataset、collator、sampler
    models/                   # tokenizer、Transformer、LoRA、value head
    objectives/               # CE、log-prob、KL、DPO、policy/value loss
    training/                 # optimizer、loop、schedule、checkpoint
    rollout/                  # 生成、奖励、工具和轨迹
    evaluation/               # NLL、离线评测、交互评测
  scripts/
  configs/
  tests/
  data/                       # 少量 fixture
  runs/                       # 配置、日志、指标、输出和 checkpoint
```

SFT、LoRA 和 DPO 在已有工程中增加数据适配或训练目标；在线 RL 再增加 rollout 与更新循环。
不为每个阶段另写一套互不相关的工程，也不把所有方法硬塞进一个 `train_step`。

## 实验约定

- 组件计算使用手算、参考张量、梯度方向和不变量验证。
- 可学习模块可以做 tiny overfit；reader 和 sampler 等非学习模块不使用过拟合测试。
- 数值对照前固定权重、输入、dtype、mask、reduction、dropout 和随机性。
- 第 7 周起，每个主要训练阶段至少做一次小型受控对照，一次只改变一个问题。
- 效果没有提升不自动代表实现错误；loss 下降也不自动代表任务能力提升。
- LoRA、蒸馏、DPO 和 RL 不必依次叠加；方法比较应从冻结的共同起点分支。

## 数据约定

每个阶段先让一条真实记录走完整链路：

```text
原始记录 -> 解析与校验 -> 分组切分 -> 模板与编码
-> Dataset -> collator -> batch -> loss -> 指标与输出
```

每个实际采用的数据子集都要记录：来源、版本、许可、字段、切分键、数量、拒绝原因、长度分布、
有效监督比例、人工抽查问题和文件 hash。必须先按文档、对话、prompt 或任务模板分组切分，再派生
chunks、候选回答或轨迹，避免同源数据跨 train/validation/test。

各阶段的重点字段：

| 阶段 | 输入 | 新增训练字段 | 重点检查 |
| --- | --- | --- | --- |
| BPE | 文本 | 字节、合并表、词表、token ID | Unicode、预切分、泄漏和权重兼容 |
| Pretrain | `text` | input_ids、labels、attention mask | 边界、特殊 token、截断、去重和 shift |
| SFT/LoRA | conversations | assistant mask、labels | 多轮监督、EOS、全 mask 和未知字段 |
| 蒸馏 | SFT 样本、teacher | teacher/student logits、KL mask | token 对齐、温度、KL 方向和缓存身份 |
| DPO | chosen/rejected | 两组 response mask 和 log-prob | 共同 prompt、长度偏差、相同答案和截断 |
| PPO/GRPO | prompt、在线 completion | old/ref log-prob、reward、advantage、value | reference 泄漏、终止、policy 版本和全组同分 |
| Tool/Agent RL | messages、tools、gt | call、observation、action mask、trajectory reward | gt 泄漏、动作边界和任务成功 |

## 设备与预算

最初环境为 Apple M4、24GB 统一内存；当前验收环境为 Windows 和 RTX 4060 Ti。确定性训练对照先在 CPU 完成，CUDA 状态往返另行验证；设备变化不能被当作跨硬件训练已一致的证据。各机器分别创建虚拟环境，不复制 .venv。

单元测试、数据处理和短跑优先本机。需要蒸馏、PPO 或多候选 rollout 时，先测量显存、速度和成本，
再决定是否租卡。付费运行前记录时间与费用上限、数据量、停止条件和产物回收方式。

如果自训 tiny 模型不足以支持有意义的偏好或 RL 实验，可以缩小任务，或在 tokenizer 和模型兼容的
前提下使用公开 SFT 权重；必须记录新的实验起点。

## 参考资料

- 官方源码：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)，冻结提交
  `f659b55761b754d306bd140573493a6543cafd7f`。
- 第一轮学习资料：[理论笔记](theory_notes/README.md)。

| 本轮周次 | 对应笔记 |
| --- | --- |
| 第 1–2 周 | [项目地图](theory_notes/week_01_project_map.md)、[Tokenizer 与 Dataset](theory_notes/week_02_tokenizer_dataset.md) |
| 第 3–6 周 | [模型主干](theory_notes/week_03_model_forward.md)、[Attention/RoPE](theory_notes/week_04_attention_rope_kvcache.md)、[Pretrain 循环](theory_notes/week_05_pretrain_training_loop.md)、[工程拆解](theory_notes/week_06_5_training_code_dissection.md) |
| 第 7–9 周 | [Pretrain 实操](theory_notes/week_05_5_pretrain_practice.md)、[SFT](theory_notes/week_06_sft_training_loop.md)、[推理采样](theory_notes/week_07_inference_sampling.md)、[实验工程](theory_notes/week_07_5_experiment_engineering.md) |
| 第 10–12 周 | [LoRA](theory_notes/week_08_lora.md)、[蒸馏](theory_notes/week_09_distillation.md)、[DPO](theory_notes/week_10_dpo.md) |
| 第 13–15 周 | [GRPO](theory_notes/week_11_grpo.md)、[PPO](theory_notes/week_11_ppo_prerequisites.md) |
| 第 16–18 周 | [Agentic RL](theory_notes/week_12_agentic_rl.md)、[Agent 实践待办](theory_notes/week_12_agentic_rl_practice_todo.md)、[历史实验](theory_notes/experiments/README.md) |
