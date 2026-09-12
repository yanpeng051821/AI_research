# Case Study: IFM K2 Horizon 与 MiniCPM5-2B

> 记录日期：2026-09-12
>
> 研究类型：前沿观察，不是复现实验
>
> 当前目的：通过两个近期开放的小模型案例，校准我们对完整训练链路、阶段 checkpoint、数据公开程度、后训练组合和评测证据的认识；不把观察直接升级为新的训练主线。

## 1. 先给结论

### MiniCPM5-2B：深读

MiniCPM5-2B 更适合成为下一阶段的 Gold Case：官方模型树同时提供 Base、Midtrain、SFT-only 和最终 RL + OPD checkpoint，使用标准 `LlamaForCausalLM` 架构，并公开了若干 UltraData 数据集和训练阶段说明。这样我们可以先做“阶段差异阅读和评测对照”，再判断是否值得做小规模复现。

它仍然不是一个可以直接照搬的完整复现 recipe。官方描述的 400B token 深思考 SFT、RL 和 OPD 的完整规模远超个人实验预算，公开的数据、代码和模型卡也不能自动证明每个阶段都能被外部独立重建。

### IFM K2 Horizon：记录

IFM K2 Horizon 的公开叙事非常适合研究“从预训练到 agentic post-training 的全生命周期”：官方声明会开放中间 checkpoint、数据或数据构造 recipe、代码、配置、训练日志和评测结果，并提供 0.9B 等小尺寸模型。

但截至本次查阅，官方 `xllm` 和 `horizon-post-train` GitHub 仓库的可见内容仍主要是 README、LICENSE 和占位说明；博客中承诺的完整训练代码尚不能视为已经可审查、可运行的公开实现。因此当前只能把它作为高价值观察案例，不能把它当作已验证的复现基线。

## 2. 这周为什么看它们

我们当前的主线不是追逐每个新模型，而是建立“独立训练计算、工程实现和可靠实验能力”。这两个案例正好覆盖两个有价值的观察角度：

| 案例 | 主要价值 | 与当前能力的关系 |
| --- | --- | --- |
| MiniCPM5-2B | 阶段 checkpoint、标准架构、公开数据入口和 SFT/RL/OPD 顺序较清晰 | 适合做训练工程地图和阶段评测对照 |
| IFM K2 Horizon | 把小模型、长链路训练、agentic post-training、日志和中间 checkpoint 放在同一叙事中 | 适合观察开放科学和 agent 训练的证据要求 |

这不是在问“哪个模型更强”，而是在问：一个研究者要怎样从最终权重倒推训练过程，并判断一项公开声明是否真的足以支持复现。

## 3. 案例 A：MiniCPM5-2B

### 3.1 官方披露的模型与阶段

官方模型卡将 MiniCPM5-2B 描述为约 2B 参数的 dense causal language model，架构为标准 `LlamaForCausalLM`，42 层，GQA（16 个 Q heads、2 个 KV heads），原生上下文长度 131072。官方模型树至少提供：

```text
MiniCPM5-2B-Base       # 预训练后
        |
MiniCPM5-2B-Midtrain   # mid-training 后
        |
MiniCPM5-2B-SFT        # SFT-only
        |
MiniCPM5-2B            # RL + OPD 后的最终版本
```

这个阶段树对我们很重要，因为它把“某个能力是否来自 SFT、RL 还是 OPD”变成了可以进行 checkpoint 对比的问题，而不是只能比较 Base 与最终模型。

### 3.2 官方描述的后训练链路

模型卡把 post-training 概括为：

```text
SFT -> RL -> OPD
```

官方说明使用大规模 deep-thinking SFT 建立推理和通用对话能力；随后使用针对数学、代码、写作、Agent 等方向的专门教师或数据进行 RL，再通过 OPD 进行蒸馏/分布对齐。模型卡还公开了部分数据入口，包括 UltraData-SFT、UltraData-SFT-Agent-2609 和 UltraData-RL-2609。

我们需要把这段话拆成三层理解：

1. `SFT` 解决的是示范行为和输出协议的建立。
2. `RL` 依赖可执行的 reward 或 verifier，把模型推向更高的任务结果。
3. `OPD` 不是普通的“再训练一遍”，而是用教师或多个专家的 token 分布继续约束学生模型。

这能帮助我们把 SFT、DPO、GRPO、PPO、蒸馏和 Agent RL 放回任务反馈的语境中，而不是把它们当成固定顺序的课程章节。

### 3.3 哪些内容对我们可复用

- **可复用的研究方法**：保存 Base/SFT/final 阶段，使用同一套固定评测和逐样本结果，观察能力增益与回归。
- **可复用的工程检查**：每个阶段都应有配置、数据版本、checkpoint、评测输出和运行日志。
- **可复用的模型选择**：标准 Transformers 架构降低了加载和框架接线的额外变量。
- **暂不可直接复用的规模**：400B token 级别的 SFT、专门教师、RL/OPD 数据生产和 131K 长上下文训练，不属于当前个人实验预算。

### 3.4 当前应如何使用它

当前不直接训练 MiniCPM5。等 Qwen3-0.6B SFT pilot 关闭后，做一次离线对照：

```text
MiniCPM5 Base -> SFT -> Final 的官方阶段语义
          对照
Qwen3 Base -> 我们的 SFT pilot -> 我们的评测结果
```

对照对象是“阶段定义、产物、指标和回归检查”，不是把两个模型的绝对分数直接横向比较。

## 4. 案例 B：IFM K2 Horizon

### 4.1 官方公开叙事

IFM 官方将 K2 Horizon 描述为包含 375B-A23B、36B-A4B、32B、7B、3.7B 和 0.9B 的模型家族，并声称覆盖从预训练到 reasoning/agentic post-training 的训练生命周期。官方重点强调：

- 中间 checkpoint；
- 训练数据或数据构造 recipe；
- 架构、训练代码和配置；
- 细粒度训练日志；
- 评测结果和最终权重；
- 多尺寸模型共享核心架构、词表、训练方法和评测基础设施。

官方博客还强调了长链路后训练，包括 mid-training、SFT、model merging、RL 和专门的 Agent 训练；小尺寸模型并不是简单缩小的 base model，而是经过能力分支和蒸馏等组合。

### 4.2 为什么重点看 0.9B

0.9B 与我们未来希望研究的小模型范围更接近。它适合观察：

- 小模型如何通过数据和后训练获得工具使用、数学或轻量 Agent 能力；
- 中间 checkpoint 如何帮助定位能力出现的阶段；
- 小模型是否需要专门的模型结构，还是可以在标准 causal LM 上通过训练获得目标行为。

但不能只看最终模型卡上的分数。必须进一步核对 checkpoint revision、训练数据版本、训练配置和评测 harness，否则无法区分架构收益、数据收益、推理时 prompt 变化和后训练收益。

### 4.3 证据边界

官方博客写明会开放完整基础设施和 Agent post-training code；然而本次查阅的官方 GitHub 仓库中：

- `ifm-ai/xllm` 当前可见文件主要是 `.gitignore`、`LICENSE` 和 `README.md`，README 只给出轻量预训练基础设施的项目说明；
- `ifm-ai/horizon-post-train` 当前可见文件主要是 `.gitignore`、`LICENSE` 和 `README.md`，README 明确写着 “Stay tuned”。

因此本卡对 IFM 的判断分为两级：

| 判断 | 证据状态 |
| --- | --- |
| K2 Horizon 的模型家族、训练生命周期和开放目标 | 官方博客/模型卡，可信地支持“官方声明” |
| 已经可以独立运行完整训练代码 | 本次查阅未验证，不能宣称成立 |
| 0.9B 的具体后训练增益来自哪一个阶段 | 需要 checkpoint、配置、逐阶段评测后才能判断 |
| 可直接复现官方分数 | 未证明，尤其需要核对评测 harness、提示词和版本 |

这是一个重要的研究习惯：官方宣布“会开放”与当前已经存在“可审查、可运行、版本固定的代码”，是两个不同事实。

## 5. 六维开放性检查表

我们以后看任何“全训练链路开源”的模型，都按下表检查，不只看最终权重。

| 维度 | MiniCPM5-2B | IFM K2 Horizon | 当前判断 |
| --- | --- | --- | --- |
| 最终权重 | 有 | 有 | 已满足 |
| 阶段 checkpoint | Base/Midtrain/SFT/Final 可见 | 官方声称有，需逐项核对 | MiniCPM5 更适合先深读 |
| 数据或 recipe | 有若干 UltraData 入口和说明 | 官方声称有数据或 recipe | 不等于完整数据可下载 |
| 训练代码 | 有官方工程和 cookbook，但未等于完整 MiniCPM5 复现脚本 | 官方仓库当前仍不完整 | 都不能直接宣称完整复现 |
| 配置与训练日志 | 部分可见，需继续核对 | 官方声称会提供 | 需要固定版本后再判断 |
| 评测与逐样本证据 | 有模型卡指标，仍需核对 harness | 有官方指标和 reward-hacking 审计叙述 | 不能只比较汇总分数 |

## 6. 对当前路线的影响

### 不改变的部分

1. 当前主线仍是 `Qwen3-0.6B` 的 SFT 工程与评测闭环。
2. 当前 pilot 的目标仍是验证数据处理、TRL 接线、显存记录、checkpoint/resume 和评测回归，不是追求 SOTA。
3. 不因为 MiniCPM5 的 RL/OPD 链路很完整，就跳过我们正在补齐的 SFT 基础工程。
4. 不因为 IFM 宣称全链路开放，就把尚未验证的代码和分数当成可复现事实。

### 可能增加的能力

- 阶段 checkpoint 管理：Base、SFT、RL/蒸馏阶段分别保存并评测。
- 训练产物清单：数据版本、代码 commit、配置、环境、显存峰值和评测 harness 一起落盘。
- 阶段增量评测：不仅保存最终分数，还要保存逐样本输出、错误类型和能力回归。
- 小模型训练报告：把“架构变化、数据变化、训练目标变化、推理设置变化”拆开记录。

## 7. 本周决定与后续动作

### 决定

- MiniCPM5-2B：`深读`，但暂不训练。
- IFM K2 Horizon：`记录`，等待官方训练代码和配置达到可审查状态。
- 新增实验：无。
- 当前 SFT pilot：继续完成，不被前沿观察打断。

### 下一次研究动作

SFT pilot 结束后，安排一个 2 至 3 小时的 MiniCPM5 阶段地图阅读：

1. 下载或读取 Base/Midtrain/SFT/Final 的模型卡和配置元数据。
2. 固定同一套短评测 prompt，先检查 tokenizer、chat template、EOS 和输出格式。
3. 逐阶段比较 validation NLL、生成行为、数学/代码/工具小评测和回归样本。
4. 记录哪些差异可以归因于阶段，哪些仍混有数据、prompt 或模型版本因素。
5. 只有当这个过程产生一个可控的小问题，才把它升级为后续实验。

## 8. 官方来源

- [IFM K2 Horizon 官方发布文章](https://ifm.ai/blog/k2/)
- [IFM K2 Horizon 0.9B 模型卡](https://huggingface.co/IFM/K2-Horizon-0.9B)
- [IFM xllm 官方仓库](https://github.com/ifm-ai/xllm)
- [IFM horizon-post-train 官方仓库](https://github.com/ifm-ai/horizon-post-train)
- [MiniCPM5-2B 官方模型卡](https://huggingface.co/openbmb/MiniCPM5-2B)
- [OpenBMB MiniCPM 官方仓库](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM5 官方模型集合](https://huggingface.co/collections/openbmb/minicpm5)

## 9. 复查条件

在以下任一条件满足前，不把 IFM K2 升级为复现实验：

- 官方训练代码出现可运行的固定 commit；
- 训练配置、数据版本或构造 recipe、checkpoint 和评测脚本可以互相对应；
- 至少有一个 0.9B 阶段可以在当前预算下做小规模行为验证；
- 我们已经完成当前 Qwen SFT pilot 的工程复盘，并能明确要验证 K2 的哪一个具体假设。

