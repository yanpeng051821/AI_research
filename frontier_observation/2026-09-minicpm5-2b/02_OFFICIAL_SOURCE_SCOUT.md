# MiniCPM5-2B：官方来源与证据审计

> 首次审计：2026-09-12
> 范围：只核对模型身份、阶段、数据入口、训练方法和可复现材料
> 下一锚点：继续做架构与训练阶段地图，不启动新实验

## 1. 研究框架

本轮不是验证 MiniCPM5-2B 是否达到官方宣称的 SOTA，也不是把它设为当前实验 baseline。
需要先回答的最小问题是：它是否提供了足够清晰的阶段 checkpoint 和官方材料，使我们
能够研究 Base、Midtrain、SFT、RL 与 OPD 的职责边界。

当前判断是“足以深读，尚不足以完整复现”。

## 2. 官方来源账本

| 来源 | 类型 | 当前用途 | 证据等级 |
| --- | --- | --- | --- |
| MiniCPM5-2B 模型卡 | 官方模型卡 | 架构、训练链路、评测声明、部署入口 | 支持官方声明 |
| MiniCPM5 模型集合 | 官方模型集合 | 核验 Base/Midtrain/SFT/Final checkpoint 身份 | 可直接核验 |
| OpenBMB/MiniCPM | 官方仓库 | 训练 recipe、部署和微调 cookbook | 可审查，但不等于完整原始训练代码 |
| 各阶段 `config.json` | 官方模型文件 | 比较模型结构和上下文配置 | 可直接核验 |
| UltraData-SFT-2605 | 官方数据卡 | 核验核心领域 SFT 数据的范围与格式 | 支持数据发布声明 |
| UltraData-SFT-Agent-2609 | 官方数据卡 | 核验工具、搜索、代码和通用 Agent 轨迹 | 支持数据发布声明 |
| UltraData-RL-2609 | 官方数据卡 | 核验可验证 RL 任务、领域与 reward 依据 | 支持数据发布声明 |
| JustRL II 说明 | 官方引用的训练方法材料 | 理解 critic-based RL 与长推理训练 | 尚需独立深读和代码核验 |

## 3. 阶段配置核验

2026-09-12 直接读取四个官方仓库的 `config.json`，得到：

| 配置 | Base | Midtrain | SFT | Final |
| --- | ---: | ---: | ---: | ---: |
| architecture | LlamaForCausalLM | LlamaForCausalLM | LlamaForCausalLM | LlamaForCausalLM |
| hidden size | 2048 | 2048 | 2048 | 2048 |
| FFN intermediate size | 6144 | 6144 | 6144 | 6144 |
| layers | 42 | 42 | 42 | 42 |
| query heads | 16 | 16 | 16 | 16 |
| KV heads | 2 | 2 | 2 | 2 |
| head dimension | 128 | 128 | 128 | 128 |
| vocabulary size | 130,560 | 130,560 | 130,560 | 130,560 |
| dtype | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| RoPE theta | 5,000,000 | 5,000,000 | 5,000,000 | 5,000,000 |
| max position config | 524,288 | 131,072 | 131,072 | 131,072 |

目前可确认四个阶段没有切换成不同的模型主体。阶段差异主要应来自权重、数据和训练目标，
而不是层数、hidden size 或 attention heads 的改变。

Base 的 `max_position_embeddings=524288` 与其余阶段的 `131072` 不同。这里只能记录配置
差异，不能据此宣称 Base 已在 524K 长度上完成训练或通过有效能力评测。需要结合
mid-training 配方、位置分布和长上下文评测继续核验。

四个仓库均提供 `chat_template.jinja`。模板文件存在不代表 Base/Midtrain 已通过对话数据
学会遵循该协议；对话行为仍应在 SFT 及后续 checkpoint 上分别验证。

## 4. 数据侧已确认内容

### SFT 数据

`UltraData-SFT-2605` 覆盖数学、代码、知识和指令遵循，并同时组织 Think 与 No-Think
样本。官方数据卡报告超过 1,500 万样本，但集合页当前显示的可见规模约为 1,220 万；
需要检查配置、split、版本和统计口径后才能解释差异。

`UltraData-SFT-Agent-2609` 面向 Tool-Use、Search-Agent、Code-Agent 和
General-Agent，包含工具调用、环境反馈、验证、错误恢复和最终交付轨迹。数据卡宣称约
50 万样本，但页面可见统计可能随配置和转换版本不同，需要固定 revision 后再计数。

### RL 数据

`UltraData-RL-2609` 当前数据卡给出 85,995 条可验证任务：

| 方向 | 样本数 | reward / verifier 依据 |
| --- | ---: | --- |
| Math | 32,412 | 提取答案后与 ground truth 匹配 |
| Code | 23,665 | 执行生成程序并跑测试用例 |
| Long-Context | 18,046 | 基于给定长文档回答并匹配答案 |
| Knowledge | 11,872 | 短答案与 ground truth 匹配 |

这里已经能看出 RL 数据和 SFT 数据的根本差异：RL 样本不只需要题目，还需要能够稳定
计算结果好坏的环境或 verifier。

## 5. 训练方法已确认内容

官方仓库将完整流程描述为：

```text
base training
-> mid-training
-> 400B-token deep-thinking SFT
-> math/code/agent/writing 等专项 RL teachers
-> OPD 合并 16 个专家，其中 5 个为 agentic experts
-> final model
```

官方描述 OPD 时，强调在 response 每个位置比较 student 与 teacher 的全词表分布，使用
reverse KL 构造 advantage，并复用 RL teacher 的 prompts。这个描述足以确定 OPD 不是
普通的 chosen/rejected DPO，但精确目标、采样更新顺序、teacher 选择和实现代码仍需核验。

官方仓库提供的 TRL、LLaMA-Factory、ms-swift、Unsloth 和 XTuner 文档主要是用户在最终
模型或基座上做 LoRA/SFT 的 cookbook，不是 MiniCPM5 原始 400B-token SFT、RL 和 OPD
全流程的复现代码。两者不能混为一谈。

## 6. 当前关键未知项

| 未知项 | 为什么重要 | 是否阻塞继续学习 |
| --- | --- | --- |
| Base training 的精确 token 预算、数据 mixture 和 schedule | 决定基础能力来源 | 不阻塞阶段地图，阻塞完整复现 |
| Mid-training 的目标数据比例和长度课程 | 决定能力增强与 128K 上下文来源 | 阻塞 mid-training 归因 |
| 400B SFT 的样本重复、混合比例和训练配置 | 公开数据量不能直接推出训练 token | 阻塞完整 SFT 复现 |
| JustRL II 的精确算法与代码 | 决定 critic、advantage 和长轨迹稳定性 | 阻塞 RL 实现判断 |
| 16 个 RL experts 的身份、数据和 checkpoint | 决定 OPD 合并了什么能力 | 阻塞 OPD 归因 |
| OPD 的完整 loss、rollout 和更新代码 | 文字描述不足以数值复现 | 阻塞 OPD 复现 |
| 每阶段同合同、逐样本评测 | 决定增益是否真能归因于阶段 | 阻塞独立效果结论 |

## 7. 下一步决定

继续留在研究阶段。Base Training 的阶段地图已经沉淀到 `03_BASE_TRAINING_STAGE.md`；
下一步先沿数据工程链路理解原始文本如何经过清洗、分词、文档切分和 packing，成为
DataLoader 可消费的预训练 sequence。之后再依次讨论 sampler、训练 step、稳定性监控和
checkpoint/resume，不提前跳到 Mid-training、SFT、RL 或 OPD。

当前不下载四套完整权重，不运行 MiniCPM5 训练，也不改变正在进行的 Qwen3-0.6B S1。

## 8. 检索记录

本轮检索围绕以下问题展开：

- 官方是否确实提供 Base、Midtrain、SFT 和 Final 四个阶段；
- 各阶段是否使用同一模型架构；
- SFT、Agent SFT 与 RL 数据是否有官方数据卡；
- RL 数据如何得到可验证 reward；
- 官方仓库提供的是原始训练代码，还是面向下游用户的微调 cookbook；
- OPD 的公开描述是否足以支持数值复现。

保留的来源均为 OpenBMB 官方仓库、官方 Hugging Face 模型/数据页面，或官方直接引用的
方法材料。搜索结果中的第三方模型介绍未用于关键判断。
