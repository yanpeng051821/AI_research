# 第二阶段阅读清单：从理解训练链路到设计实验

更新：2026-11-06。[When Scaling Meets LLM Finetuning](2026-11-finetuning-scaling/README.md)
本轮主线阅读完成；具体状态与下一步见案例入口。

## 1. 从第一阶段接过来的问题

测试专用模型这条主线的第一阶段读了 [AnaDroid](2026-11-anadroid/README.md)，
认识了业务目标、数据处理、训练、评测与部署的完整链路。TinyStories 是此前的基础案例。

这一阶段留下四点认识与缺口：

- **有效不等于可用**：离线分数提高，还要结合判断质量、分析耗时和返工，定义团队的使用门槛。
- **已有配方不等于合适配方**：尚不清楚我们的任务需要多大模型、多少独立案例，以及哪些训练阶段。
- **成绩提高不等于原因清楚**：增加数据、重复训练和增加阶段可能同时改变计算量，需要可解释的对照。
- **流程描述还需具体证据**：答案可信度、数据隔离、截断损失和检查点选择，需要落实到数据与配置。

第二阶段围绕：**在限定的测试任务中，怎样选择模型、数据与训练方法，达到可验证的使用效果？**

## 2. 接下来顺序读

### 1. When Scaling Meets LLM Finetuning — ICLR 2024

- **读什么**：模型规模、微调数据量、LoRA 与全参数微调的对照设计和效果曲线。
- **为什么读**：直接承接“多大模型需要多少数据”，学习怎样控制变量、解释收益。
- **读完应能做什么**：设计固定模型、逐步增加独立案例的比较，区分数据量与训练预算。论文是翻译/摘要任务，不照搬其比例。
- [论文全文](https://arxiv.org/html/2402.17193v1) · [会议入口](https://iclr.cc/virtual/2024/poster/19432)

### 2. Tülu 3 — 先读 8B 的 SFT/DPO 路线

- **读什么**：真实样本、数据混合、监督范围、训练配置和阶段评测。
- **为什么读**：公开配方与阶段产物较清楚，帮助把实验设计落实为可追溯的训练流程。
- **读完应能做什么**：解释样本怎样进入训练、模型怎样继承前阶段结果，以及怎样判断提升和回退；不一次展开全部大模型与 RLVR。
- [技术报告](https://arxiv.org/abs/2411.15124) · [训练配方](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

## 3. 按问题补读

以下按需要选取，不要求全部读完。

- **少量试验怎样帮助选模型：Rectified Scaling Law（ICML 2024）**。
  有多模型、多数据规模的结果和拟合代码，适合研究曲线预测；跨 tokenizer 的 loss 可比性需谨慎。
  [论文](https://proceedings.mlr.press/v235/lin24j.html) · [代码与结果](https://github.com/linhaowei1/Fine-tuning-Scaling-Law)
- **小模型配方怎样落地：SmolLM2-1.7B / SmolTalk**。
  有 SFT/DPO 配置与公开数据，可作为 Tülu 的精简补充；不同尺寸的数据配方并不相同。
  [训练配方](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)
- **领域与通用数据怎样配比：CMR Scaling Law（EMNLP 2024）**。
  研究继续预训练的配比、token 量和能力保留；准备采用 CPT 时深入，不直接套用到 SFT/DPO。
  [论文](https://aclanthology.org/2024.emnlp-main.903/)
- **怎样验证模型完成任务：SWE-Gym（ICML 2025）**。
  连接训练轨迹、执行测试与数据量实验；研究自动修复 Agent 时优先，它不是静态根因分析任务。
  [论文](https://proceedings.mlr.press/v267/pan25g.html) · [项目](https://github.com/SWE-Gym/SWE-Gym)
- **怎样扩充可验证的软件任务：SWE-smith（NeurIPS 2025 D&B）**。
  有任务生成、执行环境和轨迹，可接在 SWE-Gym 后读；合成代码故障不能代表全部测试失败。
  [项目](https://github.com/SWE-bench/SWE-smith)
- **怎样建立可信根因评测：RCAEval**。
  有多源观测与故障真值；选择后端/微服务诊断时补读，关注盲测与泄漏，不作为现成微调配方。
  [项目](https://github.com/phamquiluan/RCAEval)

## 4. 完整项目的后续阅读入口

按问题选读相关部分；详细开放范围与调研依据见[来源记录](NEXT_CASES_2026-11_SOURCES.md)。

- **MiniCPM5 / UltraData**：看数据分级、轨迹验证与阶段能力变化，补充数据治理经验。[项目](https://github.com/OpenBMB/MiniCPM)
- **Puro-2B**：看阶段预算、数据配方与检查点对照，学习训练量和成本怎样记录；重点是预训练。[资料](https://huggingface.co/thu-pacman/Puro-2B-Base)
- **ZGCM-1**：看长上下文数据、监督规则与阶段产物，补充复杂训练工程的组织方式。[项目](https://github.com/zgcagi/ZGCM-1)
- **IFM K2 Horizon**：看阶段检查点和发布证据；公开演示不等同原全量训练。[项目](https://github.com/ifm-ai/horizon-post-train)
- **Qwen**：作为多规模模型与部署参照，选型时固定具体版本，不把系列当成单一案例。[入口](https://github.com/QwenLM/Qwen3)
- **LFM / Liquid AI**：看窄任务微调和部署成本，研究混合架构的取舍。[资料](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- **Nanbeige4.2-3B**：看轨迹过滤、行动评价和推理效率；完整生产数据与配方尚未确认开放。[资料](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
- **VibeThinker-3B / OpenThoughts**：看可验证推理的轨迹筛选、蒸馏与长度策略；不直接当作根因分析方案。
  [VibeThinker](https://github.com/WeiboAI/VibeThinker) · [OpenThoughts](https://github.com/open-thoughts/open-thoughts)

## 5. 本阶段的落点

完成前两项后，整理一页**团队实验设计草案**：限定任务、使用门槛、数据与真值、隔离方式、
基线、一个主要对照变量及预算。缺少的信息列为待确认，不填猜测值。

目标是能解释“为什么这样设计、结果能回答什么”，而不是读满清单。是否启动训练另行决定。
