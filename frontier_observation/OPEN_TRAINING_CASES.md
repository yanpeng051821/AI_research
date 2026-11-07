# 开放训练案例目录

初筛日期：2026-10-29；2026-11-02 增补 AnaDroid。收录 34 个项目或论文条目，另区分训练框架。
状态：资料初筛，未完成所有论文精读，也未验证完整复现。

2026-11-05：AnaDroid 本轮阅读完成，新的优先级与点名项目复核统一见
[下一批案例](NEXT_CASES_2026-11.md)及[来源记录](NEXT_CASES_2026-11_SOURCES.md)。
下方为历史广泛候选目录，未在本轮重新逐项核验，不把早期“首批阅读”当作当前顺序。

## 1. 使用方式

**这里负责查找案例和资料，不安排阅读进度。** 每项记录研究切入点、已找到的开放材料及
关键限制；“优先”“候选”“参考”是当前学习用途的判断，不是模型性能排名。

怎样准备、讲解、讨论和记录，统一见[案例带读方式](CASE_READING_GUIDE.md)。
筛选依据放在第 3 节，需要核查入选条件时再看，不作为开始阅读的前置清单。

## 2. 候选目录

索引负责找材料；各案例的完整分析仍放在独立目录。暂不为所有候选创建空目录。

### 2.1 实现与课程

| 条目 | 本轮已找到的材料 | 适合解决的问题 | 当前定位与限制 |
| --- | --- | --- | --- |
| [MiniMind](https://github.com/jingyaogong/minimind) | 模型、数据入口、多阶段训练实现与教程 | 把张量计算、数据格式和工程链路串起来 | 已有实践主线的实现参照；不因阶段齐全就视为配方最优性的证据 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 简洁 GPT、训练与数据准备代码 | 看清预训练的最小实现 | 按需查代码；不是系统性的数据配方研究 |
| [nanochat](https://github.com/karpathy/nanochat) | tokenizer、预训练、微调、评测和推理实现 | 看低复杂度端到端系统如何组织 | 候选；固定提交和实际运行条件，不直接采用标题中的价格作预算 |
| [CS336](https://cs336.stanford.edu/) | 官方讲义、作业与测试入口 | 补模型、系统、数据、规模和后训练的方法基础 | 按研究问题选章节；课程不是一个有统一最终结果的模型案例，2025 / 2026 版本分开 |
| [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | speedrun 实现、历史提交和记录 | 在明确指标与硬件约束下研究训练效率 | 系统专题候选；达到一个 loss 的速度不等于通用能力全面提高 |

### 2.2 基础模型

| 条目 | 本轮已找到的材料 | 研究切入点 | 当前定位与限制 |
| --- | --- | --- | --- |
| [TinyStories](https://arxiv.org/abs/2305.07759) | 论文、数据、模型入口 | 受限数据分布与小模型能力 | 2026-11-02 本轮阅读完成；数据因素的系统归因仍缺少直接证据，结论见[案例目录](2026-10-tinystories/README.md) |
| [SmolLM2](https://arxiv.org/abs/2502.02737) | 论文、数据、代码、配方与 checkpoint 入口 | 容量、数据选择、阶段混合 | 首批阅读；不同尺寸不能默认共用配方，阅读范围见[指引](CASE_READING_GUIDE.md) |
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | 报告、训练代码、配置、数据说明和中间模型 | 固定小模型的较长训练与阶段能力变化 | 候选；公开来源说明不自动等于当前可取得完整原始语料 |
| [Pythia](https://github.com/EleutherAI/pythia) | 多规模模型、中间 checkpoint、训练配置与顺序说明 | 相近数据暴露下比较规模和学习过程 | 优先的规模专题；需区分不同版本、去重设置及数据的实际可访问性 |
| [OLMo 2](https://github.com/allenai/OLMo) | 报告、分阶段配置、数据与模型入口 | 稳定训练、数据阶段和工程证据 | 中期候选；先读一个尺寸和一个阶段，不同时展开所有变体 |
| [OLMo 3](https://github.com/allenai/OLMo-core) | 官方训练脚本、模型与配方入口；后训练另见 Open Instruct | 基础训练到后训练的可追溯链路 | 中期候选；不同代际使用的 trainer 和数据配置需分别核验 |
| [Puro-2B](https://arxiv.org/abs/2608.27370) | 论文、[阶段数据](https://huggingface.co/datasets/thu-pacman/Puro-2B)、[Megatron 分支](https://github.com/thu-pacman/Puro-Megatron)与权重入口 | 有限预算下的数据、优化和课程安排，以及对下游 SFT 的影响 | 中期重点候选；复杂改动较多，不能将标题成本理解为项目所有成本 |
| [MiniCPM5-2B](https://github.com/OpenBMB/MiniCPM/blob/main/README-cn.md) | 模型卡、阶段说明、UltraData 数据入口和工具代码入口 | 数据分级与 base / mid / post-training 的能力衔接 | 延续[已有案例](2026-09-minicpm5-2b/README.md)；不把框架开源等同于本模型全流程精确脚本已齐备 |

Puro 的公开代码审计明确列出未纳入该仓库的 SFT、数据消融及部分实验内容，见
[范围说明](https://github.com/thu-pacman/Puro-Megatron/blob/puro-2b/docs/ORIGIN_COMMITS.md)。
它可以研究预训练与下游能力的关联，但本轮不宣称已获得全套实验的一键复现实现。

### 2.3 数据与规模方法

| 条目 | 本轮已找到的材料 | 研究切入点 | 当前定位与限制 |
| --- | --- | --- | --- |
| [FineWeb / FineWeb-Edu](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html) | 实验说明、数据、分类器和处理代码入口 | 数据清洗、去重、质量与覆盖的取舍 | 第一批横向对照；不预设教育质量筛选适合所有任务 |
| [DCLM](https://github.com/mlfoundations/dclm) | 论文、标准化训练、数据、评测及实验结果入口 | 固定条件比较数据选择 | 第一批横向对照；汇总指标需区分版本 |
| [DoReMi](https://github.com/sangmichaelxie/doremi) | [NeurIPS 2023 论文](https://papers.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html)、代码与[作者说明](https://crfm.stanford.edu/2023/09/14/doremi.html) | 用代理模型学习域混合权重 | 优先的配比专题；优化目标和跨规模迁移仍有条件，不是任意任务的万能比例生成器 |
| [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) | 论文、[Datablations 代码及模型数据入口](https://github.com/huggingface/datablations) | 独立数据不足时，规模、重复和预算怎样取舍 | 优先的规模专题；重复数据的收益取决于所研究的条件 |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | 论文与规模分配实验说明 | 固定训练计算预算下的模型—数据分配 | 理论参照，不列为完整开放复现案例；不把经验比例直接当作专项 SFT 规则 |

### 2.4 后训练与专项能力

| 条目 | 本轮已找到的材料 | 研究切入点 | 当前定位与限制 |
| --- | --- | --- | --- |
| [AnaDroid](https://pkun.life/papers/issta26/) | 论文、本地源码与 10 条脱敏示例已核对 | 从测试失败分类、根因分析和修复建议目标设计完整训练与验证链路 | 2026-11-05 本轮阅读完成，见[案例入口](2026-11-anadroid/README.md)；未运行原训练，完整语料与权重不具备已验证的公开复现条件 |
| [SmolLM3](https://huggingface.co/blog/smollm3) | 架构与阶段配方、公开数据来源、训练代码及日志/checkpoint 入口 | 在小模型中结合基础、多语言、长上下文和推理能力 | 中期候选；工程更复杂，先选数据或一个阶段，不全盘展开 |
| [Tülu 3](https://arxiv.org/abs/2411.15124) | 论文、数据、[实际后训练说明](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)与模型入口 | SFT、DPO、RLVR 的阶段作用，技能混合和独立评测 | 后训练主案例候选；底座来自 Llama，不因此拥有其完整预训练数据 |
| [OmniEdu](https://arxiv.org/abs/2609.23088) | 论文、[官方仓库](https://github.com/haolpku/Omni-Edu)、[训练数据](https://huggingface.co/datasets/lhpku20010120/Omni-Edu)及模型入口 | 从领域目标拆解能力，组织监督数据，并观察专项提升和基础能力变化 | 领域适配优先候选；已有模型上的全参数 SFT，不是从零预训练；配比最优性与各处理步骤的独立贡献尚待验证 |
| [OpenThoughts / OpenThinker](https://github.com/open-thoughts/open-thoughts) | [论文](https://arxiv.org/abs/2506.04178)、数据生成、训练与评测入口 | 推理题源、教师轨迹、过滤和数据规模 | 专项优先候选；论文报告大量受控实验，不等于本轮已逐项审查；各代教师与配方不同 |
| [Open-R1](https://github.com/huggingface/open-r1) | SFT / GRPO 代码、数据、模型与实验文章 | 推理蒸馏与可验证奖励链路 | 历史案例；官方现提示不再维护，训练侧转向 TRL；不能视为已经等价重建 DeepSeek 全部训练细节 |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) | 受限任务、奖励/训练代码、规模对照与日志入口 | 在明确验证器下研究在线 RL 与模型起点 | 专项候选；官方现提示不再维护，建议使用 veRL；任务内现象不是通用推理保证 |
| [xLAM / APIGen](https://github.com/SalesforceAIResearch/xLAM) | 数据、模型、生成验证说明及 ActionStudio 训练入口 | 从单轮函数调用到多轮工具任务的数据组织 | 工具分支优先候选；xLAM 与 APIGen-MT 等版本分别看，数据许可与覆盖范围分别查 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | [论文](https://arxiv.org/abs/2409.00920)、公开数据与模型 | 工具覆盖、对话构造和分层验证 | 数据设计候选；官方组织说明公开数据为训练集子集，不宣称完整重建所有合成及训练过程 |
| [ZGCM-1](https://github.com/zgcagi/ZGCM-1) | 报告、阶段训练/处理代码、[数据卡](https://huggingface.co/datasets/zgcagi/ZGCM-1-Data)、模型与日志入口 | 数学与搜索目标怎样贯穿 pretrain / midtrain / SFT | 后期重点候选；能力和系统变量多，不适合第一篇；部分数据为 index-only 且需上游授权 |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | ICML 2025 论文、任务环境、训练及验证材料入口 | 软件工程任务中的执行反馈、轨迹与 verifier | 软件分支候选；环境和交互成本高，不等于静态测试分类，也不直接视为完整 RL 案例 |
| [Llama-3-SynE](https://github.com/RUC-GSAI/Llama-3-SynE) | [报告](https://arxiv.org/abs/2407.18743)、数据、代码与模型入口 | 继续预训练时的能力扩展、配比和课程策略 | 能力保留专题候选；报告中的“不损伤原能力”仅限已测范围，不能推广为绝对保证 |
| [Don't Stop Pretraining](https://github.com/allenai/dont-stop-pretraining) | [ACL 2020 论文](https://arxiv.org/abs/2004.10964)、代码与任务数据说明 | 区分领域适配和任务适配 | 概念/实验参照；以 RoBERTa 等设置为背景，不直接搬成 decoder-only 对话模型配方 |

#### OmniEdu 研究切入

**入选理由**：先将教育目标拆成解题、课程知识关联、学习困难诊断和教学引导，再组织监督
数据，能连接本轮“目标能力 → 数据 → 训练 → 评测”的主线。HF 当日论文排名只作为
发现材料的线索，不作为方法有效或配方最优的证据。

**优先问题**：为什么不同目标行为需要不同的样本与指令？教育与通用监督怎样组合？
数据分配按样本数还是监督 token 控制？评测怎样区分“答对题”和“给出合适的引导”？

**已查范围**：论文 v1 的数据流程、训练配置和评测说明，以及官方仓库和 HF 数据卡。
仓库可见数据准备与训练目录，实际文件、依赖、原始日志及全链路重建仍需后续审查，
本轮没有运行训练。资料入口见[论文全文](https://arxiv.org/html/2609.23088v1)与上表。

**证据边界**：

1. 论文目前未见系统拆分各清洗步骤和通用数据比例的消融，整体方案有效不等于每一步
   必要，也不证明当前比例最优。
2. 三个尺寸的模型起点并非只改变参数量，不能把它当作模型大小与数据量关系的纯变量实验。
3. 通用评测的总体改善不代表所有子项没有回退，更不等于真实课堂学习效果已经得到验证。
4. 涉及图像和文本；深读时明确哪些结论来自文本任务，哪些依赖多模态输入，不混作同一种能力。

### 2.5 开放权重与产业参照

这些项目仍值得读，但本轮不将其列为“资料已齐全的全链路 gold case”。

| 条目 | 目前可研究的内容 | 本轮尚未取得的关键证据 | 当前定位 |
| --- | --- | --- | --- |
| [LFM2](https://www.liquid.ai/research/lfm2-technical-report) | 硬件约束下的架构选择、蒸馏与后训练；[350M 模型卡](https://huggingface.co/LiquidAI/LFM2-350M) | 未确认完整预训练语料及训练全过程的精确代码/配置 | 架构和端侧能力参照；开放权重不等于全训练链路开放 |
| [Qwen3](https://arxiv.org/abs/2505.09388) | 多尺寸模型、训练阶段与后训练策略 | 本轮未找到完整原始训练语料及原始全流程运行配方 | 技术报告参照和潜在基座；不按系列名称合并不同版本 |
| [Llama 3 系列报告](https://arxiv.org/abs/2407.21783) | 数据工程、规模实验、多阶段后训练与评测组织 | 原始完整训练数据和对应全流程产物未由本轮取得 | 工业决策参照；权重许可与数据开放程度分别判断 |

### 2.6 框架区分

用户提到的 `openrl` 不能直接替换成 `Open-R1`，这里分别保留：

1. [OpenRL](https://github.com/OpenRL-Lab/openrl)：通用强化学习框架。
2. [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)：语言模型后训练 / RL 框架。
3. [Open-R1](https://github.com/huggingface/open-r1)：有具体数据和模型产出的推理研究项目。

框架用来研究实现和系统组织；要研究某种配方为何有效，仍需绑定一项具体实验。

## 3. 筛选依据

### 3.1 研究价值

优先选择能回答以下至少一个问题的材料：

1. **能力与数据**：数据内容、难度、格式和覆盖范围，如何影响目标行为？
2. **模型与规模**：模型容量、独立数据量、训练 token 和计算预算，如何一起影响结果？
3. **能力平衡**：专项数据、通用数据及训练顺序，如何影响专项提升和原有能力？

参数量小不自动意味着案例简单；阶段多、榜单高或论文新，也不自动意味着更适合先读。
有明确对照、失败结果和适用边界的较早论文，可能比新模型的发布介绍更有研究价值。

### 3.2 开放范围

分别检查下面几项，不用一个“开源”标签替代全部判断：

1. **说明**：论文或报告是否交代目标、数据、配置和评测？
2. **数据**：公开的是最终语料、子集、来源清单、筛选代码，还是仅有配比说明？
3. **实现**：公开的是实际训练代码和配置，还是只有模型推理、微调示例或通用框架？
4. **产物**：是否有权重、中间 checkpoint、曲线、日志和消融结果？
5. **条件**：许可、访问申请、教师 API、硬件和版本依赖，会不会阻止重建？

本轮检查以官方论文、README、模型卡、数据卡和部分具体文件为主，未下载大规模权重或
语料，未实际执行训练，也未逐个验证日志链接的全部内容。目录中的“有入口”不代表
已经证明所有文件可下载，更不代表已经验证逐位复现。真正深读时再固定精确版本。

### 3.3 证据边界

- 官方宣称、论文实验结果、实际开放文件和我们的推断，分别记录。
- 某项目采用一种配方，不等于实验已经证明该选择是提升的原因。
- 跨论文榜单不直接拼成对照：模型起点、数据、算力、解码预算和评测版本可能不同。
- 预训练、继续预训练、SFT 和在线 RL 的数据量与监督信息不同，不共用一个比例定律。
- 阅读成本与复现成本分开。能分析大模型案例，不意味着必须承担其训练费用。

## 4. 归档边界

本文件是筛选索引，不另设一份目标路线。MiniMind 已有实践安排保持不变，本轮资料研究
也不自动增加实现作业。已有 MiniCPM5、IFM 材料保留在各自目录。

正式开始研究的案例按约定建立独立目录，具体文档随讨论增加，避免为候选铺出大量空文档。
暂不新增第二套庞大的目录分类；数据、规模、训练阶段等通过案例内的主题索引交叉链接。
