# 下一批案例：来源核验记录

核验日期：2026-11-05。仅使用论文、正式会议入口、作者仓库和官方模型/数据卡。
研究范围为阅读选材，不做性能复测；未运行训练，未下载大规模权重或语料。
排序与短结论见[NEXT_CASES_2026-11.md](NEXT_CASES_2026-11.md)。

## 1. 核验口径

- **读到**：实际获取全文或 README，并检查相关段落。
- **找到文件**：通过仓库树或数据文件索引确认存在；不等于运行成功。
- **开放声明**：只代表作者说明，不替代文件和训练身份核对。
- **未确认**：访问失败、没有匹配文件或尚未逐项审计；不直接判断未公开。
- 既有 2026-10-29 清单为历史初筛，本轮涉及条目按以下证据更新，其他条目不重新背书。

## 2. 正式会议与方法研究

### S1. When Scaling Meets LLM Finetuning

- [ICLR 2024 官方会议页](https://iclr.cc/virtual/2024/poster/19432)已读取，确认题目、作者与会议。
- [arXiv 2402.17193v1 全文](https://arxiv.org/html/2402.17193v1)已读取实验设置、结果与局限。
- 两组双语模型，1B/2B/4B/8B/16B；翻译与多语言摘要，比较全参数、Prompt、LoRA。
- 附录：最多 200K 步或 100 epoch，部分设置 300K 步；训练预算并非全部等量。
  不能把它当成统一固定 FLOPs 的数据规模实验。主要拟合 PPL，补 BLEURT/ROUGE-L。
- 作者明确局限于所测生成任务，数据质量、多模态和多任务仍待扩展。OpenReview 本次
  返回 403/错误页，改从官方会议页与 arXiv 核对。未确认该研究完整原训练工程开放。

### S2. Selecting Large Language Model to Fine-tune via Rectified Scaling Law

- [ICML 2024 / PMLR](https://proceedings.mlr.press/v235/lin24j.html)和
  [2402.02314v2](https://arxiv.org/html/2402.02314v2)已读取。
- [作者仓库](https://github.com/linhaowei1/Fine-tuning-Scaling-Law) HEAD 树：
  `5214568013af65e3cefd03860ecd7bc9358b308c`，25 个文件，未截断。
- README 说明 30 个模型、WMT19/Gigaword/FLAN、0 至 1,638,400 子集规模；已实际获取
  `benchmark/flan.csv`，树中有另外两份 CSV、`src/fit_law.py`、`src/model_select.py`。
- 核心是少量试验预测完整数据后的表现和选型，不是原始训练全闭环，也没有测试领域
  的人类可用门槛。模型身份、损失可比性与适用范围需后续深读。

### S3. CMR Scaling Law

- [EMNLP 2024 正式入口](https://aclanthology.org/2024.emnlp-main.903/)与 PDF 已读。
- 金融、学术论文领域，主实验模型规模含 460M、940M、1.6B、3.1B；定义通用损失
  容忍范围下的混合比例，附录公开拟合系数。不能将 loss 保留等同所有通用能力保留。
- 本轮未找到完整原始训练代码/权重入口；它是方法证据优先，不按完整开放工程评级。

### S4. Don't Stop Pretraining

- [ACL 2020 正式入口](https://aclanthology.org/2020.acl-main.740/)已读取，确认四域、八个
  分类任务、DAPT/TAPT 的比较。它不是 decoder-only 对话模型的直接配方。
- 历史作者仓库入口为 [allenai/dont-stop-pretraining](https://github.com/allenai/dont-stop-pretraining)，
  本轮未逐文件复查，不称已验证复现。

## 3. 软件任务与可验证数据

### S5. SWE-Gym

- [ICML 2025 正式页](https://proceedings.mlr.press/v267/pan25g.html)已核对标题，
  [2412.21139v1 全文](https://arxiv.org/html/2412.21139v1)已读数据量实验与限制。
- [仓库](https://github.com/SWE-Gym/SWE-Gym) HEAD 树
  `005c2fbeda594d329f53770707b2ae7832aa6ac8`，41 个文件；有 OpenHands/Moatless 指引、
  verifier 数据准备/评测脚本和模型数据入口。README 报告 2.4K 任务、11 个 Python 仓库。
- 全文包含到 491 条专家轨迹的训练数据量曲线、不同模型及推理预算实验；小样本曲线
  不等于测试领域普遍 scaling law。重复容易任务带来的偏置也有讨论。
- 当前 README 警告环境中存在 git 历史捷径，使用前需清理；必须区分当前代码与论文
  版本。首次尝试的 pan25c 链接不是这篇论文，已纠正为 pan25g，未引用错误页面。

### S6. SWE-smith

- [作者仓库](https://github.com/SWE-bench/SWE-smith)标注 NeurIPS 2025 D&B Spotlight，
  [2504.21798v1](https://arxiv.org/html/2504.21798v1)已读数据生成、训练与消融。
- HEAD 树 `9b74ac08118a85c39c356802f7961893af73e07f`，344 个文件；有任务生成、
  harness 和训练轨迹相关代码。README 给出 52K 任务、26K 轨迹及约 5K 训练子集入口。
- [数据](https://huggingface.co/datasets/SWE-bench/SWE-smith)、
  [轨迹](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)、
  [模型](https://huggingface.co/SWE-bench/SWE-agent-LM-32B)为作者链接，未下载实体。
- 任务保留条件包含破坏单元测试，训练保留成功轨迹；是软件修复而非直接的移动失败分析。
  官方教程页本次 SSL 失败，不能称教程端到端已验证。会议身份依据作者声明，未独立核对会场页。

### S7. RCAEval

- [仓库 README](https://github.com/phamquiluan/RCAEval)与
  [数据卡](https://huggingface.co/datasets/phamquiluan/RCAEval)已读取。
- 作者链接 WWW 2025 DOI `10.1145/3701716.3715290`，另有 ASE 2024 v1；不能误记为 ICSE。
- 当前发布说明 9 个数据集、735 个案例；有 metrics/logs/traces、注入时间与服务根因。
  数据卡明确故障注入、重复次数及缺失案例，不能将全部称为真实生产事故。
- 目录名/索引直接编码根因，评测应使用盲任务视图；当前 README 提供 LLM agent 接口。
  是 RCA 评测资料，不直接给出微调数据配方；适配接口功能测试还需明确任务差异。

## 4. 透明训练配方

### S8. Tülu 3

- [open-instruct](https://github.com/allenai/open-instruct)树
  `ab4132769c4f282e0e3769446232bcc01ccd273c`，1134 个文件，包含多个 SFT/DPO 配置。
- 已读 [docs/tulu3.md](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)：
  8B SFT 命令绑定 Llama-3.1-8B 与 tulu-3-sft-mixture，给出长度、epoch、LR、batch、
  输出与阶段模型入口；作者称接近原始命令，原设置 64 H100，并说明调整累积的方法。
- 不把当前主分支或 `model_revision=main` 当成冻结的原运行身份；内部实验链接也
  不等于公共日志。[报告](https://arxiv.org/abs/2411.15124)按技术报告引用，未核对正式会场。

### S9. SmolLM2

- [smollm](https://github.com/huggingface/smollm)树
  `438be18dae658a638f3a6976fdf5190bb6d68619`，712 文件，确认 SmolLM2 预训练配置与评测目录。
- 已读 [后训练配方](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)
  README 和 `sft/config.yaml`：1.7B 用 SmolTalk，135M/360M 用 smol-smoltalk，再做 DPO。
- [预训练说明](https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md)
  给出 native checkpoint 和继续预训练入口。不同代际必须区分；普通微调示例不能冒充
  生产 recipe。实际数据下载、训练与逐位复现均未做。

### S10. MiniCPM5-2B

- [MiniCPM 仓库](https://github.com/OpenBMB/MiniCPM)树
  `316cfb1cea39f39cfa16b4f5703b77495c2340be`，228 文件；已读当前 README 与微调说明。
- 主文给出 base/mid/post、SFT/RL/OPD 路线，链接 Base/SFT/最终模型及 UltraData。
- 已读 [UltraData-SFT-Agent-2609 卡](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609)：
  约 500K 轨迹，字段 uuid/messages/tools/source/domain，说明轨迹验证、过滤和适用限制。
  公开上游混合数据保留各自条款，不把统一标签当作所有上游材料同许可。
- 分阶段模型与数据入口较多，但完整原始采集数据、生产 RL/OPD 精确运行配置未逐项
  核验；约 400B SFT token 的报告不能直接当作团队小规模训练预算。

## 5. 用户点名模型与开放边界

### S11. Puro-2B

- 已读 [Base 模型卡](https://huggingface.co/thu-pacman/Puro-2B-Base)、
  [数据卡](https://huggingface.co/datasets/thu-pacman/Puro-2B)和数据根目录索引，确有 phase1、phase2。
- Puro-Megatron HEAD 树 `a7b80e873a0b5e1820ae425b9abd1b9ec578dc5c`；
  [源码范围说明](https://github.com/thu-pacman/Puro-Megatron/blob/puro-2b/docs/ORIGIN_COMMITS.md)
  明确排除部分 SFT/课程、数据消融、长上下文、DCU 移植及原运行产物。
- 模型卡提供 Uniform 不同 Phase2 预算、Curriculum、SMA 输入等模型入口，说明拟合
  无不确定区间，课程、延续训练、参数平均尚未完全独立归因。最终卡是 Base，不是 Instruct。
- 标题低成本口径与最终生产成本不同，且均非全部研究成本。按技术报告看，不标注未确认会议。

### S12. ZGCM-1

- [仓库](https://github.com/zgcagi/ZGCM-1)树
  `369f5ba5440fedc2d2f112946ded7f7aa13ec57b`，7189 文件；已读根 README、SFT README、数据分类目录。
- 有 data-process/pretrain/midtrain/sft，SFT 配置明确长上下文、token accounting、
  运行 manifest、硬件和短跑验收范围；短跑不证明收敛质量。
- 官方数据入口 `https://huggingface.co/datasets/zgcagi/ZGCM-1-Data` 本次返回 401。
  不能把历史记录中的开放描述更新成“全部数据已下载核验”，也不能据此断言数据永久不可用。
- [报告](https://arxiv.org/abs/2609.13356)，7.39B 从头训练，数学与 Agent 搜索主线；
  256K 和大集群条件使其不适合下一篇全工程复现。

### S13. IFM K2 Horizon

- 已读 [0.9B 模型卡](https://huggingface.co/IFM/K2-Horizon-0.9B)、
  [xLLM](https://github.com/ifm-ai/xllm)、[后训练 README](https://github.com/ifm-ai/horizon-post-train)。
- xLLM HEAD 树 `692bc256fac19094cb12eee380aadb0724376784`，324 文件，有 0.9B/3.7B/7B 示例。
  模型 refs API 确认 mid、RL 等实际标签，卡称 branches 而 API 当前为 tags，引用时应核对 revision。
- 0.9B 卡最后更新标 2026-09-28，报告仍列为 In Progress，预计日期已经过去；不能把预计
  发布视为完成。W&B、TxT360-v2 等入口存在但本轮未逐项下载复核。
- 后训练 README 的可运行演示为 32 coding tasks 过拟合，任务需自备；原示范用早期
  checkpoint，当前公开 checkpoint 结果可能不同，完整集成仍需在目标集群测试。
  较 9 月旧记录有进展，但不是原全部后训练链路已经复现。

### S14. Qwen

- 已读 [Qwen3 官方 README](https://github.com/QwenLM/Qwen3)，本轮不宣称选定系列最新版本。
  存在多尺寸、部署和微调入口，适合作为候选模型；未确认原始完整训练数据和各阶段
  精确配置全部发布。MoE 总参数与激活参数、Instruct/Thinking、不同版本分别比较。

### S15. Liquid AI / LFM

- 已读 [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B)、
  [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)模型卡和官方微调入口。
- 有混合卷积/注意力架构、训练量及大致配比描述，后者链接 CPT/SFT/DPO/GRPO notebooks。
  部分原始材料来自 licensed sources，未确认其全量发布；不能把 notebook 当作生产训练代码。
- 小模型窄任务和部署效率有参考价值，知识密集或编程任务有作者适用提醒；具体许可及
  海光支持需在实际选型时核查。

### S16. VibeThinker-3B

- 已读 [模型卡](https://huggingface.co/WeiboAI/VibeThinker-3B)和
  [2606.16140v1](https://arxiv.org/html/2606.16140v1)。
- [官方仓库](https://github.com/WeiboAI/VibeThinker)树
  `5158cc982d1d8dbe68f2447c95aa4b32c7769d21`，26 文件且未截断，主要为 PDF、评测与图片。
  HF 作者数据集查询返回空列表；这是本次查找范围内未找到，不证明其他渠道绝无发布。
- 基于 Qwen2.5-Coder-3B；作者明确未训练工具调用/Agent 编程，聚焦可验证数学/代码/STEM。
  报告讨论 64K RL 与早期截断的影响，适合作为长度专题；未见正式顶会身份核验。

### S17. Nanbeige4.2-3B

- 已读 [模型卡](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)与
  [2607.22083v1](https://arxiv.org/html/2607.22083v1)：循环 Transformer，3B 为非 embedding
  口径，卡列总参数约 4B。架构比较需同时考虑循环计算成本。
- 报告说明 SFT、RLHF、推理 RL、Agent RL，轨迹/回合过滤和 rubric 评测，含部分自有评测框架。
- 已核对 Base/Instruct 身份与部署代码入口；HF 作者数据集列表有 ToolMind 等，但没有
  证据证明它们等于 4.2 全部训练语料。完整生产训练脚本/数据仍未确认，按技术报告引用。

### S18. OpenThoughts

- [官方仓库](https://github.com/open-thoughts/open-thoughts)树
  `e86cf2dbbe69ae178aec302be33a3454a7183457`，3905 文件；存在数据生成、过滤、污染检查与实验配置。
- 主 README 和数据入口已查，选定时必须固定一代版本；本轮只列补充项，不将数学推理
  的可验证收益直接解释为测试分析的可用效果。

## 6. 本轮检索限制与明早更新

### 用户追问后的独立复核（2026-11-05）

实际使用两个 gpt-6.1-sol 子代理分别审查方法专题与完整项目。它们提供辅助证据，
不以多数投票替代主助手判断。未改源码或运行训练。

- 主助手再次核对 ICLR 全文表 1：LoRA/Prompt 的数据规模为 8K 至 100K，FMT 各任务
  的范围更大，部分边界重合，不能说完全无重叠，也不是完整交叉、等预算对照。
- 主助手再次核对 Rectified 式 3、附录 D.2/E：按目标 token 平均 loss，tokenizer/vocab
  随模型不同；重复三个 seed 不能消除跨 tokenizer 指标单位差异。论文方法可以研究，
  不直接使用其原始 loss 排名替团队选型。
- 完整项目复核进一步读取了 Tülu 的数据样本（`id/messages/source`）及 8B SFT/DPO
  阶段模型卡，发现指标有升有降；Puro 样本、保留启动配置和预算分支可用于核对
  数据身份/训练量，但不能替代领域 SFT 或用户体验评估。
- 复核入口：[Tülu 数据](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)、
  [Tülu DPO 阶段卡](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)、
  [Puro 保留启动配置](https://github.com/thu-pacman/Puro-Megatron/blob/a7b80e873a0b5e1820ae425b9abd1b9ec578dc5c/examples/puro/run_puro_2b.sh)。
- 结果用于明确短期学习任务，不宣称对全部候选进行了相同深度的完整审计。

### 获取与排程限制

网页检索工具多次返回空内容，本轮通过官方 HTTP 页面、arXiv 全文、GitHub API 和 HF
元数据核查。GitHub 有部分 403/限流，个别 raw 请求超时，已记录成功替代来源；不使用
第三方宣传补齐失败访问。仓库树按当次 HEAD 记录，不等于所有实验所用的历史版本。

用户要求明早更新。2026-11-05 调用自动化接口返回“工具不再可用”，未创建定时任务。
拟复核时点为北京时间 2026-11-06 08:30，尚未自动排程；无后台执行或隔夜完成保证。
届时重点复核 ZGCM 数据权限、ICLR 研究代码开放、IFM 报告/原训练身份和其他新增材料。
