# Pretrain 512：数据切分快照

> 2026-10-21。只完成数据准备与切分，尚未执行本次真实语料训练。模型规模、优化器计划、训练步数和运行预算仍待确定。

## 1. 数据范围

固定来源为 jingyaogong/minimind_dataset，revision 为 312afb4f76391145c6902f765bb51691c09a12f5，文件为 pretrain_t2t_mini.jsonl。从全文件按 seed=42 选取 10,000 条候选；正文大于 510 tokens 的记录整条过滤，另应用 12 条内容复核决定。正文保持原样。

| 分组 | 记录数 | 正文 tokens | shift 后有效目标 tokens |
| --- | ---: | ---: | ---: |
| 训练 | 8,907 | 1,860,620 | 1,869,527 |
| 验证 | 500 | 100,545 | 101,045 |
| 排除 | 593 | 不参与训练 | 不参与验证 |

593 条排除记录包括 590 条超长记录和 3 条人工排除记录。本次无两者重叠。切分 seed=43，从未参与内容讨论的 9,398 条可用记录中抽取 500 条验证；9 条已审保留记录固定用于训练。输出按来源行号排序，训练时由 sampler 决定消费顺序。

## 2. 文件与重建

本目录保存适合 Git 跟踪的少量元数据：

- data_manifest.json：输入、tokenizer、构造脚本与输出身份，以及切分规则和统计。
- candidate_manifest.json：候选来源与抽样记录的字段快照。
- analysis_report.json：过滤前长度分布和完全重复统计的字段快照。
- review_decisions.jsonl：12 条人工初筛后的复核决定，不包含候选正文。

真实正文保存在 implementation/data/processed/pretrain_512_v1/ 下的 train.jsonl、validation.jsonl；exclusions.jsonl 保存排除原因与来源。数据文件位于 Git 忽略目录，不随本目录自动同步。

已有候选、分析与审查产物时，在 implementation 目录运行：

~~~powershell
uv run --locked python scripts/build_pretrain_splits.py
~~~

脚本拒绝已有输出目录，防止覆盖已记录数据。复练可指定另一个未创建的目录：

~~~powershell
uv run --locked python scripts/build_pretrain_splits.py --output-dir data/processed/pretrain_512_rebuild
~~~

新机器先按[第七周教程](../../practice_notes/week_07_pretrain_experiment.md)恢复固定来源，再生成候选和分析产物。审查决定可从本目录恢复到候选目录的 analysis/review_decisions.jsonl；它是人工结果，不由抽样脚本自动重建。源 manifest 与分析报告可由对应脚本生成，本目录文件供核对字段。哈希对应原产物实际字节，不能将换行或序列化格式不同的字段快照直接视为字节相同的输入。

## 3. 核验与限制

已经独立核验分组互不交叉且完整覆盖全部候选、正文与原记录相同、审查约束生效、输出哈希正确。相同输入和当前环境在另一个目录重建，四份输出逐字节相同。现有 Dataset/collator 实际遍历验证集后，shift 后有效 token 数为 101,045，与 manifest 一致。

候选整体没有正文完全重复，未保证近似重复或原始文档/对话严格隔离。语料存在质量风险且未逐条事实核实，验证 NLL 不能作为事实能力证明。当前仅固定数据切分；训练入口尚需配置设备、512 长度、学习率计划及周期保存，不能直接沿用第六周的 64 长度 smoke 配置开始正式实验。
