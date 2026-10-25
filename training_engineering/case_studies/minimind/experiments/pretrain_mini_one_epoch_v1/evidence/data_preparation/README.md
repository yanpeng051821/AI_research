# MiniMind mini：正式数据准备检查

> 2026-10-24 的数据准备证据。最新冻结数据元信息见 [data_recipe](../../data_recipe/README.md)，下文保留构造前的来源与长度分析。

## 1. 完整来源

固定来源仍为 jingyaogong/minimind_dataset 的 pretrain_t2t_mini.jsonl，revision 为 312afb4f76391145c6902f765bb51691c09a12f5。本轮重新核验原文件 SHA256 与既有 manifest 一致。

| 检查项 | 实际结果 |
| --- | ---: |
| 文件字节数 | 1,241,043,656 |
| 物理行数 / 结构通过记录 | 1,270,238 / 1,270,238 |
| 结构无效记录 | 0 |
| 记录字段 | 全部只有 text |
| 不同的完整正文 | 1,270,047 |
| 完全重复正文组 / 额外记录 | 191 / 191 |
| 最大完全重复组大小 | 2 |

完整报告见 [source_inspection.json](source_inspection.json)。身份比较使用原始 text 的 UTF-8 SHA256，不做空白或 Unicode 归一化。结构通过不表示事实正确；完全重复检查不能推断原始文档归属、近似重复或截断后相同前缀。

## 2. 固定样本的长度对比

沿用之前随机抽取的 10,000 条候选，重新使用本地已核验 tokenizer 编码，正文总数为 2,584,015 tokens。本表只代表这份样本，不是完整来源的 token 统计。

| 最大长度 | 正文容量 | 完整容纳记录 | 需要截尾记录 | 截尾损失正文 tokens | 定长位置数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 510 | 9,410，94.10% | 590 | 321,373，12.44% | 5,120,000 |
| 768 | 766 | 9,632，96.32% | 368 | 209,414，8.10% | 7,680,000 |
| 1,024 | 1,022 | 9,664，96.64% | 336 | 118,886，4.60% | 10,240,000 |

完整字段和脚本身份见 [length_comparison.json](length_comparison.json)。此处假设保留全部样本并截尾，不是排除整条超长记录。768 比 512 多保留 111,959 个正文 tokens，约增加 4.95%，定长位置数增加 50%。位置数量不能直接等同于 GPU 耗时；Attention 还受到序列长度与实现影响。

样本中未发现正文编码含 pad ID 的记录，不代表完整来源具有相同结论。正式构造仍需检查编码后输入重合和特殊 token 条件。

## 3. 对正式切分的影响

旧 512 切分继续用于工程验证，不能充当完整 mini 训练切分。正式构造保留原 text，由 Dataset 按已选长度截断、添加 BOS/EOS；训练/验证的正文内容身份与编码规则分别记录。768 保持候选，训练预算与最终长度尚未冻结。

本次来源没有原始文档或对话 ID，至少按完整正文 SHA256 建立组，组内记录必须分配到同一集合。当前候选方案是保留组内重复记录以减少对官方数据分布的额外改变，而不是直接删除 191 条；正式 manifest 明确该策略。训练集与验证集无完全相同正文，不等于原始文档级严格独立。

验证集拟先从符合约束的不同正文组中固定抽取 5,000 组，seed=43；若抽到重复组，实际记录数可能多于 5,000。这个规模是预算讨论的候选值，并非普遍最优比例，最终还要核验有效 token 数和评测耗时。已有人审约束、重复组约束与编码后重合检查需要一起进入构造流程，不能只按行号随机分配。

## 4. 复练与协助范围

在 implementation 目录执行，输出使用新的文件名以保留已有产物：

~~~powershell
uv run --locked python scripts/compare_pretrain_lengths.py --verify-source --output data/processed/pretrain_candidate_v1/analysis/length_comparison_recheck.json
uv run --locked python scripts/inspect_pretrain_source.py --output data/processed/pretrain_mini_v1/source_inspection_recheck.json
~~~

本轮比较脚本与完整来源检查由助手承担，使用学习者已有 reader、validator 与 tokenizer。旧统计和切分产物保持不变。学习者随后完成正式构造的核心函数；助手按后续委托补齐身份门禁、发布入口、全量编码审计并执行，协助范围与结果见 [data_recipe](../../data_recipe/README.md)。实践入口见[第七周教程](../../../../practice_notes/week_07_pretrain_experiment.md)。
