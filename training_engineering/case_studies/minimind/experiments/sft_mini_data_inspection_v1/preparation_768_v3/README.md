# 第三版 SFT 数据修订与冻结

## 1. 交付结果

本轮在第二版基础上，仅将音乐社团与天气答复两条具体暂存记录修订后恢复；其他保守暂存规则不变。固定输入仍为 808,513 条，正式数据共 787,425 条。数据配方的冻结不包含训练授权，也不表示全量事实正确。

| 输入去向 | 数量 |
| --- | ---: |
| 保留原文 | 787,409 |
| 身份修订后保留 | 14 |
| 内容修订后保留 | 2 |
| 修订后重复另存 | 15 |
| 原样暂存复核 | 21,073 |
| 输入合计 | 808,513 |

暂存由协议问题 226 条、上游名称语境 20,417 条、疑似第三方自述 423 条和具体内容问题 7 条组成。名称暂存可能包含有效引用，不等于全部判错。相比第二版，训练增加 2 条、150 个有效助手目标；验证集内容 SHA256 完全不变。

正式约定见[合同说明](../../sft_mini_v1/data_recipe/DATA_CONTRACT.md)、[机器合同](../../sft_mini_v1/data_recipe/DATA_CONTRACT.json)和[冻结标记](../../sft_mini_v1/data_recipe/data_freeze.json)。旧版报告、数据和候选合同未覆盖。

## 2. 判法与修订

### 2.1 处理依据

先看输入是否足以支持正确答案，再决定能否修订。缺少代码却要求程序输出，不能靠编造代码补齐；虚构个人经历还伴随错误思考，不能只换一个名字。已提供的问题与工具结果足以支持简洁答复时，可以精确修正错误解释或无依据推断。

[典型案例与两条完整修订文本](../quality_review_v1/README.md#5-典型判法与最终修订)记录了原问题、处理理由和委托范围。原始审核建议继续保留作历史证据，不反写成“当时已经通过”。

### 2.2 实现约束

[processing_policy.v3.json](../../sft_mini_v1/data_recipe/processing_policy.v3.json)为两条修订分别指定原始行 SHA256、原助手消息 SHA256、消息索引和替换文本。prepare_sft_recipe.py 的 content_revision 先核对这些身份，再复制记录并只替换指定助手消息。

音乐题删除错误解释与旧思考；天气题只改最后答复，保留声明、调用和工具结果。其余记录不生成新的实质性答案。原始身份不匹配会让准备任务失败，不静默跳过指定修订；所有指定修订也必须在输入中成功执行。随后使用真实编码器重新检查协议、长度和有效目标，再去重和切分。

## 3. 完整链路

```text
固定候选 + 来源对应表 + 精确策略 + 开发样例
  -> 原协议检查
  -> 精确内容修订 / 有限身份修订 / 原样暂存
  -> 完整模板编码、长度及有效目标检查
  -> 修订后完整对话去重
  -> 共享归一化问题/对话的连通分组
  -> train / validation / 索引 / 来源动作
  -> 自动生成候选合同
  -> 接受记录 + 最终合同字节独立复验
  -> 正式合同 + 最后写入 data_freeze.json
```

准备器负责生成数据与候选，不自行批准冻结；独立验证器重新检查全部分流、文件和编码；冻结器在接受记录与候选精确对应后，验证最终合同字节，发布合同，再写提交标记。失败时没有有效冻结标记，不能只看一个 pending 文件就认定成功。

| 文件 | 职责 |
| --- | --- |
| [prepare_sft_recipe.py](../../../implementation/scripts/prepare_sft_recipe.py) | 应用策略、编码、去重、切分并自动生成候选与统计 |
| [verify_sft_recipe.py](../../../implementation/scripts/verify_sft_recipe.py) | 核对全部输入动作与输出、修订边界、隔离及实际读取 |
| [freeze_sft_data_recipe.py](../../../implementation/scripts/freeze_sft_data_recipe.py) | 接受记录绑定、最终复验、冻结提交及只读身份检查 |
| [acceptance.json](../../sft_mini_v1/data_recipe/acceptance.json) | 记录用户委托、助手执行范围和本轮接受的限制 |

## 4. 切分与监督

| 指标 | train | validation |
| --- | ---: | ---: |
| 记录 | 785,377 | 2,048 |
| plain / reasoning_only / tool_only | 516,815 / 220,400 / 48,162 | 1,413 / 519 / 116 |
| 长度 P50 / P95 / P99 | 492 / 708 / 754 | 479 / 700 / 750 |
| 最大长度 | 768 | 767 |
| 未 padding 输入 token | 377,717,358 | 959,968 |
| shift 后有效助手目标 | 326,249,045 | 833,052 |

共 659,187 个连通组，最大 30,195 条，验证包含 1,868 个完整组。固定 seed 为 20261031；已用于开发的 1,197 个来源行所在整组强制进入 train，包括本次两条修订。验证预算不是记录级等概率抽样；形态标记不是领域标签，tool_only 也不等于实际工具调用次数。

沿用固定 6,400 词 tokenizer、完整模板、不截断和右 padding 768。仅助手正文、保留的思考/调用及完整结束序列参与监督，角色头、用户、工具结果和 padding 不作为这些位置自身的目标；有效目标按 causal shift 后计数。具体字段以合同为准，切分 seed 不是未来训练 sampler seed。

## 5. 产物与命令

### 5.1 目录职责

大文件在 implementation/data/processed/sft_prepared_768_v3/，不上传 Git/HF。accepted 是切分前数据；train/validation 是实际输入；review_hold 和 postrewrite_duplicates 保留未进入训练的记录；lineage 记录全部输入的来源与动作；encoding_index 保存长度、目标数和编码指纹；split_index 保存组与切分身份。

本目录保存小型报告和复验证据；data_recipe 保存策略、接受记录及正式合同。索引和审核表不是额外训练样本，报告也不是大数据备份。

### 5.2 准备命令

在 implementation 目录运行。已有输出不可覆盖，复练应换 output/report/contract 路径，并重新检查版本：

```powershell
uv run --locked python scripts/prepare_sft_recipe.py `
  --candidate-report ../experiments/sft_mini_data_inspection_v1/tool_quarantine_768_v1/report.json `
  --policy ../experiments/sft_mini_v1/data_recipe/processing_policy.v3.json `
  --groups-report ../experiments/sft_mini_data_inspection_v1/identity_template_groups_768_v1/report.json `
  --development-samples `
    data/processed/sft_candidate_inspection_v1/random_sample.jsonl `
    data/processed/sft_candidate_inspection_v1/plain_examples.jsonl `
    data/processed/sft_candidate_inspection_v1/reasoning_only_examples.jsonl `
    data/processed/sft_candidate_inspection_v1/tool_only_examples.jsonl `
    data/audit/sft_identity_inventory_768_v1/samples.jsonl `
    data/audit/sft_identity_groups_768_v1/top_group_contexts.jsonl `
    data/processed/sft_prepared_768_v1/quality_review_sample.jsonl `
  --output-dir data/processed/sft_prepared_768_v3 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/preparation_768_v3 `
  --contract ../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.candidate.v3.json `
  --workers 4 --chunk-size 256
```

实际准备约 343.09 秒，[run_manifest.json](run_manifest.json)为 completed。模板、特殊 token 和词表信息直接由脚本读取并写入合同，不再人工补记候选后漏验最终身份。

### 5.3 冻结与复查

先审阅候选并生成与候选 hash 绑定的接受记录，再运行冻结。下面是本轮实际命令；不能以脚本默认值替代接受判断：

```powershell
uv run --locked python scripts/freeze_sft_data_recipe.py `
  --candidate ../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.candidate.v3.json `
  --acceptance ../experiments/sft_mini_v1/data_recipe/acceptance.json `
  --recipe-dir ../experiments/sft_mini_v1/data_recipe `
  --verification-output ../experiments/sft_mini_data_inspection_v1/preparation_768_v3/verification.frozen.json
```

冻结器不覆盖旧合同，先验证 pending 中的最终内容，通过后原字节重命名为正式合同。receipt 绑定正式合同 SHA256，不是仅绑定批准前的旧候选。最后写 data_freeze.json，任何失败都不能被当作完整冻结。

日后只查身份、不重做数据处理：

```powershell
uv run --locked python scripts/freeze_sft_data_recipe.py `
  --check ../experiments/sft_mini_v1/data_recipe/data_freeze.json
```

此门禁检查数据、合同、策略、证据、运行源码、tokenizer 资产清单及包版本是否仍匹配。它不会训练。后续 SFT 启动器还需要显式调用它；目前没有声称已实现训练自动检查。

## 6. 验证与边界

[独立复验](verification.frozen.json)约 111.92 秒通过，检查全部输入去向和写出字节、完整修订范围、修订后精确重复、归一化问题/对话和分组跨集重叠、开发样例隔离，并重新编码全部 2,048 条验证。真实 SFTDataset 分别加载训练与验证，各取首/中/尾与原 JSONL 编码对照，collator 得到 [3,768]。最终合同 SHA256 为 55bea98df2af3f6105fc03e06711b2602ba38e678b412720dcfff928e9f1da4d；冻结后只读门禁再次通过。

[自动测试记录](tests.json)：全项目 480 passed（143.67 秒），配方定向 4 passed。新增检查是为了证明精确修订不会改错上下文、冻结不能覆盖已有版本、编码约定或输入被改动会拒绝；测试数量不是内容质量分数。

本轮没有证明全量事实/推理正确、匿名工具结果语义逐一对应、所有语义近重复或外部污染已清除，也没有解决全部来源与公开分发许可。接受的是这些限制下的教学数据版本。未做的质量项目仍列在[统一进度表](../REVIEW_CRITERIA.md)，下一步讨论训练和评测配方，不自动开始短跑。
