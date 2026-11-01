# 第二版 SFT 数据准备与验收

> 本页保留第二版时点的候选状态、数量和命令，不代表最新状态。最终修订与冻结见[第三版报告](../preparation_768_v3/README.md)。本目录另存准备器、验证器、策略和候选合同的 `.snapshot` 文件，便于检查历史身份；旧合同复验需要匹配当时的运行代码，不能直接与已升级的当前入口混用。

## 1. 当前结论

从固定的 808,513 条本轮输入生成 787,423 条派生候选，完成全量编码、来源追溯、有限身份修订、修订后去重和分组切分。工程独立复验通过，**内容接受与最终冻结仍待确认，没有训练**。

| 去向 | 数量 |
| --- | ---: |
| 保留原文 | 787,409 |
| 修订身份后保留 | 14 |
| 修订后重复另存 | 15 |
| 暂存复核 | 21,075 |
| 输入合计 | 808,513 |

暂存占本轮输入约 2.61%，由协议问题 226 条、上游名称语境 20,417 条、疑似第三方自述 423 条和固定内容复核 9 条组成。后两类名称检索可能包含正常引用，暂存不是判错；也不能把保留率解释为内容正确率。

策略见[processing_policy.json](../../sft_mini_v1/data_recipe/processing_policy.json)，可读合同见[DATA_CONTRACT.md](../../sft_mini_v1/data_recipe/DATA_CONTRACT.md)，实际数值和身份见[准备报告](report.json)与[机器合同](../../sft_mini_v1/data_recipe/DATA_CONTRACT.candidate.json)。

## 2. 实现与处理边界

### 2.1 格式和工具协议

新检查复用当前字段适配后消息，对完整序列、工具声明和参数 Schema 做全量核查。共发现 51 条 unexpected_assistant、175 条 conversation_not_finished_by_assistant、2 条 invalid_tool_arguments、24 条 unfinished_tool_exchange；类别有重叠，合并为 226 条协议暂存，不能直接把四项相加。

Schema 使用 jsonschema 4.26.0，依赖及传递依赖已由 uv.lock 固定；不运行数据中的工具、表达式或外部引用。没有 ID 的合并结果只检查块级顺序，仍记录语义对应限制。报告的 notices 统计检查过的输入记录，不应误读为最终训练集中真实调用的数量。

### 2.2 身份和内容

自动修订只面向两条消息组成的单轮、已知前十固定介绍与明确身份问句清单的交集。修改助手正文并清空该轮旧思考，不改用户原文，也不把模型名全改成训练者姓名。修订后再精确去重，最终仅 14 条修订样本；不保证模型训练后一定稳定遵循这些自述，仍须生成评测。

没有为答非所问的数据生成替代答案。其他上游语境与疑似第三方自述整条暂存，保留复查可能；另按[固定 24 条内容复核](../quality_review_v1/README.md)暂存 9 条具体记录。法律、医疗等未核实事实不作通过结论，也不扩展为按领域或关键词批删。

### 2.3 监督和读取

编码器拒绝源字段中的 BOS/EOS 字面边界，防止用户正文伪造助手区域；编码器和 collator 均检查 shift 后有效目标。保留记录重新完整编码，不截断，长度均不超过 768。

第一版实际 Dataset 加载因 Arrow 推断前段普通消息 schema 而拒绝后段工具字段，详见[故障记录](../preparation_768_v1/README.md)。本版用 JSONL 行偏移读取，保留异构可选字段，worker 流按 PID 独立打开，pickle 不携带文件句柄。这是读取方式修复，不是删除工具字段规避错误。

## 3. 切分结果

归一化 user 正文或完整对话共享的记录组成连通组，只分组不改原文。共 659,186 组，最大 30,195 条；开发来源 1,197 行所在组件强制训练。固定 seed=20261031，按组来源 hash 顺序装入 2,048 条验证预算，验证包含 1,868 个完整组。

| 指标 | train | validation |
| --- | ---: | ---: |
| 记录 | 785,375 | 2,048 |
| plain 形态 | 516,814 | 1,413 |
| reasoning_only 形态 | 220,400 | 519 |
| tool_only 形态 | 48,161 | 116 |
| 输入长度 P50 / P95 / P99 | 492 / 708 / 754 | 479 / 700 / 750 |
| 最大输入长度 | 768 | 767 |
| 输入 token 总数（未 padding） | 377,716,766 | 959,968 |
| shift 后有效助手目标 | 326,248,895 | 833,052 |

形态沿用扫描器的原始字段/标记分类，非专业领域标签，也不是“真实调用了几次工具”的数量；序列化空工具字段也可能触发形态标记。验证使用整组预算，不是记录级等概率抽样，不据此估计全量数据质量率。归一化精确问题隔离不等于所有语义改写、原文来源或 benchmark 污染均被排除；validation 不是最终盲测，未建立 test。

## 4. 产物地图

大文件统一位于 `implementation/data/processed/sft_prepared_768_v2/`，内容与身份见 report.json 的 outputs：

| 文件 | 用途 |
| --- | --- |
| accepted.jsonl | 切分前派生候选，包含 14 条有限修订；不是“内容全通过”的标签 |
| train.jsonl / validation.jsonl | 按组切分的实际输入 |
| review_hold.jsonl | 可逆暂存原样记录，不作为新训练示范 |
| postrewrite_duplicates.jsonl | 身份修订后重复，不叠加到训练集 |
| lineage.jsonl | 全部 808,513 条输入的动作、原始源行/hash、理由和输出内容 hash |
| encoding_index.jsonl | 每条保留数据的长度、目标数及 input_ids/labels 指纹 |
| split_index.jsonl | 每条数据的组 ID、split 和内容 hash |
| quality_review_sample.jsonl | 本版另留的固定抽样，不再用于反复调规则到“全部通过” |

原始来源和第一版数据仍在原路径，本轮没有删除它们；本报告不是这些大文件的备份，也未将语料上传 Git/HF。

## 5. 复跑与独立复验

需先完成此前输入、身份分组和固定抽查；代码不能代替缺失的数据文件。以下命令是本版实际准备命令。已有输出目录与合同不能覆盖，复跑需给 output/report/contract 换版本名：

```powershell
uv run --locked python scripts/prepare_sft_recipe.py `
  --candidate-report ../experiments/sft_mini_data_inspection_v1/tool_quarantine_768_v1/report.json `
  --policy ../experiments/sft_mini_v1/data_recipe/processing_policy.json `
  --groups-report ../experiments/sft_mini_data_inspection_v1/identity_template_groups_768_v1/report.json `
  --development-samples `
    data/processed/sft_candidate_inspection_v1/random_sample.jsonl `
    data/processed/sft_candidate_inspection_v1/plain_examples.jsonl `
    data/processed/sft_candidate_inspection_v1/reasoning_only_examples.jsonl `
    data/processed/sft_candidate_inspection_v1/tool_only_examples.jsonl `
    data/audit/sft_identity_inventory_768_v1/samples.jsonl `
    data/audit/sft_identity_groups_768_v1/top_group_contexts.jsonl `
    data/processed/sft_prepared_768_v1/quality_review_sample.jsonl `
  --output-dir data/processed/sft_prepared_768_v2 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/preparation_768_v2 `
  --contract ../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.candidate.json `
  --workers 4 --chunk-size 256
```

生成约 327.54 秒。脚本自动生成合同主体；首次独立验证约 113.78 秒通过后，在合同中补记已核对的特殊 token/词表信息和独立验收状态，再重新复验最终候选文件。最终证据为[verification.final.json](verification.final.json)，绑定当前合同 SHA256 `6932af6465ef4bcc36e78a4dd7ebc1bc63d5bca0f9882a92eddfee2720012d73`，不是仅验证补记前的旧文件。

```powershell
uv run --locked python scripts/verify_sft_recipe.py `
  --contract ../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.candidate.json `
  --output ../experiments/sft_mini_data_inspection_v1/preparation_768_v2/verification_rerun.json `
  --load-datasets
```

最终独立核验约 101.68 秒，全部文件 hash/行数/字节数、全部输入分流和修订范围通过；修订后精确对话重复为 0，归一化 user 正文、完整对话及组件跨集重叠为 0，开发记录进入验证为 0。全部 2,048 条验证重新编码与原指纹一致。真实 Dataset 加载训练 785,375 条、验证 2,048 条，各取首/中/尾核对编码并得到 `[3,768]` batch；不是只对一个模拟 Dataset 做测试。

协议、边界、配方 CLI 和 JSONL 读取相关定向测试 31 项通过；全项目 479 passed（133.42 秒）。独立验证器还拒绝 `python -O`，防止关闭断言绕过本工具检查；CLI 测试包含篡改输入后拒绝、非空输出拒绝和分组保护。

## 6. 接受边界

工程检查已形成可核验候选，但正式内容接受不能自动勾选。需要用户查看：是否接受保守暂存 21,075 条（包括可能正常的名称引用），以及本轮以有限语义审核开展教学实践的风险。未声明全量事实正确、工具实际执行成功、匿名结果全语义对应、语义近重复全清除或发布许可完全解决。

因此 DATA_CONTRACT.candidate.json 保持 `candidate_pending_review_not_frozen`、`ready_for_training=false`，没有生成最终冻结合同、训练配方或启动 SFT。统一进度以[进度表](../REVIEW_CRITERIA.md)为准。
