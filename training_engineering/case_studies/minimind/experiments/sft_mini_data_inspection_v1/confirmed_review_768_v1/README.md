# 用户确认审核决定的应用

## 1. 当前结论

用户于 2026-10-31 明确确认：在后续派生数据中整条排除原始来源行 731,011 与 776,360 的原样记录。助手已执行，只排除这两条，没有按关键词或新扫描标记扩大范围。

| 产物 | 记录数 | 字节数 |
| --- | ---: | ---: |
| 输入：精确去重候选 | 809,713 | 1,539,525,786 |
| 应用决定后的候选 | 809,711 | 1,539,522,945 |
| 本次审核排除记录 | 2 | 2,841 |
| 新来源对应表 | 809,713 | 167,218,129 |

原始文件和上一步候选均不覆盖。剩余记录仍未全部接受；没有修改身份、补写回答、切分数据或冻结配方。本次不重新编码，不能把输入的目标总数直接当作新候选的目标总数。

## 2. 决定与精确匹配

[decisions.confirmed.jsonl](../calibration_v1/decisions.confirmed.jsonl)保存用户确认的动作、理由、原始文件 hash、原始行号及原行 hash。旧 [decisions.assistant.jsonl](../calibration_v1/decisions.assistant.jsonl)保留最初建议，不覆盖历史，也不作为本入口可执行的确认记录。

| 原始来源行 | 输入候选行 | 原样排除理由 |
| ---: | ---: | --- |
| 731,011 | 678,855 | 没有提供代码，助手却编造程序和执行结果 |
| 776,360 | 721,544 | 调用的 `get_current_weather` 不在本条声明工具中 |

决定不是新训练样本，也不是“看到下面代码就删除”的规则。执行匹配必须同时满足原始文件身份、原始行号和原行 hash；发现记录已变化、决定未确认、重复决定或决定未命中当前候选时拒绝，不能静默略过。

## 3. 实现与复跑

入口为 [apply_sft_review_decisions.py](../../../implementation/scripts/apply_sft_review_decisions.py)。它读取实际去重候选和来源对应表，对每条记录判断是否命中确认决定：命中则原样另存，否则原样保留。新对应表记录新输出行号以及 `keep`/`exclude_review` 动作，避免排除两条后后续行号失去追溯。

全部输入与对应表 hash、数量核验完成、所有决定精确命中后才发布最终命名文件；输出 hash 再次读取核对。失败或只有孤立文件时不视为完整产物。输出目录必须是新目录。

在 `implementation` 目录使用 PowerShell 复跑：

```powershell
uv run --locked python scripts/apply_sft_review_decisions.py `
  --dedup-report ../experiments/sft_mini_data_inspection_v1/deduplication_768_v1/report.json `
  --decisions ../experiments/sft_mini_data_inspection_v1/calibration_v1/decisions.confirmed.jsonl `
  --output-dir data/processed/sft_reviewed_768_rerun `
  --report-dir ../experiments/sft_mini_data_inspection_v1/confirmed_review_768_rerun
```

## 4. 文件与验证

大文件位于忽略目录 `implementation/data/processed/sft_reviewed_768_v1/`：

| 文件 | SHA256 |
| --- | --- |
| `reviewed_candidate.jsonl` | `cd203f2996197e67dc476efa7a2167bc31029141d9fc47341a4eb90c70c2a593` |
| `review_excluded.jsonl` | `76de8735479bd2b909a49bceb58a1a68dbdc22466f643d1aed08bdbb54f2c2aa` |
| `lineage.jsonl` | `4fcb370d81679d4184722dcffeb9bec49283d02e71f253d3b9396d5d14a242d6` |

[report.json](report.json)保存决定文件、输入、代码与输出身份，以及两条具体命中记录。8 项定向测试覆盖精确排除及新对应表可消费、未确认或错误来源决定拒绝、不同内容 hash、未命中、重复决定和对应表损坏。工具检查与决定应用共新增 14 项测试，全项目 442 项通过（77.66 秒）。

助手另以独立只读循环逐条比较输入、两份输出和新对应表，确认全部 809,713 条输入只按这两条决定分流，其余记录的字节和顺序不变，原行身份、保留输出位置与完整文件 hash 均一致。与此同时，独立重新解析工具字段，核对工具问题清单。

在本次仅应用两条具体决定后，候选仍含 1,198 条未声明调用标记。随后用户确认统一规则，已在[下一派生版本中隔离](../tool_quarantine_768_v1/README.md)，保留 808,513 条。其他内容审核、身份修改、完整协议检查、最终分组切分及处理后的编码验收仍需继续；这不是可直接训练的最终接受集。
