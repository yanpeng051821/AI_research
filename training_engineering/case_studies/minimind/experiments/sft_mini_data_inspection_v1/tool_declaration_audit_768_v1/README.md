# 工具声明与调用检查

## 1. 当前结论

对精确去重后的 809,713 条候选进行完整扫描，检查结构化函数调用是否出现在该对话的工具声明中。本次只记录问题，不执行排除。

| 指标 | 结果 |
| --- | ---: |
| 检查记录 | 809,713 |
| 含非空声明、调用或 tool 消息的记录 | 47,584 |
| 结构化函数调用总数 | 50,820 |
| 含未声明函数调用的记录 | 1,199 |
| 字段适配错误记录 | 0 |
| 工具声明名称重复记录 | 0 |

这里的 1,199 是有至少一次问题调用的记录数，不是调用次数。其余记录无此标记，不代表工具协议或内容全部合格。

## 2. 判法与实现

入口为 [audit_sft_tool_declarations.py](../../../implementation/scripts/audit_sft_tool_declarations.py)：

1. 从去重报告取得实际候选和逐条来源对应表，经共用的 [sft_candidate_lineage.py](../../../implementation/scripts/sft_candidate_lineage.py)核对完整文件身份、行号和原行 hash。
2. 调用已有 `prepare_sft_messages`，复用同一套 tools/tool_calls 字符串解析和字段形态检查，不修改原记录。
3. 收集声明的函数名，对每个结构化 `assistant.tool_calls` 检查名称成员关系；另记录声明名重复。
4. 对问题记录写入候选行号、原始来源行号、原行 hash、消息及调用索引、调用名和声明名。没有把完整语料或工具结果复制进小型 Git 报告。

例如来源行 771,194 只声明 `web_search`，却调用 `get_current_weather`。已有校准案例 776,360 也出现在本次问题清单中。函数名按原值精确匹配，不通过改名、补造工具声明或实际执行工具来让记录通过。

本次不检查完整参数 JSON Schema、调用 ID 与工具结果的对应、工具执行结果真实性、最终回答质量，或仅嵌在正文里的工具调用文本。这是完整协议审核中的一个具体检查，不是全协议验收。

## 3. 复跑与证据

在 `implementation` 目录使用 PowerShell，新输出路径避免覆盖已有运行：

```powershell
uv run --locked python scripts/audit_sft_tool_declarations.py `
  --dedup-report ../experiments/sft_mini_data_inspection_v1/deduplication_768_v1/report.json `
  --details-dir data/audit/sft_tool_declarations_768_rerun `
  --report-dir ../experiments/sft_mini_data_inspection_v1/tool_declaration_audit_768_rerun
```

[report.json](report.json)保存输入与代码身份、计数和前 10 条问题例子；完整 1,199 条问题定位在忽略目录 `implementation/data/audit/sft_tool_declarations_768_v1/issues.jsonl`，其 hash 同样入报告。

6 项定向测试验证声明内调用、未声明或无声明调用、纯文本提及不误算调用、合法空助手正文、重复声明名以及字段适配错误。另一次独立只读核对直接解析原始字段而不调用适配函数，重新计算全部 809,713 条的声明/调用名差集，与这 1,199 条问题定位完全一致。

## 4. 与人工决定的衔接

用户随后只确认排除已讨论的来源行 731,011 和 776,360，执行见[确认决定的应用](../confirmed_review_768_v1/README.md)。其中只有 776,360 在本次工具标记中，因此新候选仍有 **1,198 条**带此工具问题的记录，尚未批量排除。

后续用户已确认统一规则，剩余 1,198 条已在[新版本中隔离并复查](../tool_quarantine_768_v1/README.md)，保留 808,513 条，该项标记为 0。本文保留扫描时的完整计数，不回写成对新候选运行的报告。禁止把无标记等同质量全部通过，也不补造工具或整批排除工具样本。
