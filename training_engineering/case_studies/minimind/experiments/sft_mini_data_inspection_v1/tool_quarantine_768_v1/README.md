# 未声明函数调用的规则隔离

## 1. 当前结论

用户已确认将剩余未声明函数调用记录整条隔离，不补造声明、不按工具名关键词删除普通对话。已执行 1,198 条排除，保留 808,513 条；输入和更早版本不覆盖。

| 项目 | 结果 |
| --- | ---: |
| 输入候选 | 809,711 条 |
| 本次隔离 | 1,198 条 |
| 保留并完整复查 | 808,513 条 |
| 处理后工具相关记录 | 46,385 条 |
| 处理后结构化调用 | 48,961 次 |
| 处理后声明检查标记 | 0 条 |

原清单 1,199 条中，来源行 776,360 已在上一步排除，本次核对其 hash 后不重复处理。“标记为 0”只说明这一项检查没有残留，不代表完整协议或内容质量全部通过。

## 2. 规则与实现

[确认规则](../review_rules/01_UNDECLARED_FUNCTION_CALLS.json)保存日期、问题类别、动作、原始来源 hash，以及固定的问题报告和候选报告 hash。它授权当前清单中的精确命中，不代表用户逐条人工读过 1,198 条，也不授权其他清洗条件。

[apply_sft_tool_quarantine.py](../../../implementation/scripts/apply_sft_tool_quarantine.py)依次执行：

1. 核对规则已确认、输入版本与规则一致、来源和去重依赖相同，以及完整问题清单 hash。
2. 只选择 `undeclared_function_call`，扣除经 hash 核验的既往排除项，其他问题类别不自动删除。
3. 将剩余项转成原始行号/hash 精确决定，复用已有执行器分流并生成新来源对应表；所有计划必须命中。
4. 对全部保留记录重跑工具检查，有残留标记就不生成成功报告；再核对输入、代码和输出身份。

保留和隔离记录都沿用原始行字节和相对顺序，不修改声明、回答或思考。审核清单与来源对应表不作为训练样本。

## 3. 复跑与文件

在 `implementation` 目录用 PowerShell 复跑，新路径避免覆盖已有产物：

```powershell
uv run --locked python scripts/apply_sft_tool_quarantine.py `
  --candidate-report ../experiments/sft_mini_data_inspection_v1/confirmed_review_768_v1/report.json `
  --audit-report ../experiments/sft_mini_data_inspection_v1/tool_declaration_audit_768_v1/report.json `
  --rule ../experiments/sft_mini_data_inspection_v1/review_rules/01_UNDECLARED_FUNCTION_CALLS.json `
  --output-dir data/processed/sft_tool_quarantine_768_rerun `
  --report-dir ../experiments/sft_mini_data_inspection_v1/tool_quarantine_768_rerun
```

大文件位于忽略目录 `implementation/data/processed/sft_tool_quarantine_768_v1/`：

| 文件 | 用途 |
| --- | --- |
| `reviewed_candidate.jsonl` | 808,513 条保留候选，1,537,731,790 字节 |
| `review_excluded.jsonl` | 1,198 条隔离记录，1,791,155 字节 |
| `lineage.jsonl` | 全部 809,711 条输入的动作与来源对应 |
| `rule_exclusions.jsonl` | 每条命中的原始来源、输入行号及理由 |
| `remaining_tool_issues.jsonl` | 处理后检查清单，本次为空 |

全部身份见 [report.json](report.json)。保留候选 SHA256 为 `3ce282774693db1c403a3afb984e36193c471303c4dc1f5d6231b2a02f7e44a6`，实际分流和重查约 33.30 秒。失败或只有孤立文件时，不视为完整产物。

## 4. 验证与后续

6 项新测试覆盖授权范围、既往排除不重复处理、未确认规则、来源依赖、问题清单及原行身份、重复标记和计数错误，以及不修改输入对象。全项目 448 项通过（80.97 秒）。

独立只读复核逐条比较全部输入、两份输出和来源对应：隔离项必须精确属于固定清单，其余记录原样保留；直接解析原始声明和调用，确认隔离项确有未声明调用，保留项没有这一问题，工具数量与调用数量重算一致。全部输入输出 hash 一致。

尚未实施身份改写，其他内容与完整协议检查、运行时门禁、分组切分和最终编码验收仍需继续，未冻结配方或启动训练。下一步先盘点身份自述，再审核和定向修订，不全局替换作者姓名。
