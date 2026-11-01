# 768 长度候选筛选

## 1. 当前结论

用户已确认本轮 SFT 长度采用 768，并对超限记录整条排除，不截断。筛选已执行，原始文件不变；保留的是长度合格候选，不是最终训练集，也没有划分训练/验证集或冻结配方。

| 输出 | 记录数 | 有效助手目标 | 字节数 |
| --- | ---: | ---: | ---: |
| 长度合格候选 | 856,531 | 340,351,475 | 1,583,975,632 |
| 超长排除记录 | 49,187 | 36,978,625 | 155,225,538 |
| 合计 | 905,718 | 377,330,100 | 1,739,201,170 |

超长排除记录另存，便于复查和以后改变策略时重新构造；它们不是坏数据。这里的长度是当前模板、tokenizer 和思考/工具转换之后的完整输入 token 数，768 本身保留，不是只计算助手正文。

## 2. 实现与复跑

入口为 [prepare_sft_length_subset.py](../../../implementation/scripts/prepare_sft_length_subset.py)。它复用[全量编码预检](../full_encoding_preflight_v1/README.md)的逐条长度，而不是重新定义一套长度口径：

1. 检查编码器、validator、预检辅助代码及 tokenizer 身份与报告一致，完整候选无编码诊断标记。
2. 同时遍历原始 JSONL 和编码明细，逐条核对来源行号及原始行 hash。
3. 按 `input_tokens <= 768` 分流，原样写出整行字节；保留原有相对顺序，不改对话、名字、思考、工具字段或标签规则。
4. 核对完整来源与明细的 hash、数量后，将临时文件改为最终文件名；再次读取输出核对 hash，生成完成报告。

若运行失败，不应把孤立的输出或 `.partial` 文件视为验收通过。只有完成报告、输入身份及输出 hash 一致，才使用这次产物。入口拒绝覆盖已有目录；重跑使用新版本路径。

在 `implementation` 目录，用 PowerShell 复跑以下命令；下面使用新路径，避免覆盖本次产物：

```powershell
uv run --locked python scripts/prepare_sft_length_subset.py `
  --source data/raw/minimind/sft_t2t_mini.jsonl `
  --encoding-report ../experiments/sft_mini_data_inspection_v1/full_encoding_preflight_v1/report.json `
  --output-dir data/processed/sft_length_768_rerun `
  --report-dir ../experiments/sft_mini_data_inspection_v1/length_selection_768_rerun `
  --max-length 768
```

## 3. 文件与身份

大文件位于忽略目录 `implementation/data/processed/sft_length_768_v1/`：

- `length_eligible.jsonl`：SHA256 `45262dc172255c8eee953a28cdaae1998607969fc0b1355a8498d6925a15e7ad`。
- `overlength_excluded.jsonl`：SHA256 `aa0c380f6c3eac69404320d1dd8802f4215036c6d5928d0c8fbafd6e7c87cce5`。

机器可读的输入、代码、编码路径与输出身份见 [report.json](report.json)。输入编码路径发生变化时，先重新预检，再生成新候选；不能把旧长度明细直接用于新编码规则。

## 4. 验证与边界

4 项定向测试覆盖 768 包含边界、记录字节与顺序不变、错误身份不产生最终命名文件、明细错位与拒绝覆盖。实际产物的记录、目标与字节数合计和来源一致，数量与此前模拟分析一致，输出完整 hash 已重新读取核对。

本次没有应用内容决定、身份修订或去重。完整来源中的重复量不能直接从 856,531 中扣除，必须在实际保留范围和后续修订版本上重新计算；内容与协议检查、隔离切分以及最终派生数据编码验收仍需继续。

下游进展：之后已在本次长度候选上完成[精确去重与来源追溯](../deduplication_768_v1/README.md)，保留 809,713 条；这是新的派生版本，不覆盖本文记录的两个长度筛选文件。
