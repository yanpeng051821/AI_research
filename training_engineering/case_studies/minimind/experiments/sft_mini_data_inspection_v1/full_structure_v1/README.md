# 完整 Mini SFT 来源结构扫描

> 2026-10-29 完成。原始来源已下载并核验，全量结构已扫描；没有过滤、改名、套模板编码或启动训练。

## 1. 结果摘要

| 项目 | 实际结果 |
| --- | --- |
| 来源 | `jingyaogong/minimind_dataset`，revision `312afb4f76391145c6902f765bb51691c09a12f5` |
| 文件 | `sft_t2t_mini.jsonl`，1,739,201,170 字节 |
| SHA256 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`，与固定版本 API 的 LFS SHA256 一致 |
| 记录 | 905,718 行，全部可解析并通过现有 `validate_sft_record` 基本结构校验 |
| 消息 | 2,584,310 条：user 1,172,003、assistant 1,249,691、system 84,832、tool 77,784 |
| 抽样 | seed=20261029；全体有效记录的均匀蓄水池样本 1,000 条；每个形态组合最多另取 16 条 |

完整计数、文件身份、抽样方法与样本文件哈希见 [structure_report.json](structure_report.json)。结构通过不等于事实、行为、模板兼容性或整体数据质量已通过。

## 2. 内容形态

以下组合按原始字段和文本标记划分，互斥且合计为全部记录。思考标记指非空 reasoning_content 或正文中的 think 标签；工具标记指非空 tools、tool_calls 或 tool 角色。

| 组合 | 记录数 | 占比 |
| --- | --- | --- |
| 无思考/工具标记 | 547,017 | 60.396% |
| 有思考标记、无工具标记 | 273,869 | 30.238% |
| 有工具标记、无思考标记 | 84,832 | 9.366% |
| 同时有两类标记 | 0 | 0% |

这些不是人工核实的任务或质量标签。尤其“无思考标记”不等于套模板后没有空 think 段；本步骤没有套用模板，不能把源字段统计当作最终训练 token 统计。

原始记录中有 84,832 条包含 tools，76,574 条包含 tool_calls 和 tool 角色。所有非空 tools/tool_calls 字段在文件中都是字符串，解析后均为列表。两种工具标记计数不相同，因此不能只靠是否出现 tool 角色识别全部工具相关记录。

有 33,441 条助手消息的 content 为空，涉及 33,161 条记录。这只是需要结合工具调用等字段检查的信号，不是直接排除名单。观察样例中的源行 776,360 就有空 content 的助手工具调用，后续存在工具结果与最终回答。

正文出现 `jingyaogong` 的记录为 1,362 条，reasoning_content 中出现的为 1,248 条；两组可能重叠，不能直接相加作为身份修改条数。没有执行任何名字替换。

## 3. 抽样与追溯

全体随机样本包含 599 条无思考/工具标记记录、311 条思考标记记录、90 条工具相关记录。它是一次固定种子的样本，不要求比例精确等于总体；后续用它估计编码长度时，仍要报告样本规模和局限。

三类非空组合各保留 16 条观察样例，共 48 条；第四类样例文件为空。这些分组样例用于观察字段和优先复核，不能不加权地计算总体占比，也不是已经冻结的最终测试集。

原文样本保存在 implementation/data/processed/sft_candidate_inspection_v1 的 Git 忽略目录中。每条附带 source_line、包含原始换行的 source_line_sha256、形态组合及原始 record。报告保存各样本文件哈希；不把完整语料或未经复核的抽样原文加入 Git。

生成后已核对全部样本文件哈希，并重新读取原始来源，在 1,048 个不同的抽样位置逐条检查行哈希及 record 内容一致。该核验确认提取身份，没有把样本标记为质量合格或允许训练。

## 4. 复跑步骤

在 implementation 目录的 PowerShell 中执行。下载工具使用固定版本及预期整文件哈希，已有完整来源只核验、不覆盖。分段缓存与原下载前缀用于续传；没有关闭 TLS 校验或修改系统代理。

```powershell
& .\scripts\download_sft_source.ps1 -Connections 8 -ChunkMiB 16
```

首次本轮扫描使用以下命令。输出目录拒绝覆盖；复跑时将两个 v1 输出目录一起换成新的版本，不改写本轮报告。

```powershell
uv run --locked python scripts/inspect_sft_source.py `
  --source data/raw/minimind/sft_t2t_mini.jsonl `
  --expected-sha256 abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518 `
  --expected-size 1739201170 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/full_structure_v1 `
  --samples-dir data/processed/sft_candidate_inspection_v1 `
  --sample-size 1000 --stratum-sample-size 16 --seed 20261029
```

扫描时按行读取并同步计算 SHA256，结构异常按原因计数，标记可重叠、组合互斥；使用独立 Random 实例抽样，不改变训练使用的全局 RNG。输入身份不匹配时不输出分析报告。脚本不读取 Hugging Face token，也不会调用样本中的工具。

## 5. 验证与下一步

扫描器先通过 5 项小输入定向测试，又与前 20 条已独立统计的结果核对一致，再运行完整来源。SFT 编码、collator 与 Dataset 的 11 项测试和上述检查合计 16 项通过；本次全项目回归为 366 passed，用时 59.73 秒。新增测试由助手按委托整理，不记为学习者独立编写。

当前尚未处理工具 JSON 到模板输入的转换，没有计算全量 token 长度、人工审核全部样本、替换身份、划分训练验证集或冻结 SFT 配方。下一项工作是补清编码兼容性，再对固定样本统计模板后的长度与有效助手目标，之后讨论数据取舍。
