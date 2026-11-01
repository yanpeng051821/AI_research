# 第一版派生候选：已被修订版替代

本轮从 808,513 条候选出发，暂存 21,066 条待复核记录，身份修订后再去重 15 条，得到 787,432 条，分为训练 785,384 条、验证 2,048 条。处理脚本完成了全量编码和产物 hash 核验，但这不等于训练输入验收完成。

独立复验在实际构造训练集 `SFTDataset` 时失败：原读取器调用 `datasets.load_dataset("json")`，前段普通消息形成的 Arrow schema 不含 tools/tool_calls，读到后段工具记录时报错：

```text
Couldn't cast array of type
struct<role: string, content: string, reasoning_content: string, tools: string, tool_calls: string>
to
{role: string, content: string, reasoning_content: string}
```

因此没有生成成功的 verification.json，也没有将此版冻结或启动训练。随后将 SFT 读取层改为 JSONL 字节偏移索引，支持异构可选字段，并检查多 worker 的私有文件流。

另从本版训练部分固定抽取普通、思考、工具形态各 8 条，共 24 条。主助手查看完整上下文，记录任务响应、思考质量和不确定事实，提出 9 条具体暂存。见[内容复核](../quality_review_v1/README.md)。这不是整体质量率估计，也不是继续抽到“看起来满意”为止。

本目录保留旧报告、候选合同和处理代码/策略快照作为历史证据。当前应查看[第二版准备报告](../preparation_768_v2/README.md)及[当前数据合同](../../sft_mini_v1/data_recipe/DATA_CONTRACT.md)。旧文件中的源码 hash 对应当时快照，不要求当前已修正的工作区源码继续匹配旧版。
