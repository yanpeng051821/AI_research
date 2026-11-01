# SFT 工具字段适配验证

> 2026-10-30。按用户委托，由助手实现字段适配并接入编码器。此处记录编码兼容性，不是质量审核结论、训练数据接受清单或冻结配方。

## 1. 实现与范围

[prepare_sft_messages](../../../implementation/src/minimind_lab/data/sft.py) 先复用基础 validator，再深复制消息，将首条 system 的 tools 提取为列表，将助手 tool_calls 转为列表，返回 `(prepared_messages, tools)`。编码器把两者交给模板，再沿用既有 tokenization、助手 labels、Dataset 和 collator。

缺失、null、空白字符串和空列表按可选字段的空值处理；非空 tools 仅支持首条 system，非空 tool_calls 仅支持 assistant。工具声明和调用采用本来源的 function 包装结构；检查函数名、parameters 的对象类型和 arguments 的 JSON 对象形态，错误报告消息位置。arguments 若原为合法 JSON 字符串则保留其格式，原为字典则保留字典。reasoning_content 允许文本或 null，工具返回正文不作无条件 JSON 转换。

不排序消息、不补参数、不增加 system、不移除空 think、不改原始文件。未知字段随深复制保留；tools 在副本中移入模板独立参数，不是丢弃声明。此适配不检查工具是否已声明、参数是否满足完整 JSON Schema、工具结果对应关系或内容正确性；这些需另行审核。正文伪边界、超长与零监督目标门禁仍未解决。

## 2. 验证方法与结果

在 implementation 目录执行：

```powershell
uv run --locked pytest -q tests/test_sft_data.py tests/test_sft_tool_fields.py tests/test_sft_source_inspection.py
uv run --locked pytest -q
```

定向检查 45 项通过（3.06 秒），全项目 395 项通过（47.53 秒）。测试由助手编写，覆盖字符串/列表输入、深复制、空值、字段类型、JSON 错误、声明位置、调用角色、缺失参数、模板参数传递与原有编码组批行为。

另以本地 tokenizer 做真实数据核验：

| 输入 | 检查 | 结果 |
| --- | --- | --- |
| 原始前 20 条对话 | 与适配前直接套模板的 input_ids、labels 逐条比较 | 全部一致 |
| 固定分组样例 48 条 | plain/reasoning_only/tool_only 各 16 条，核对样本文件 hash 后编码，检查标签长度、有效目标与输入未变 | 全部成功编码，均有有效目标；不表示内容全部正确 |
| 前两条普通记录 | Dataset 与 DataLoader 真实组批 | 三张量 `[2,809]`，真实长度 `[549,809]`，目标 `[414,687]` |
| 源行 794,513 的单位换算 | 查看两个助手监督片段，核对调用被选中、工具结果未选中 | 462 token、86 个 shift 后目标，区间 `[327,393)`、`[442,462)` |
| 完整来源文件 | 重新读取并核对 SHA256 | 与原结构报告一致，来源未改写 |

源行 776,360 的未声明天气工具问题依然存在；它现在也能被编码，恰好说明“编码成功”和“示范合格”是不同结论。没有把已发现的问题记录标成已接受，也没有执行工具。

输入身份与机器可读结果见 [report.json](report.json)。48 条来自固定分组样例，不代表全量，未统计全量长度、处理超长、生成切分数据或启动 SFT。下一步先讲清本次接口，再确定未解决的边界检查与长度分析规则。
