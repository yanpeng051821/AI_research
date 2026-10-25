# 完整 mini 数据切分与编码审计

> 2026-10-24 数据合同 `pretrain_mini_data_v1` 已冻结。2026-10-25 按实验集中归档到本目录；数据合同、manifest、编码审计及冻结清单字节保持不变。对应训练约定见 [training_recipe](../training_recipe/TRAINING_CONTRACT.md)。

## 1. 结果

| 项目 | 训练 | 验证 | 排除 |
| --- | ---: | ---: | ---: |
| 记录数 | 1,265,235 | 5,000 | 3 |
| 截断前正文 tokens | 328,643,558 | 1,310,713 | 不参与编码审计 |
| 长度 768 下保留的正文 tokens | 300,545,455 | 1,201,758 | — |
| causal shift 后有效预测 targets | 301,810,690 | 1,206,758 | — |
| 需要截尾的记录 | 49,628（3.92%） | 201（4.02%） | — |
| 正文长度 P95 / P99 | 571 / 1,433 | 569.05 / 1,417.04 | — |

完整原始 text 的 SHA256 为分组单位，保留组内重复记录；验证从允许的组中排序后使用独立 random.Random(43) 抽取 5,000 组。本次恰好选中 5,000 个单条组。12 条既有人工决定转成 3 个排除组和 9 个仅训练组，按整个正文组应用。

全量编码使用本地官方 tokenizer，输入为 BOS + 正文前 766 tokens + EOS。5,000 条验证输入均不同；训练输入与验证输入完全相同的记录为 0，保留正文中包含 pad ID 的记录为 0。审计先验证文本切分文件 hash，结束后复查，未修改任何数据。

另外以独立三路归并方式将输出按 source_line 合并，再逐行与原始文件比较：1,270,238 条全部覆盖，来源行号和原正文一致。精确输入隔离不代表近似文本、原始文档或事实质量已经得到保证。

## 2. 证据与本地产物

本目录集中保存本次实验采用的数据配方及冻结身份。文本处理记录在 manifest 的 split_policy，编码规则和检查结果记录在编码报告；模型初始化、优化器、预算和评测规则放在同级 training_recipe 中。实现讲解与冻结管理见[第七周教程](../../../practice_notes/week_07_pretrain_experiment.md)。

- [DATA_CONTRACT.md](DATA_CONTRACT.md)：已冻结的数据合同，统一定义文本身份、切分、tokenizer、长度、序列构造、监督位置和已知限制。
- [data_contract_freeze.json](data_contract_freeze.json)：记录合同 ID、合同文件 SHA256 及其引用的数据、编码和 tokenizer 身份。
- [manifest.json](manifest.json)：原始来源、人工复核与历史候选锚点、切分策略、输出计数、文件大小和 SHA256。
- [encoding_audit_768.json](encoding_audit_768.json)：完整训练/验证编码统计、tokenizer 和编码代码身份、输入重合检查及限制。

大文件不进入 Git，本机位置为 implementation/data/processed/pretrain_mini_v1/split_seed43_v1/。本目录只归档小型元数据，不能代替原始数据和训练文件的备份。manifest 的状态仅表示文本切分完成，编码审计的 passed 仅表示该候选长度通过本报告中的检查。

训练固定 768 位置时，有效输入位置占约 31.19%，其余为 padding。这是序列位置统计，不是 GPU 利用率。冻结数据合同采用长度 768；训练 batch、学习率、检查点和验证频率已在 training_recipe 中确定并通过短跑。本次没有引入 packing 或改变既有 collator。

## 3. 重建

在 implementation 目录执行。输出目录必须不存在；保留旧产物时改用新目录，并将相同路径传给审计入口。

```powershell
uv run --locked python scripts/build_pretrain_mini.py --output-dir data/processed/pretrain_mini_v1/split_seed43_v1
uv run --locked python scripts/audit_pretrain_encoding.py --split-dir data/processed/pretrain_mini_v1/split_seed43_v1 --max-length 768
uv run --locked pytest tests/test_pretrain_mini.py tests/test_pretrain_encoding_audit.py -q
```

第一条命令使用默认固定来源清单、完整原始 mini 文件、历史 10,000 条候选文件与已归档复核决定。10,000 条候选只用于证明人工复核的出处，不限制此次完整语料训练集的范围。不同操作系统上的路径、时间等 manifest 字段可能不同，文本输出固定 UTF-8/LF；复现切分应比较数据文件 hash 和策略，不要求整份 manifest 字节相同。

## 4. 协作范围

学习者在带练中完成分组、人工约束读取、验证抽样、写入及流程串联。经用户明确委托，助手完成来源门禁、manifest 发布、CLI、编码审计与回归，实际执行完整构造和独立对照，并维护归档。后续转向训练配方与设备测量，不再要求学习者逐行补写这些重复性工程代码。
