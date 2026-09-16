# 02 一条真实样本如何变成训练 Batch

## 本次要解决的问题

一条数学题经过候选选择、tokenization、监督标记、排序和 padding 后，最终哪些位置参与
loss？DataLoader 已经取走的样本是否一定已经被训练？

对应案例文档：上一级 03、04。我们用同一条小样本贯穿整个主题。

## 代码阅读顺序

| 文件 | 本次重点 |
| --- | --- |
| [audit.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/audit.py) | _select_verified_generation、audit_openr1_rows 中选择、拒绝与分流 |
| [data.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/data.py) | build_single_turn_sft_sample、IndexedTokenizedSFTDataset、collate_sft_batch |
| [runner.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/runner.py) | _build_dataloaders 的实际装配 |
| [trl_reference.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/trl_reference.py) | frozen_epoch_indices、OrderedCompletionMaskDataset |
| [正式脚本](D:/pythonlearning/small_model_post_training/independent_implementation/scripts/train_sft_trl.py) | TRL collator 如何接收 completion_mask |

审计先抓主流程和一个接受/拒绝分支，外部排序等大规模文件操作按需要展开。

## 一条样本的完整链路

```text
原始题目 + 候选回答 + 验证字段
-> 候选选择 / 去重 / 污染与长度检查
-> chat template + tokenize
-> input_ids / assistant-only labels
-> 冻结 tokenized JSONL
-> Dataset 按索引读样本
-> sampler 决定索引顺序
-> DataLoader 收集本批样本
-> collator 动态补齐
-> 张量 batch 交给模型
```

Dataset 读到的正式 artifact 已经 tokenized，并非每个 epoch 都重新调用 tokenizer。
IndexedTokenizedSFTDataset 初始化时扫描、解析并校验记录、检查 sample_id 重复，持久保存
offset 列表；取样时 seek 并解析对应行。它不只是无校验地建立 offset。

独立实现的 StatefulRandomSampler 提供索引，DataLoader 形成批次，collate_sft_batch
产生张量。正式路径先冻结索引，再由 OrderedCompletionMaskDataset 适配字段，
RemainingIndices 提供剩余位置，批次由框架 DataLoader 与 TRL collator 构建。
正式路径不调用独立 sampler 的 commit。

## 手工追踪监督位置

用示意 token ID：

```text
input_ids = [10, 11, 12, 20, 21, 99, 30]
labels    = [-100, -100, -100, 20, 21, 99, -100]
```

10–12 表示 prompt/assistant 开始部分，20、21 是回答，99 是 assistant 结束标记，
30 是不监督的尾部。这里 99 也要学习。ID 是教学示例，不是 Qwen 的实际词表 ID。

再与长度为 5 的样本组成一批，画出 [2,7] 的 input_ids、labels 和 attention_mask。
padding input 使用 pad ID，padding labels 为 -100，padding attention 为 0。
本批最长长度决定张量宽度；16K 是 artifact 长度上限，两者用途不同。

随后解释 attention_mask、模型 causal mask 和 labels mask 的区别。忽略 prompt 的
直接 token loss 不等于 prompt 不参与计算或完全没有梯度影响。

## 测试与动手

阅读 [test_sft_data.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_sft_data.py) 和
[test_artifact_data.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_artifact_data.py)，先关注动态 padding 与 lazy reading。

```powershell
uv run python -m pytest tests/test_sft_data.py tests/test_artifact_data.py -q
```

练习：在临时数据上构造两条长度不同的样本，手算 batch 后写断言，覆盖 padding 不参与
监督、原始 labels 不被 collator 修改。先检查已有测试，选择一个未覆盖的边界。
然后口头跟踪 sampler 的 position 与 committed_position；恢复细节留到主题 05。

## 验收与学习记录

- [ ] 能跟踪一个候选回答从原始数据到 batch 的字段变化。
- [ ] 能区分原始 JSONL、tokenized artifact 和 review decisions。
- [ ] 能手画两个不同长度样本的三种张量。
- [ ] 能说明正式与独立 sampler 的差异。
- [ ] 完成一个小数据断言或补测。

学习日期、独立完成部分、查询或提示、测试结果、剩余问题：待填写。

