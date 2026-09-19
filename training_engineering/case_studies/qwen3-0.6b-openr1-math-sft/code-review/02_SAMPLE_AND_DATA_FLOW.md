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

现有测试已经覆盖变长样本 padding、padding 不参与监督、原始 labels 不被 collator
修改、冻结顺序与独立 sampler 第一轮一致，以及 checkpoint 保存 committed 而不是
prefetched 进度。因此没有重复添加同类断言。

本次补充了 `test_indexed_jsonl_dataset_pickle_separates_open_file_streams`：先读取一条记录，
确保原 Dataset 已打开文件流，再执行 pickle/unpickle。测试验证 pickle 不替换或关闭原
Dataset 的流，恢复对象初始不继承该流，第一次读取后会打开一个不同的独立流并返回正确
记录。它补上了原测试仅在 `_stream is None` 时 pickle 的覆盖缺口。

学习过程中还区分了两个容易混淆的进度：`position` 表示 sampler 已交给 DataLoader 的
索引，可能包含 worker 预取但尚未训练的样本；`committed_position` 表示已经成功纳入
完整 optimizer step 的样本。checkpoint 字段虽然名为 `position`，实际保存的是后者。

正式 TRL 路径不显式 commit，而是在单卡、单 epoch、固定 batch/accumulation、冻结顺序
等约束下，用 Trainer `global_step` 推导 `resume_offset`。这些约束变化时必须重新验证
恢复合同，不能直接复用该公式。

人工复核的 `review_decisions.jsonl` 是审计证据和决策输入，不是第二份训练样本。审计流程
根据这些决定生成最终 train/validation artifact；训练入口只读取冻结后的 artifact，不会把
review decisions 再追加进去，因此不会因为保留复核记录而重复训练样本。

## 快速问答

**Q：`sample_id` 和 `content_sha256` 分别标识什么？** `sample_id` 标识题目，同一道题只能有
一个候选进入冻结数据；`content_sha256` 同时约束题目和被选回答，用于确认最终选择的是哪条内容。

**Q：Dataset、Sampler、DataLoader 和 collator 各负责什么？** Dataset 按索引读取样本，
Sampler 决定索引交付顺序，DataLoader 调度读取和组批，collator 把变长样本 padding 成张量。

**Q：为什么不能用 DataLoader 已经取到的位置作为恢复点？** worker 可能提前预取但这些样本
尚未完成参数更新；严格恢复应以已经纳入完整 optimizer step 的 committed 位置为准。

**Q：每次 `__getitem__` 都 seek/read JSONL，会不会一定成为严重瓶颈？** 不一定。offset 避免
全文件重读，顺序 I/O、系统页缓存和 worker 预取会降低成本；只有基准显示供数跟不上 GPU 时，
才有依据改成 mmap、Arrow、WebDataset 等格式。

**Q：assistant-only labels 把 prompt 设成 `-100`，是否意味着 prompt 不参与模型计算？** 不是。
prompt token 不产生直接交叉熵项，但仍作为上下文影响 assistant token 的 hidden state 和梯度。

## 验收与学习记录

- [x] 能跟踪一个候选回答从原始数据到 batch 的字段变化。
- [x] 能区分原始 JSONL、tokenized artifact 和 review decisions。
- [x] 能手画两个不同长度样本的 `input_ids`、`attention_mask` 和 `labels`。
- [x] 能说明正式与独立 sampler 的差异。
- [x] 完成一个真实未覆盖边界的补测。

学习日期：2026-09-18。

独立完成部分：推导两层索引和两个 micro-batch 的样本顺序；手算动态 padding 后的三种
张量；解释冻结顺序不能再次打乱；跟踪 `position/committed_position`；编写已打开文件流
的 Dataset pickle 隔离测试，并根据第一次失败修正对象属性访问与断言。

查询或提示：在 Dataset/Sampler/DataLoader 职责边界、预取与提交进度、多进程文件流及
测试缺口选择上接受引导。第一次测试把恢复对象误当字典，失败调用链证明 `obj[key]` 会
进入 Dataset `__getitem__`；修正为属性访问后通过。

测试结果：`test_sft_data.py`、`test_artifact_data.py` 和 `test_trl_reference.py` 共 29 项
通过；新增测试单独通过；Ruff 格式、静态检查和补丁格式检查通过。

剩余边界：`__getstate__` 主要保护需要 pickle 的 spawn 路径；Linux fork 还要求 worker
创建前不持有打开流。远端新增 `_sample_ids/sample_id_at()`，既避免生成顺序清单时重复
解析完整 JSON，也让正式 DataLoader 创建 worker 前保持 `_stream is None`。
