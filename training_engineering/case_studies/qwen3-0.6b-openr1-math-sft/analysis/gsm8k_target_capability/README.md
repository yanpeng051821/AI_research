# GSM8K 目标能力分析索引

## 当前状态

状态：`COMPLETE`。本目录已经完成 54 条固定分层样本的人工复核，覆盖 improved、regressed
和 both-wrong 三组。`01` 至 `06` 是按分析发生顺序保存的中间结论，其中出现的“下一步”
表示当时紧接着要做的分析节点，不是当前待办；最终综合结论以 `07` 为准。

## 阅读顺序

1. `PLAN.md`：分析问题、证据边界和停止条件。
2. `01_PAIRED_EFFECT_FORMAT_AND_LENGTH.md`：总体配对变化、格式与长度信号。
3. `02_FIXED_STRATIFIED_SAMPLING.md`：固定分层抽样合同。
4. `03_MANUAL_REVIEW_PROTOCOL.md`：人工复核字段与判定规则。
5. `04_REVIEW_BATCH_01_REGRESSIONS.md`：第一批回归样本。
6. `05_REVIEW_BATCH_02_AND_REGRESSION_SYNTHESIS.md`：回归样本综合。
7. `06_IMPROVED_REVIEW_SYNTHESIS.md`：改善样本综合。
8. `07_BOTH_WRONG_AND_FINAL_SYNTHESIS.md`：both-wrong 与最终综合结论。
9. `CHECKLIST.md`：分析任务的完成状态。

## 当前结论边界

这次分析支持对固定样本中输出长度、格式、终止行为和错误类型的判断，但不把 54 条样本
外推成完整数据分布，也不把 GSM8K 的改善解释成数学能力全面提升。下一项候选实验是固定
同一批 B0/S1 样本，比较 256 与 512 token 解码预算；在新合同冻结前不直接追加训练。
