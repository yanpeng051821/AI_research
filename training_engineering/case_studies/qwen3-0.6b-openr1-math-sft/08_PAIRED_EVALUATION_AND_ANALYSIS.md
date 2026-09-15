# 08 配对评测、结果解释与错误分析

## 1. 评测不是训练结束后的附属步骤

评测合同应在训练前冻结，B0 先执行，S1 后按原合同重放。否则无法区分模型变化和评测
变化。正确流程是：

```text
冻结 evaluation contract
-> B0 全量评测和逐样本输出
-> 训练
-> S1 相同合同评测
-> 合同一致性检查
-> 聚合差值
-> 逐样本变化与回归
-> 结论和下一轮假设
```

## 2. Validation NLL 的配对比较

对每条样本保存：

```text
sample_id
loss_sum
valid_token_count
mean_nll
```

比较命令：

```bash
uv run python scripts/compare_validation_nll.py \
  --baseline-records runs/<b0>/validation-nll/records.jsonl \
  --trained-records runs/<s1>/validation-nll/records.jsonl \
  --output runs/<comparison>/validation-nll.json
```

比较器应拒绝 sample ID 集合不同、重复 ID、valid token count 不一致或缺失样本。全局 NLL
按所有有效 token 加权，不是简单平均每条样本的 mean NLL。

解释边界：

```text
NLL 下降
=> 模型更会预测目标 validation completion token

NLL 下降
!=> 数学最终答案正确率一定提高
!=> 通用能力没有回退
```

## 3. LightEval 配对比较

```bash
uv run python scripts/compare_lighteval_results.py \
  --baseline-root runs/<b0>/evals \
  --trained-root runs/<s1>/evals \
  --output-dir runs/<comparison>/lighteval
```

比较器要先校验：

```text
task/version
dataset revision
prompt/chat template
few-shot
generation parameters
input/gold/choices
scorer
每题采样数
```

一致后再输出 aggregate delta、changed predictions 和 regressions。只看两个终端百分数并
手工相减，不是可信配对分析。

## 4. 三层结果解释

### 第一层：训练行为

看 train loss、validation NLL、grad norm、LR、有效 token 和稳定性。回答“模型是否在学
训练目标”。

### 第二层：目标任务能力

看 MATH-500、GSM8K 和逐题结果。回答“最终答案能力是否改善”。

### 第三层：回归与副作用

看 MMLU、ARC、HellaSwag、格式、生成长度、EOS 和错误类型。回答“改善是否以其他能力为
代价”。

三个层次不能互相替代。

## 5. 错误分析最少要做什么

对 B0 与 S1 的同一题按状态分组：

```text
B0 错 -> S1 对：真正改善候选
B0 对 -> S1 错：回归样本
B0 错 -> S1 错：仍未解决
B0 对 -> S1 对：保持能力
```

再按原因人工抽样：

- 推理方向正确但算术错误；
- 推理冗长，未在长度预算内结束；
- 格式不符合 scorer；
- 最终答案正确但解析失败；
- 题目理解错误；
- SFT 模仿了训练风格但没有提高求解能力；
- 生成提前 EOS、无 EOS、空输出或重复；
- 通用任务发生明显偏置或遗忘。

错误分析的输出不是故事，而是下一轮可证伪假设。例如：

```text
若主要回归来自过长推理
-> 下一轮测试长度分布/终止行为，而不是直接增加训练 steps

若 NLL 下降但答案准确率不变
-> 检查数据质量、目标答案与 verifier，而不是宣称训练无效
```

## 6. 正式 S1 已完成的配对结果

以下结果使用冻结 B0 和正式 S1，并保留逐样本证据：

```text
token-weighted validation NLL: 0.780483 -> 0.546614（-29.96%）
GSM8K qem:                    0.476118 -> 0.514784（+3.87pp）
59-task regression macro:    0.543412 -> 0.528562（-1.49pp）
57-task MMLU macro:          0.545157 -> 0.530376（-1.48pp）
ARC-Challenge acc_norm:      0.453925 -> 0.431741（-2.22pp）
HellaSwag acc_norm:          0.533459 -> 0.522008（-1.15pp）
```

NLL 证明模型学到了目标 completion 分布；GSM8K 给出目标能力的正向证据；通用面板则
显示这种改善伴随广泛回退。59 项中按各项主指标统计，15 项改善、43 项下降、1 项不变。

### MATH-500 的边界

20 题、每题 4 次、512-token 的同合同 smoke 从 pass@1:1 `0.25` 降至 `0.05`，但 S1 的
80 条生成全部触及 512-token 上限，因此这个结果同时混入了严重截断效应，不能代替完整
MATH-500。随后启动的 500 题、每题 4 次、32K-token 评测在完成 204/2000 条后，已运行
约 2 小时 55 分钟且动态剩余时间约 17 小时，因成本门禁主动中止。

这次中止不是模型评测分数，也不能记成失败样本为零分。它证明原 v1 评测合同不适用于
当前 S1 的生成长度分布。

## 7. 评测合同 v2 修订

原 v1 文件保存在 `configs/gate0b/evaluation_v1_executed.yaml`，仅作为本次已执行实验的
身份凭据，不再作为默认入口。v2 增加：

- 50 题、每题 4 次、2048-token 的预算探针；
- 停止率、截断率、预计 GPU 小时和预计费用门禁；
- `<|im_end|>` 与 `<|endoftext|>` 两个显式 stop token；
- 4K-token 的有界主评测与单独受保护的 32K 参考评测；
- 所有 MATH 套件当前标记为 `deferred`，必须记录新决策并显式解锁。

用户已决定本阶段不继续 MATH-500。后续只有在新的数据分析提出可验证假设后，才重新
开启探针；不得为了补齐表格直接运行完整 32K 合同。

## 8. 验收门与产物

- [x] Validation NLL、GSM8K 和 regression 的 B0/S1 合同一致。
- [x] 已完成运行有终态 invocation manifest、聚合指标和逐样本输出。
- [x] NLL 配对通过 sample/token 合同检查。
- [x] 输出 changed samples 和 regressions。
- [x] 结论区分训练行为、目标能力和回归。
- [x] MATH smoke 未被写成正式能力分数。
- [x] 不完整的 MATH full 被标记为预算中止，不伪造结果。
- [x] 后续分析完成后，决定本阶段不执行 v2 MATH 探针并保持 `deferred`。

正式评测后已经完成 54 条固定分层 GSM8K 样本的逐条人工复核，包括 improved、regressed
和 both-wrong 三组；阅读入口为 `analysis/gsm8k_target_capability/README.md`。MATH v2
的 `deferred` 是一次明确的成本决策，不表示配对结果分析尚未开始或工程收尾未完成。
