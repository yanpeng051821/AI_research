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

## 6. 本案例当前可以与不可以比较的结果

可以直接比较：

```text
B0 full validation NLL 0.7804831795
vs
Pilot-100 full validation NLL 0.5952246359
```

因为 validation artifact、记录数和有效 token 均相同。

不可以直接比较：

- B0 GSM8K 1,319 题 `0.4761` 与 pilot 50 题 `0.52`；
- B0 完整 regression 与 pilot 小样本 smoke；
- B0 完整 MATH-500 与未完成终态的 pilot MATH-500；
- 20-step 的 128 条 validation 子集与全量 1,968 条 NLL。

## 7. 当前评测阻塞

Pilot MATH-500 一次 invocation 长期保留 `running`，另一次 vLLM KV cache 初始化失败；
因此没有可靠 pilot MATH-500 结果。正式 S1 前要在 B0 和 pilot/final-like checkpoint 上用
同一小规模入口验证：

- context 不会被压缩为零；
- vLLM 能分配 KV cache；
- 成功写 completed；
- 失败写 failed、return code、命令和 traceback；
- 输出可以被 paired comparator 读取。

## 8. 验收门与产物

- [ ] B0/S1 所有 suite 合同一致。
- [ ] 每个运行有终态 invocation manifest。
- [ ] 聚合指标和逐样本输出都存在。
- [ ] NLL 配对通过 sample/token 合同检查。
- [ ] 输出 changed samples 和 regressions。
- [ ] 结论区分训练行为、目标能力和回归。
- [ ] 不把 smoke/pilot 子集指标写成正式能力分数。
- [ ] 下一轮动作来自错误证据而不是主观猜测。
