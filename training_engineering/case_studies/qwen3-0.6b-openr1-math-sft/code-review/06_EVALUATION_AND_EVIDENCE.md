# 06 评测、结果配对与证据闭环

## 本次要解决的问题

模型训练完以后，怎样证明它在目标任务上发生了什么变化？
为什么评测完成、分数可比较、能力改善是三个不同判断？

对应案例文档：上一级 05、08、10、11，以及 analysis/gsm8k_target_capability。

## 代码阅读顺序

| 路径 | 重点 |
| --- | --- |
| [evaluate_validation_nll.py](D:/pythonlearning/small_model_post_training/independent_implementation/scripts/evaluate_validation_nll.py) → [nll_evaluation.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/nll_evaluation.py) | 模型加载、逐样本 loss sum 与 token 数 |
| [nll_comparison.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/nll_comparison.py) | 样本集合、有效 token 与配对变化 |
| [run_frozen_eval.py](D:/pythonlearning/small_model_post_training/independent_implementation/scripts/run_frozen_eval.py) → [eval_runner.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/eval_runner.py) | 合同到命令、执行记录与终态 |
| [eval_comparison.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/eval_comparison.py) | invocation、prompt 与逐样本结果对齐 |
| [generation_budget.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/generation_budget.py) | 长度、截断与成本判断 |
| [gsm8k_analysis.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/gsm8k_analysis.py) | changed pairs 与固定分层抽样 |

## 先走两条评测链

```text
冻结 validation artifact
-> teacher-forcing forward
-> 每条 loss_sum / valid_tokens
-> 全集合 sum(loss_sum) / sum(valid_tokens)
-> B0/S1 按 sample_id 配对
```

NLL 评测使用已有目标 token，不是让模型自由生成答案。model.eval() 控制部分层的行为；
no_grad 控制 autograd，两者不能互相代替。

入口固定使用 `batch_size=1`，不是因为 NLL 数学上只能逐条计算，而是当前
`masked_sft_loss_sum_and_count()` 返回整个 batch 的聚合值；单条 batch 才能把 loss、有效
token 数和 `sample_id` 无歧义地绑定并落入 `records.jsonl`。

本项目同时保留两个不同平均口径：

- summary 的全局 NLL 是 `sum(loss_sum) / sum(valid_tokens)`，每个监督 token 等权；
- paired report 的 mean NLL 是逐样本 `mean_nll` 的算术平均，每条样本等权。

两者回答不同问题，数值不能混称。NLL 的 `S1-B0 < 0` 表示改善；higher-is-better 的外部
任务指标则是 `S1-B0 > 0` 表示改善。

```text
冻结任务、prompt、scorer、解码与预算合同
-> 对 B0 或 S1 运行评测
-> 保存每条输出与评分 / invocation 终态
-> 检查身份和样本匹配
-> 汇总改善、回退、不变
-> 固定抽样与错误分析
```

不同任务可能使用生成或候选答案 likelihood 打分，不能把所有回归任务都当成自由生成。
生成预算的 max_new_tokens 也不同于训练中 batch 动态 padding 的最大长度。

外部评测比较有三层门禁：先要求 invocation `completed` 且 return code 为零；再要求软件、
任务、prompt/解码和预算合同相同；最后要求逐样本 example、full prompt、few-shot、input
tokens、gold 和 choices 等不变量相同。模型 checkpoint 与输出目录允许不同，因为前者是实验
变量，后者必须隔离产物。执行成功、合同相同、逐样本可配对三者全部成立后，分数才可比较。

`changed_records` 不只保留指标翻转：只要 prediction 文本不同，即使 exact match 仍为
`0 -> 0`，也会保留双侧输出，以便检查题意、推理、格式、终止和截断行为。

## 用本次实际结果检验理解

NLL 从 0.780483 降到 0.546614；GSM8K 从 0.476118 升到 0.514784；
59 项回归主指标宏平均从 0.543412 降到 0.528562。
三个值回答不同问题，不能互相代替。宏平均还必须固定任务与指标选择，
不要与其他口径的 aggregate 数字直接相减。

MATH 的 512-token smoke 与完整 32K 评测不是同一合同；完整 S1 MATH 因预算中止，
没有可用的正式分数。未完成项不能计零，也不能只挑已完成的题当完整结果。
v2 当前延期是明确决策。54 条 GSM8K 复核支持这些抽样的错误判断，
不等于已经审核全部任务或证明所有变化由同一原因引起。

GSM8K 的全量配对结果为 improved 199、regressed 148、both-correct 480、both-wrong 492，
共 1,319 题；B0 正确 `148+480=628`，S1 正确 `199+480=679`，净增 51 题。exact
McNemar 双侧 `p=0.00718`，题目级 paired bootstrap 95% 区间约为 `+1.14pp` 到
`+6.60pp`。这些证据支持当前冻结 GSM8K 合同下存在稳定正向配对效应，但不能扩大为数学
推理能力全面提升。

54 条人工复核使用 coverage-first、hash-stable 的固定分层抽样：improved、regressed 和
both-wrong 各 18 条，并覆盖 marker transition 与长度层。它适合发现和复核机制，不是按
总体比例抽取，因此不能把人工标签比例外推到全部 1,319 题。

下一轮候选假设是：在相同 B0、冻结数据/顺序、token 预算和评测合同下，LoRA 相比本次
全参数 SFT 能减少 59-task 通用回归，同时保留部分 GSM8K 提升。LoRA 参数范围是主变量；
学习率适配需通过低成本 pilot 单独记录，换基座模型则应作为另一个案例，不能混入这次消融。

## 测试与动手

```powershell
uv run python -m pytest tests/test_nll_comparison.py tests/test_eval_comparison.py tests/test_generation_budget.py -q
```

练习：构造 4 对小结果，包含 improved、regressed、both-correct 和 both-wrong，
手算总分差与变化数；随后故意改一条 prompt 或样本 ID，判断比较器应该拒绝什么。
使用现有 mismatch 测试作参照，只增加尚未覆盖且有意义的检查。
这一步可以在 CPU 本地完成，不需要重新生成模型输出。

## 一个实验何时能关闭

先检查实际执行状态，再检查结果可比性，最后解释模型行为。
至少能定位训练身份、模型发布版本、评测合同、逐样本输出、比较报告与限制说明。
发布模型的 final_model 与可恢复训练的 checkpoint 是不同交付物；
代码、日志、权重、数据各自有不同归档位置。

## 验收与学习记录

- [x] 能独立计算 token-weighted NLL，解释与自由生成的差别。
- [x] 能说明 B0/S1 必须冻结哪些字段，模型 checkpoint 为什么允许不同。
- [x] 能追踪一次失败或预算中止如何进入最终结论。
- [x] 能用逐样本变化解释总体分数，而不只读一个平均值。
- [x] 能写一个比较合同的最小测试。
- [x] 能提出一个下一步可验证假设，并说明需要哪些新证据。

学习日期：2026-09-22。

本轮独立完成了 token/sample 两种 NLL 口径、外部评测控制变量、配对 outcome、未完成 MATH
评测边界和固定分层抽样问题。新增
`test_preserves_changed_predictions_when_metric_is_unchanged`，验证 exact-match 不变时仍保留
行为已变化的预测；编写过程中定位并修复了循环变量作用域和重复字典 key 两个问题。

验证结果：`test_nll_evaluation.py`、`test_nll_comparison.py`、`test_eval_runner.py`、
`test_eval_comparison.py`、`test_generation_budget.py` 和 `test_gsm8k_analysis.py` 合计 21 个
测试通过，Ruff 与 `git diff --check` 通过。
