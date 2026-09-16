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

## 用本次实际结果检验理解

NLL 从 0.780483 降到 0.546614；GSM8K 从 0.476118 升到 0.514784；
59 项回归主指标宏平均从 0.543412 降到 0.528562。
三个值回答不同问题，不能互相代替。宏平均还必须固定任务与指标选择，
不要与其他口径的 aggregate 数字直接相减。

MATH 的 512-token smoke 与完整 32K 评测不是同一合同；完整 S1 MATH 因预算中止，
没有可用的正式分数。未完成项不能计零，也不能只挑已完成的题当完整结果。
v2 当前延期是明确决策。54 条 GSM8K 复核支持这些抽样的错误判断，
不等于已经审核全部任务或证明所有变化由同一原因引起。

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

- [ ] 能独立计算 token-weighted NLL，解释与自由生成的差别。
- [ ] 能说明 B0/S1 必须冻结哪些字段，模型 checkpoint 为什么允许不同。
- [ ] 能追踪一次失败或预算中止如何进入最终结论。
- [ ] 能用逐样本变化解释总体分数，而不只读一个平均值。
- [ ] 能写一个比较合同的最小测试。
- [ ] 能提出一个下一步可验证假设，并说明需要哪些新证据。

学习日期、独立完成部分、查询或提示、测试结果、剩余问题：待填写。

