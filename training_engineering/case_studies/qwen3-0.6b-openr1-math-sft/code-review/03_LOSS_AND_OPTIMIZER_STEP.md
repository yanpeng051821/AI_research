# 03 SFT Loss、梯度累积与一次参数更新

## 本次要解决的问题

给定 batch，怎样确认模型优化的确实是目标 assistant token 的平均负对数似然？
为什么不同长度 micro-batch 不能随意平均各自的 mean loss？

对应案例文档：上一级 04；本主题以独立实现为计算主线，为下一篇 TRL 对照建立标准。

## 代码阅读顺序

| 文件与函数 | 观察重点 |
| --- | --- |
| [training.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/training.py)，forward_sft_batch | 模型实参、outputs.logits 和 labels |
| [sft.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/sft.py)，masked_sft_loss_sum_and_count | shift、mask、sum、count |
| [engine.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/engine.py)，run_optimizer_step | 完整累积窗口、backward、clip、update |
| 同文件 build_optimizer / build_scheduler | 参数组、动量、weight decay、warmup |
| [shadow.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/shadow.py) | 同 batch 的参考 loss 和梯度比较 |

## 沿张量走一次更新

```text
input_ids / attention_mask / labels [B,T]
-> model forward
-> logits [B,T,V]
-> logits[:, :-1] 对齐 labels[:, 1:]
-> 有效目标的交叉熵之和与 token 数
-> backward 累积参数梯度
-> gradient norm / clipping
-> optimizer.step
-> scheduler.step
```

参数梯度与对应参数 shape 一致，因为求导对象最终是标量目标。
本模型不是对所有 logits 都有直接监督；但受监督位置通过上下文依赖影响模型参数。

累积窗口中第 i 个 micro-batch 的 loss sum 为 S_i，有效 token 数为 N_i，目标为：

```text
L = (S_1 + ... + S_k) / (N_1 + ... + N_k)
```

engine 先从 shifted labels 算出整个窗口的分母，再逐批对 S_i / 总 token 数 backward。
因此不需要保存整个窗口全部计算图。已经 detach 的统计值用于日志。
这里“全局”指当前单设备的一次更新窗口；多卡归一化不能直接从这段实现推断。

## 一个必须自己算的例子

A 有 2 个有效 token、loss sum 为 4；B 有 8 个有效 token、loss sum 为 8。
token 平均为 12/10；两个 batch mean 再平均为 (2+1)/2。
两种目标不同。token 平均让有效 token 的系数相同，并不表示每个 token 的实际梯度大小相同，
也不意味着它对所有任务都优于样本平均。

读 engine 时再检查：zero_grad 在窗口前；backward 每个 micro-batch 一次；
optimizer 与 scheduler 每个更新窗口各一次；末尾不足完整累积窗口也要处理。
小心日志中记录的是本次更新使用的 LR，还是 scheduler 更新后的 LR。

## 测试与动手

```powershell
uv run python -m pytest tests/test_sft_loss.py tests/test_sft_training.py -q
uv run python -m pytest tests/test_engine.py -k "optimizer_step or scheduler" -q
```

阅读 [test_sft_training.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_sft_training.py) 中已有数值对照。
现有测试已经用相同初始参数、输入和 SGD 状态比较 full batch 与有效 token 数不同的
micro-batch 累积，并在一次更新后比较全部参数，因此没有重复添加同类测试。讨论中进一步
确认：即使日志仍按正确的 loss sum/count 计算，错误地对每个 micro-batch mean loss
backward 也可能只让参数对齐断言失败，说明“日志正确”不能证明优化目标正确。

本次新增
`test_optimizer_step_reports_current_lr_before_scheduler_updates_next_lr`，使用一个每次将 LR
减半的 scheduler stub，验证 `StepMetrics.learning_rate` 记录本次 `optimizer.step()` 实际
使用的 0.1，scheduler 只执行一次，并为下一步把 optimizer LR 更新为 0.05。第一次运行
因类名 `HavingScheduler/HalvingScheduler` 不一致在进入生产函数前失败；修正名称并删除无关
`tmp_path/config` 后通过。

同时复核了裁剪前 `grad_norm` 的口径：持续高于阈值只能说明原始梯度被强烈裁剪，不能据此
声称“没有裁剪”或“裁剪后又变大”。AdamW 的 `m/v`、scheduler、RNG 与数据位置共同决定
可恢复训练状态，只保存模型权重不足以严格续训。

## 快速问答

**Q：为什么要使用 `logits[:, :-1]` 和 `labels[:, 1:]`？** 位置 `t` 的 logits 预测位置
`t+1` 的 token；最后一个 logits 在当前序列中没有下一个标签，首个标签也没有对应的前置预测。

**Q：为什么先累计 loss sum 和有效 token count，而不是平均每个 micro-batch 的 mean loss？**
后者会让短 batch 中每个 token 获得更大的系数；`sum(loss)/sum(tokens)` 才让当前更新窗口内
每个有效 token 使用相同的归一化系数。

**Q：一次累积窗口包含多个 micro-batch，是否要同时保留所有计算图？** 不需要。先确定整个
窗口的有效 token 总数，再让每个 `loss_sum/total_tokens` 依次 backward，梯度会在线性累加。

**Q：为什么 optimizer step 前检查梯度，step 后检查参数？** step 前梯度已经形成但参数尚未
更新，适合验证更新输入；step 后参数已经变化，适合检查更新结果是否出现 NaN/Inf。

**Q：日志中的 learning rate 应记录 scheduler 前还是后的值？** 应记录本次
`optimizer.step()` 实际使用的当前值；随后 `scheduler.step()` 才产生下一次更新使用的值。

**Q：validation NLL 下降是否等于数学答题准确率提高？** 不等于。NLL 衡量 teacher-forced
目标 token 概率，任务准确率还受到自由生成轨迹、格式和最终答案判定等因素影响。

## 验收与学习记录

- [x] 能解释每个维度和 shift 后有效 token 的计数。
- [x] 能借助现有实现说明核心 loss 的完整计算。
- [x] 能说明何时清梯度、何时更新参数与学习率。
- [x] 能解释数值对齐测试的控制变量和适用边界。
- [x] 能区分 loss 下降、生成正确与任务得分提升。

学习日期：2026-09-19。

独立完成部分：手算 causal shift、有效 mask 和 token count；区分 micro-batch mean 与全窗口
token mean；解释 zero_grad、backward、optimizer 和 scheduler 的时序；计算 tail accumulation
与 warmup step；编写并修正 LR 时序测试。

查询或提示：在全窗口 backward 分母、裁剪前 grad norm 口径、AdamW `m/v` 恢复语义和
测试缺口选择上接受引导。能够说明 validation NLL 改善与 GSM8K exact-match 不变并不
矛盾：前者是 teacher-forced token 概率，后者是完整自由生成后的离散结果。

测试结果：`test_sft_loss.py`、`test_sft_training.py` 和 `test_engine.py` 共 38 项通过；新增
测试单独通过；Ruff 格式与静态检查通过。

剩余边界：当前对齐测试使用确定性 tiny model 和单设备计算；dropout、多卡归一化、不同
精度及 batch-dependent 层会改变严格数值对齐条件，不能从本测试直接外推。
