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
练习：用 tiny model 比较同一组样本的 full batch 与不等长 micro-batch 累积。
step 前比较梯度，step 后比较参数；模型初始参数、精度、optimizer 状态和随机性必须一致。
可在临时演示代码故意使用 batch mean 平均，预测哪个断言应失败，再运行验证。

## 验收与学习记录

- [ ] 能解释每个维度和 shift 后有效 token 的计数。
- [ ] 能借助 API 文档独立写核心 loss。
- [ ] 能说明何时清梯度、何时更新参数与学习率。
- [ ] 能解释数值对齐测试的控制变量和适用边界。
- [ ] 能区分 loss 下降、生成正确与任务得分提升。

学习日期、独立实现部分、查询或提示、测试结果、剩余问题：待填写。

