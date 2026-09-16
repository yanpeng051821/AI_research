# 05 Checkpoint、失败处理与显存

## 本次要解决的问题

训练中断后，“能加载模型”和“能继续同一次训练”有什么差别？
看到 GPU 接近满载时，怎样通过落盘数据判断原因？

对应案例文档：上一级 06、07、09、10、11。

## 代码阅读顺序

| 文件 | 重点 |
| --- | --- |
| [checkpointing.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/checkpointing.py) | 独立实现的状态保存、加载、空间与保留策略 |
| [data.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/data.py) | StatefulRandomSampler 的 position / committed_position |
| [engine.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/engine.py) | 更新成功后 commit，保存与 soft stop |
| [trl_training.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/trl_training.py) | 原生 TRL 恢复、SaveAndStopCallback、CudaMemoryTelemetryCallback |
| [observability.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/observability.py) | 独立实现的异常快照 |
| [train_sft_trl.py](D:/pythonlearning/small_model_post_training/independent_implementation/scripts/train_sft_trl.py) | 正式路径 finally 与 attempt 记录 |

## 先走独立实现的状态变化

假设一个更新窗口消费两个 micro-batch。sampler 每 yield 一个索引，position 前进；
DataLoader 可以预取，因此 position 可能领先实际训练。成功完成一次 optimizer update 后，
engine 按该窗口样本数调用 commit，committed_position 才前进。

checkpoint 保存的是恢复所需的已提交位置，以及模型、optimizer、scheduler、RNG、训练步数
和合同身份。恢复时不能直接沿用“曾被预取”的位置，否则会漏训练样本。
如果在一个未保存的窗口中途失败，从上一个完整 checkpoint 恢复会重新执行此后未持久化的工作。

正式 TRL 路径通过原生 checkpoint 和固定 epoch 的 offset 恢复，不使用这个 commit 对象。
global_step × accumulation × batch_size 的 offset 只在其约束成立时适用。

## 保存物与可恢复能力

| 保存内容 | 能做什么 |
| --- | --- |
| final_model + tokenizer | 推理、评测，或作为新训练的模型起点 |
| 模型 + optimizer/scheduler/RNG/训练状态与数据身份 | 在相应实现支持下恢复原训练 |
| manifest、配置、日志和 hash | 审计实验身份与过程，不能替代参数或 optimizer 文件 |

本案例明确放弃了约 14GB 的 Trainer optimizer checkpoint 归档。
历史上恢复验证成功，不意味着现在仍有所有文件能从 step 479 精确续训。
已发布 final model 与审计证据应分别描述。

## 观察一次显存变化

```text
模型放置
-> optimizer 构造
-> micro-batch forward/backward
-> 第一次 optimizer.step
-> zero_grad
-> checkpoint 保存
```

allocated 是 allocator 当前活跃张量占用，reserved 是其保留池，nvidia-smi 还包含其他
设备开销。三者不能互换。AdamW 的 m/v 可能在第一次 step 才创建，所以只测 backward 不够。
峰值是历史最大值；summary 中记录到峰值的阶段不一定就是分配最初发生的精确位置。

当前 telemetry 对每次 train 调用的第一个累积窗口记录 micro-step 前后，其余阶段按
callback 记录。它不是每个 micro-batch 全程连续采样。
CUDA 调用、保存和异常路径的覆盖范围都要从代码确认。

## 测试与动手

```powershell
uv run python -m pytest tests/test_artifact_data.py -k sampler -q
uv run python -m pytest tests/test_checkpointing.py -q
uv run python -m pytest tests/test_trl_training.py -k cuda_memory -q
```

练习：用 5 个索引手画“预取 4 个，只提交 2 个，然后恢复”的顺序；
阅读已有 consumed-not-prefetched 测试验证推理。再给一组模拟 allocated/reserved 数值，
说明哪些结论有证据、哪些仍需后续采样。CPU mock 测试能验证 telemetry 字段与调用行为，
不能证明真实显存预算足够。

## 验收与学习记录

- [ ] 能区分推理导出与训练恢复 checkpoint。
- [ ] 能手推独立 sampler 和正式 TRL 各自的恢复位置。
- [ ] 能指出 soft stop、异常失败与计划暂停的区别及保存时机。
- [ ] 能解释一次 optimizer 首步 OOM 的可能原因。
- [ ] 能指出当前证据和已删除 checkpoint 的边界。

学习日期、独立完成部分、查询或提示、测试结果、剩余问题：待填写。

