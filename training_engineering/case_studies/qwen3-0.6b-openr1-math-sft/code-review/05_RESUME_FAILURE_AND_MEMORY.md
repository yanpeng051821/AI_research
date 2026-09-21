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

以冻结顺序 `[C, A, D, B, E]` 为例：DataLoader 已预取 `C/A/D/B` 时，sampler 的
`position=4`；若只有 `C/A` 所在的更新成功并执行 `commit(2)`，则
`committed_position=2`。checkpoint 必须保存 2，恢复后重新交付 `D/B/E`。保存 4 会把
仅被预取、尚未形成成功更新的 `D/B` 永久跳过。

完整恢复还需要两层协作：上层从 checkpoint 目录加载模型权重；
`load_training_checkpoint()` 恢复 optimizer、scheduler、sampler、TrainingState 和 RNG。
只执行其中一层都不是同一次训练的精确续训。

## 保存物与可恢复能力

| 保存内容 | 能做什么 |
| --- | --- |
| final_model + tokenizer | 推理、评测，或作为新训练的模型起点 |
| 模型 + optimizer/scheduler/RNG/训练状态与数据身份 | 在相应实现支持下恢复原训练 |
| manifest、配置、日志和 hash | 审计实验身份与过程，不能替代参数或 optimizer 文件 |

本案例明确放弃了约 14GB 的 Trainer optimizer checkpoint 归档。
历史上恢复验证成功，不意味着现在仍有所有文件能从 step 479 精确续训。

这里必须区分两类证据：恢复测试证明“当前代码在受控条件下具备恢复机制”；具体运行仍可
恢复则要求“该次运行的完整 checkpoint 现在仍然存在”。本案例保留的约 2.3GB final model
可用于推理、评测或作为新实验起点，但不能反推出已经删除的 AdamW m/v、scheduler、RNG
和 Trainer 状态。以它开始 DPO 或后续 SFT 属于新实验，不是恢复原 S1。

## 结束状态与保存语义

| 状态 | 触发方式 | 保存与恢复含义 |
| --- | --- | --- |
| `completed` | 达到完整训练计划 | 导出 final model，并按合同保存终态证据 |
| `paused` / 独立路径 `interrupted` | `stop_after_steps` 在完整更新边界触发 | 要求先保存 checkpoint，再正常退出训练循环 |
| `soft_stopped` | 训练保护器观察到危险趋势 | 独立路径先保存 checkpoint，再记录事件并抛出专用异常 |
| `failed` | 非有限值、OOM、I/O 或代码异常 | 留下 traceback/失败证据；不保证故障点本身可恢复 |

正式 TRL 的 loss、gradient 和 parameter 门禁分别发生在 backward 前、optimizer step 前和
参数更新后。异常发生时通常应从最近一个完整 checkpoint 恢复，而不是把可能处于半更新状态的
当前内存强行保存。`finally` 能为普通 Python 异常落盘 manifest 和显存摘要，但无法抵抗
`kill -9`、宿主机掉电或容器被直接销毁；此时 manifest 可能永久停在 `running`。

判断任务是否仍存活需要组合 PID、GPU process table、日志/metrics/telemetry 更新时间、
checkpoint 推进和外层 launcher 的退出状态，不能只看 manifest 或单个显存数字。

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

判断时使用以下边界：

- `allocated` 是当前活跃 PyTorch tensor 占用；持续跨窗口上升且 zero-grad 后基线不回落，
  才构成计算图泄漏嫌疑，还需控制 batch 形状并检查引用或 memory snapshot。
- `reserved` 是 PyTorch allocator 保留池，包含可复用但当前未被 tensor 使用的空间；不回落
  本身不等于泄漏。
- `peak_allocated` 是自 reset 后的历史峰值。在哪个 phase 读到峰值，不足以证明峰值最初
  就在该 phase 发生。
- `nvidia-smi` 还包括 reserved cache、CUDA context、库工作区和其他分配，不能当作活跃
  计算图大小。
- AdamW 的 m/v 可能在第一次 optimizer step 懒创建，所以只跑 forward/backward 不能证明
  正式训练显存安全；同时 allocator 可能复用 `reserved-allocated`，也不能仅凭接近满卡就
  断言首步一定 OOM。

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

- [x] 能区分推理导出与训练恢复 checkpoint。
- [x] 能手推独立 sampler 和正式 TRL 各自的恢复位置。
- [x] 能指出 soft stop、异常失败与计划暂停的区别及保存时机。
- [x] 能解释一次 optimizer 首步 OOM 的可能原因。
- [x] 能指出当前证据和已删除 checkpoint 的边界。

学习日期：2026-09-22。

本轮独立完成了 sampler 预取/提交位置、终态分类和显存证据题，并新增
`test_checkpoint_restores_rng_state`，验证 Python、NumPy、Torch CPU RNG 及 TrainingState
能够从 checkpoint 恢复。CUDA RNG 的实现存在，但本轮 CPU 测试没有验证真实 CUDA 状态；
新增测试会改变进程全局 RNG，测试隔离改进留到主题 07 统一处理。

验证结果：checkpoint/sampler 7 个测试、observability/显存门禁 4 个测试、独立恢复与合同
门禁 5 个测试、失败证据 1 个测试、正式 TRL 暂停恢复 3 个测试全部通过，Ruff 与
`git diff --check` 通过。Transformers 在当前文件系统上退化为按 checkpoint 数字排序的
mtime 警告不影响结果。
