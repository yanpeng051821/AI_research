# 04 TRL 正式入口与框架职责

## 本次要解决的问题

既然成熟框架负责训练，我们为什么还写 FrozenOrderSFTTrainer？每个 override 改变了什么，
移除之后会失去哪一条合同？哪些测试证明它确实与独立实现对齐？

对应案例文档：上一级 04、07、09。先完成主题 03，再深入框架调用。

## 代码阅读顺序

| 文件 | 阅读重点 |
| --- | --- |
| [train_sft_trl.py](D:/pythonlearning/small_model_post_training/independent_implementation/scripts/train_sft_trl.py) | model、dataset、collator、callback 的装配 |
| [trl_reference.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/trl_reference.py) | labels 转 completion_mask，固定样本顺序 |
| [trl_training.py](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/trl_training.py) | build_trl_sft_args 与所有 override |
| [test_reference_alignment.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_reference_alignment.py) | 基础数值合同 |
| [test_trl_training.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_trl_training.py) | tail update、原生恢复、正式 CLI |

## 框架职责地图

```text
项目：artifact、顺序、配置身份、adapter、callback
-> TRL：SFTTrainer 与 completion-only collator
-> Transformers：Trainer 循环、optimizer/scheduler、原生 checkpoint
-> Accelerate：按当前配置协助设备与训练执行
-> PyTorch：张量、autograd、optimizer、CUDA
```

这是职责地图，具体某一步在哪一层执行，应沿锁定版本的源码确认。
不要把所有训练动作归给 TRL，也不要假设我们的 engine 在 Trainer 内部运行。

正式模型收到 labels 后，框架/模型 loss 路径可以计算 causal LM loss。
独立路径则取 logits 后在外部计算。两者需要通过 shift、mask、reduction 和梯度对照，
不能只凭“都用了交叉熵”断定相同。

## 从启动到一次参数更新

正式 S1 的主调用链是：

```text
scripts/train_sft_trl.py::main
-> 读取并校验 ExperimentConfig
-> 构造冻结顺序的 Dataset、collator、model 和 TrainingArguments
-> 构造 FrozenOrderSFTTrainer
-> trainer.train(resume_from_checkpoint=...)
-> Transformers Trainer 创建 DataLoader 并进入训练循环
-> training_step
-> compute_loss
-> model(**inputs)
-> Qwen3ForCausalLM.forward(labels=...)
-> Transformers ForCausalLMLoss
-> Accelerate.backward
-> 梯度累积、裁剪、optimizer.step、scheduler.step、zero_grad
-> callback、checkpoint 与结束收尾
```

`train()` 仍然拥有完整循环。`compute_loss()` 不是另一个训练入口，而是 Trainer 在每个
micro-batch 内调用的一个职责节点：它取得标量 loss，供 `training_step()` 反向传播。
当前自定义 `compute_loss()` 调用父类实现，只额外检查 loss 是否有限；真正的 causal shift
和交叉熵由模型/Transformers 的 causal-LM loss 路径完成。

DataLoader 也不是入口脚本显式创建的。父类 `Trainer.train()` 会经
`get_train_dataloader()` 和 `_get_dataloader()` 构造它，再由 Accelerate 做设备和运行时适配。
因此“代码里没直接写 DataLoader”不表示不存在数据加载循环。

Accelerate 主要承担设备放置、混合精度、反向传播、梯度同步/裁剪和模型解包等运行时工作；
它不决定本项目的数据合同、SFT 监督位置、样本顺序或评测口径。

## 逐项解释定制点

1. 固定 Dataset 索引与 RemainingIndices：让训练和恢复消费可追踪的顺序。
2. train：从 trainer_state.json 读取 global_step，计算本 epoch 剩余位置。
3. set_initial_training_values：恢复后的 DataLoader 只暴露剩余数据，但保持原始调度与末尾更新语义。
4. compute_loss：调用父类后检查有限性；它不是自己重写一套 SFT loss。
5. training_step、create_optimizer、evaluate、_save_checkpoint：围绕父类行为记录显存。
6. SaveAndStopCallback：检查梯度/参数有限性，在暂停或最后一步要求保存。

恢复公式基于固定 batch、固定累积、单 epoch 和更新边界 checkpoint。
不能机械用于多 epoch、多卡或动态 batch。私有方法 override 与框架版本有关，
升级后需要重新验证签名和行为。

### 恢复职责的边界

TRL/Transformers 负责恢复模型、optimizer、scheduler、global step 和 RNG 等通用训练状态。
本项目额外接管“下一条该消费哪条样本”：

```text
checkpoint global_step
-> 计算 _resume_offset
-> 父类通过动态分派调用我们的 _get_train_sampler()
-> RemainingIndices 只暴露冻结顺序中尚未消费的逻辑位置
```

`_resume_offset` 不是 Trainer 内置变量；只有因为 `_get_train_sampler()` 主动读取它，才会影响
训练。随机顺序早已冻结在 `OrderedCompletionMaskDataset` 中，`RemainingIndices` 只是对这份
逻辑顺序取后缀，不会再次随机化。

如果 dataset、顺序和全部相关配置严格不变，原生 Trainer 的 data skip 很可能也能恢复到
正确位置。本项目的后缀 sampler 是为了把位置变成可审计合同，并精确覆盖单 epoch 的 tail
accumulation；它不是在证明框架原生恢复一定错误。该方案同时带来私有方法耦合，升级框架、
扩展到多 epoch、多卡、packing 或动态 batch 时必须重新验证，不能直接照搬。

恢复后 DataLoader 只含剩余样本，但 scheduler 和训练终点仍应属于原始完整训练计划。
`set_initial_training_values()` 的定制正是为了避免把“剩余后缀”误当成一场新的短训练。

## 需要辨别的配置

build_trl_sft_args 当前显式设置 packing=False、skip_prepare_dataset=True、
completion_only_loss=True、ignore_data_skip=True、eval_strategy="no"。
分别解释为什么 artifact 不应再次自动加工、谁处理数据跳过、为什么正式训练不会自动
调用 validation NLL。该 NLL 在外部评测入口完成。

optimizer 参数分组和 loss 归一化要看锁定实现及测试。独立 build_optimizer 使用一组参数，
Trainer 默认行为可能不同；weight_decay 为零时部分分组差异未必改变本次数值。

`sft_smoke.yaml` 属于独立训练入口，不是正式 TRL S1 配置。其中 128 条训练样本、batch 1、
累积 128 次，构成每个 epoch 一个 optimizer step；20 个 epoch 和 `max_steps=20` 的目标是
反复验证更新、保存、停止和恢复，而不是用 smoke 估计模型效果。

## 回调与观测时机

- `create_optimizer`：记录 optimizer 创建后的状态；AdamW 的 m/v 可能到第一次 step 才懒创建。
- `training_step`：在父类 micro-batch 行为外加遥测，`finally` 保证异常时也留下观察记录。
- `on_pre_optimizer_step`：在全部 backward 和梯度裁剪后、参数更新前检查梯度有限性。
- `on_optimizer_step`：optimizer 更新完成后的框架事件。
- `on_step_end`：参数更新和 global step 推进后检查参数，并设置保存或停止控制信号。
- `_save_checkpoint`：仍由父类执行真实保存，我们只包裹遥测。

回调设置 `control.should_save=True` 只表示“本轮应保存”，并不表示文件已经写入；父类训练循环
随后读取控制信号并执行保存。`evaluate()` 的观测封装在正式 S1 中不会被调用，因为配置明确
使用 `eval_strategy="no"`，validation NLL 和冻结外部评测走独立入口。

## 测试与动手

```powershell
uv run python -m pytest tests/test_trl_training.py -k "formal_plan or native_trl_tail" -q
```

先读 test_native_trl_tail_update_and_resume_match_independent 的小模型和 fixture，再运行。
练习：手算 5 条记录、每批 1 条、累积 2 次的一轮更新数和最后窗口大小；
用现有 test_formal_plan_keeps_tail_and_rounds_warmup 的结构检查另一组边界数据。
不在不了解调用合同前修改私有 override。

## 验收与学习记录

- [x] 能逐项解释 Trainer 定制点，而非只解释类名。
- [x] 能说清独立实现与正式路径共享和不共享的部分。
- [x] 能指出单卡、单 epoch 和版本限制。
- [x] 能用测试证据解释 tail update 和恢复对齐。
- [x] 能说明升级 TRL/Transformers 时需要重验哪些合同。

学习日期：2026-09-21。

本轮从入口追踪了 Trainer、模型 loss、DataLoader、Accelerate、恢复和 callback 的实际调用链，
并独立回答了配置、tail accumulation、恢复位置、梯度/参数检查时机等问题。新增
`test_save_and_stop_callback_rejects_non_finite_gradient`，用真实参数梯度验证非有限值门禁。

验证结果：`tests/test_trl_training.py` 8 个测试通过；
`tests/test_reference_alignment.py` 与 `tests/test_shadow.py` 合计 3 个测试通过；Ruff 检查通过。
剩余的系统性 pytest 学习统一放到主题 07，不阻塞主题 04 验收。
