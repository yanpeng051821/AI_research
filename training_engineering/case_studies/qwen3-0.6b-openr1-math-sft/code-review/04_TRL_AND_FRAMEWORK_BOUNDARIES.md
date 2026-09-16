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

## 需要辨别的配置

build_trl_sft_args 当前显式设置 packing=False、skip_prepare_dataset=True、
completion_only_loss=True、ignore_data_skip=True、eval_strategy="no"。
分别解释为什么 artifact 不应再次自动加工、谁处理数据跳过、为什么正式训练不会自动
调用 validation NLL。该 NLL 在外部评测入口完成。

optimizer 参数分组和 loss 归一化要看锁定实现及测试。独立 build_optimizer 使用一组参数，
Trainer 默认行为可能不同；weight_decay 为零时部分分组差异未必改变本次数值。

## 测试与动手

```powershell
uv run python -m pytest tests/test_trl_training.py -k "formal_plan or native_trl_tail" -q
```

先读 test_native_trl_tail_update_and_resume_match_independent 的小模型和 fixture，再运行。
练习：手算 5 条记录、每批 1 条、累积 2 次的一轮更新数和最后窗口大小；
用现有 test_formal_plan_keeps_tail_and_rounds_warmup 的结构检查另一组边界数据。
不在不了解调用合同前修改私有 override。

## 验收与学习记录

- [ ] 能逐项解释 Trainer 定制点，而非只解释类名。
- [ ] 能说清独立实现与正式路径共享和不共享的部分。
- [ ] 能指出单卡、单 epoch 和版本限制。
- [ ] 能用测试证据解释 tail update 和恢复对齐。
- [ ] 能说明升级 TRL/Transformers 时需要重验哪些合同。

学习日期、独立完成部分、查询或提示、测试结果、剩余问题：待填写。

