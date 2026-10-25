# MiniMind mini 单轮预训练合同

> 合同 ID：`pretrain_mini_one_epoch_v1`。状态：已确定，实际冻结身份见同目录 `training_freeze.json`。日期：2026-10-25。

## 1. 实验目标

使用自己的单设备训练工程，从随机初始化的 MiniMind-3 Dense 出发，在固定 mini 预训练数据上完成一个 epoch。实验检验两件事：训练及恢复链路能否可靠完成；模型在未参与更新的同分布验证文本上的 token 加权 NLL 是否低于随机初始化基线。

本轮的任务能力验收范围是同分布 next-token prediction。聊天、事实可靠性、指令遵循和外部分布泛化没有独立评测，不能由 NLL 下降推断。

## 2. 数据引用

本合同引用已冻结的[数据合同](../pretrain_mini_preparation/split_seed43_v1/DATA_CONTRACT.md)和[数据冻结清单](../pretrain_mini_preparation/split_seed43_v1/data_contract_freeze.json)，不另建第二套数据定义。

| 项目 | 固定值 |
| --- | --- |
| 数据合同 ID | pretrain_mini_data_v1 |
| 来源 | jingyaogong/minimind_dataset，固定 revision 的 pretrain_t2t_mini.jsonl |
| 训练记录 | 1,265,235 |
| 验证记录 | 5,000 |
| 训练/验证有效 causal target | 301,810,690 / 1,206,758 |
| tokenizer | 本地冻结的官方 MiniMind tokenizer，词表 6,400 |
| 编码 | BOS + 正文前 766 个 token + EOS，右侧 PAD 至 768 |
| 监督 | causal shift 后全部非 -100 target；PAD 忽略，EOS 参与 |
| 数据切分 | seed=43，按完整正文 SHA256 分组隔离，保留组内重复 |

来源、人工排除、文本切分、tokenizer、截断和 padding 的细节与限制以数据合同为准。完整事实核查、近似去重和原始文档级隔离未完成，属于本轮已接受的数据风险。

## 3. 模型与起点

| 项目 | 固定值 |
| --- | --- |
| 模型 | MiniMind-3 Dense，本地实现 |
| hidden / FFN intermediate | 768 / 2,432 |
| 层数 / Q heads / KV heads | 8 / 8 / 4 |
| Q/K Norm | 开启，epsilon=1e-6 |
| RoPE base | 1,000,000 |
| 词表权重 | embedding 与 LM head 共享 |
| 独立参数数 | 63,912,192 |
| 初始化 | CPU FP32 构造，model seed=42，不加载训练权重 |
| 初始状态 SHA256 | 272d31cc9061d16cf6a00f2486ba97de6777eda428fcd08c061faeb10cde1ab4 |

初始化哈希覆盖有序 state_dict 的名称、dtype、形状与实际字节。正式启动先在 CPU 重建并核对，再调用短跑验证过的原训练入口；seed 相同本身不作为权重一致的唯一证据。

## 4. 更新规则

| 项目 | 固定值 |
| --- | --- |
| 训练设备 | 本机单张 RTX 4060 Ti，环境身份记录在冻结清单 |
| 参数、梯度、AdamW m/v | FP32 |
| 主要矩阵计算 | CUDA BF16 autocast |
| RoPE 角度、Norm 统计、Softmax、loss | 明确 FP32 计算边界 |
| TF32 / GradScaler | 关闭 / 不使用 |
| 确定性算法 | 开启 |
| sampler seed / workers | 42 / 0 |
| micro-batch / accumulation | 4 / 64，有效 batch=256 条记录 |
| optimizer | AdamW，全部唯一可训练参数一个参数组 |
| lr / betas / eps / weight decay | 5e-4 / (0.9, 0.999) / 1e-8 / 0.01 |
| foreach / fused | false / false |
| 梯度 | 累积各 micro-batch loss sum 的梯度，再除以本窗口有效 target 总数 |
| 裁剪 | 全局 L2 norm，max_grad_norm=1.0，记录裁剪前 norm |
| scheduler | 无 warmup，按 optimizer update 余弦衰减至 5e-5 |

模型前向和 loss 接到已有 train_update；每个完整更新后提交 sampler 位置，再增加 global_step。调度器在 optimizer.step 之后推进，日志区分本次 lr_used 与下次 next_lr。

## 5. 预算与完成条件

一个 epoch 精确定义为冻结 sampler 排列中的所有训练记录各提交一次：

```text
总更新数 = ceil(1,265,235 / 256) = 4,943
完整更新 = 4,942 次
最后一次更新 = 83 条记录，不丢弃、不重复填充
最终 committed_position = 1,265,235
最终 valid_tokens_seen = 301,810,690
```

最后一个窗口仍按其自身有效 target 总数平均。配置中 max_updates=null 表示由上述数据量解析预算，不表示无限训练。第 1/2,472/4,943 次参数更新使用学习率 5e-4 / 2.75e-4 / 5e-5。

## 6. 评测与判定

完整验证固定使用全部 5,000 条、batch=4、BF16 autocast、相同 tokenizer 与监督规则。在 step 0、每 500 步至 4,500、最终 4,943 计算：

```text
validation NLL = 全验证集有效 target 的 loss_sum / 1,206,758
```

短跑核对的随机初始化 NLL 为 8.912246169216113，正式启动需要保存自己实测的初始基线。恢复时增加一次诊断验证，其范围与正式验证一致。

本轮预先确定的质量判定是最终完整验证 NLL 严格低于本次初始 NLL；不设实用模型能力或达到官方模型质量的阈值，也不把这个方向性条件当作统计显著性检验。达成后报告实际差值，未达成则记录目标未达成，不能事后换验证集或阈值。

工程完成还要求数据/配置/初始化身份相符、完整消费一个 epoch、最后 checkpoint 与 receipt 可验证、日志与结果齐全。非有限 loss/梯度、OOM 或数据异常应停止诊断；有故障后可在保持合同的条件下从有效 checkpoint 恢复，故障和重算不能从报告中省略。完整训练结束才可标记 completed。

## 7. 保存与恢复

日志、运行合同快照与结果使用 `implementation/runs/pretrain_mini_one_epoch_v1`；checkpoint 使用 `E:\AI_research_checkpoints\minimind\pretrain_mini_one_epoch_v1`。

每 100 步及最终 4,943 保存，共 50 个，预计约 36 GiB，全部保留。先写同盘临时文件，经 flush、fsync 后原子替换，再生成文件 SHA256 receipt；已存在的目标拒绝覆盖。主动暂停只在完整更新后额外保存，成功后记 paused；可捕获异常记 failed，强制退出可能留下陈旧 running。

恢复依赖已通过只读检查的 checkpoint 与 receipt，不能把 manifest 的最后 step 直接当作可恢复进度。加载完整 model、optimizer、scheduler、sampler、CPU/CUDA RNG、global_step 与 valid_tokens_seen，继续原 4,943-step 计划；新的启动使用独立日志目录。路径变化不替代内容身份核验。

## 8. 证据与冻结

[正式候选短跑](../pretrain_mini_resume_gate_v1/README.md)在同一核心运行代码和配置下通过连续 4 步与 2+2 恢复的完整状态比较；两份 checkpoint 字节 SHA256 相同。后续至 step 20 的稳定性观察显示 NLL 降至 7.09767，step 8～20 不再裁剪。这些证据支持开始正式实验，没有覆盖整轮或跨硬件精确恢复。

`training_config.json` 是可执行输入；`contract.json` 是现有入口解析出的配置、运行代码与依赖身份；`training_freeze.json` 固定这些文件、本文、数据合同、初始化身份、依赖锁和短跑证据。冻结启动器核对后调用原 run_experiment；它没有改写 forward、backward、参数更新或恢复算法。

此前通过的核心合同 SHA256 为 `b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58`。冻结启动器自身及本文由外层冻结清单另外固定，避免把新门禁误称为此前短跑已经覆盖的实现。

## 9. 成本与变更

正式短跑外推约 16.3 小时，包含数据准备及参数更新、完整验证和 checkpoint。它是预算估计；恢复、启动和系统波动会增加时间，不作为训练完成阈值。

改变数据、tokenizer、模型、初始化、优化策略、更新预算、评测口径或核心运行代码，应建立新版本并重新评估证据；不能原地改文件后仍声称同一冻结实验。每次运行的 manifest、metrics、checkpoint 是执行产物，随进度变化，不需要预先冻结其未知内容。新配置的算法对照另行设计。
