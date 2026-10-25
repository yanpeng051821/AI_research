# MiniMind Dense：单设备训练链路短跑

> 2026-10-24：RTX 4060 Ti 8GB 上的 BF16 工程短跑完成，连续运行与暂停恢复对照通过。不是完整 mini 训练，也不是官方权重质量复现。

## 1. 实验目的

将已实现的模型、数据、loss、更新、验证、sampler 与 checkpoint 接入同一份配置驱动的入口，检查正式尺寸和候选长度下的完整参数更新、显存、周期保存及恢复。只运行 8 次 optimizer update；验证使用固定验证集前 64 条，共 12,469 个有效目标，不代表完整 5,000 条验证结果。

## 2. 固定设置

| 项目 | 本次选择 |
| --- | --- |
| 模型 | MiniMind Dense，63,912,192 独立参数，从随机初始化开始 |
| 数据 | split_seed43_v1，候选长度 768；不修改已接受的文本版本 |
| 计算 | 单卡 CUDA，FP32 参数与 AdamW 状态，BF16 autocast 前向，FP32 loss |
| batch | micro-batch=2，累积=4，有效窗口通常 8 条 |
| 优化器 | AdamW，lr=5e-4，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8；单组唯一参数，foreach/fused 关闭 |
| 调度 | 原定 8 个更新上的余弦衰减，最低倍率 0.1，无 warmup |
| loss 与裁剪 | 窗口有效 token 平均，梯度裁剪上限 1.0；日志 grad_norm 为裁剪前范数 |
| 保存/验证 | 每 4 个更新，另在暂停或结束边界保存并验证 |
| 恢复对照 | 连续 0→8；另一条 0→3 暂停，再 3→8 |

这是独立的工程短跑配方。累积 token 平均、按 optimizer update 的调度、有效 batch 和验证安排均明确采用本工程规则，不宣称与官方训练更新轨迹相同。全量正式预算、batch 和保存验证频率仍待确定，不能把 max_updates 改大后冒充原合同的延续。

## 3. 实际结果

| 检查 | 结果 |
| --- | ---: |
| 完成 optimizer updates | 8 |
| 提交训练记录 | 64 |
| 累计有效训练目标 | 15,597 |
| 连续路径排除首步后的更新耗时中位数 | 0.37985 秒 |
| 连续路径峰值 PyTorch allocated | 2.47088 GiB |
| 初始固定子集 NLL | 8.909696 |
| 结束固定子集 NLL | 7.845094 |
| 恢复对照 | 363 个 tensor 叶子精确相等，其他训练状态及逐步指标相等 |

本机连续与恢复路径的最终 checkpoint 字节 hash 也相同。但验收首先比较实际模型、优化器、调度器、sampler、CPU/CUDA RNG 和元数据，不将一般情况下的 checkpoint 序列化字节相同设为必要条件。这里只证明当前机器、当前配置与已测试步数，不保证跨硬件或任意长时间训练的逐位一致。

NLL 下降说明这个固定小子集上的计算与早期学习响应正常，不能证明模型已具备自然语言能力或完整数据上的泛化改善。0.38 秒只对应本次 micro-batch/累积设置的更新计算，不包括完整启动、周期验证及保存时间，不是全量长跑的耗时承诺。

## 4. 小型证据

- [contract.json](contract.json)：解析后的完整设置、数据与编码身份、代码及依赖身份。
- [paused_result.json](paused_result.json)：第 3 步的主动暂停结果。
- [resumed_result.json](resumed_result.json)：从第 3 步继续到原预算第 8 步。
- [continuous_result.json](continuous_result.json)：从头连续运行到第 8 步。
- [comparison.json](comparison.json)：独立状态与逐步指标比较。
- [continuous_metrics.jsonl](continuous_metrics.jsonl)、[paused_metrics.jsonl](paused_metrics.jsonl)、[resumed_metrics.jsonl](resumed_metrics.jsonl)：实际更新、验证与保存事件。

检查点保留在 implementation/runs/pretrain_mini_probe_*_v1/，不复制到 Git。归档报告不能替代这些真实 checkpoint 的备份。默认候选配置见 [pretrain_mini_probe.json](../../implementation/configs/pretrain_mini_probe.json)，实现与执行教程见[第七周实践](../../practice_notes/week_07_pretrain_experiment.md)。

## 5. 核验范围

新增 CPU 回归覆盖连续/恢复的尾部窗口、数据与配置变化拒绝、预检、保存失败状态及无效参数；保留旧 smoke 接口。2026-10-24 全项目 317 passed。实际 CUDA 路径则由本次完整尺寸短跑与独立对照验证，不能仅用 CPU 测试替代。

当前入口限制为单设备、num_workers=0、单次数据遍历内的预算；支持 FP32 或 CUDA BF16，不提供 FP16 scaler、多 epoch 或分布式训练。硬件故障或强制 kill 仍可能留下 running 状态，需要结合日志、进程和 checkpoint 判断，不能只依赖 manifest。
